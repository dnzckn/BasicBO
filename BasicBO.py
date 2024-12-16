import asyncio
import random

import numpy as np
import torch
from scipy.interpolate import griddata

from ax.core.observation import ObservationFeatures
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.acquisition import qNegIntegratedPosteriorVariance
from botorch.models.gp_regression import SingleTaskGP


class ZReader:
    """A synthetic response object simulating a noisy Gaussian-like surface as the objective.
    In reality you would replace with your actual objective.

    E.g., for solar cells, you could seek to optimize the cell's measured
    open-circuit voltage as a function of its chemistry.
    """

    def __init__(self, centers, sigma=0.1, n_samples=1):
        """
        Initialize the Gaussian response surface.

        Args:
            centers (list): The "peak" of the Gaussian for each parameter.
            sigma (float): Spread of the Gaussian.
            n_samples (int): Number of samples to simulate measurement noise.
        """
        self.centers = np.array(centers)
        self.sigma = sigma
        self.n_samples = n_samples

    def read(self, params):
        """
        Compute the objective value (mean and SEM if multiple samples).

        Args:
            params (list): Parameter values at which to evaluate.
        Returns:
            tuple: (mean, sem) representing the evaluated objective and its SEM.
        """
        params = np.array(params)
        # Gaussian-like function
        z = np.exp(-np.sum((params - self.centers) ** 2) / (2 * self.sigma**2))
        if self.n_samples > 1:
            # Add noise for multiple samples and compute mean and SEM
            noisy_z = [
                z + np.random.normal(0, self.sigma) for _ in range(self.n_samples)
            ]
            mean_z = np.mean(noisy_z)
            sem_z = np.std(noisy_z) / np.sqrt(self.n_samples)
            return mean_z, sem_z
        return z, 0.0


async def get_next_batch_async(ax_client, batch_size):
    """Asynchronously request the next batch of trials from AxClient."""
    loop = asyncio.get_event_loop()
    return await asyncio.gather(
        *[
            loop.run_in_executor(None, ax_client.get_next_trial)
            for _ in range(batch_size)
        ]
    )


async def complete_trial_async(ax_client, trial_index, result):
    """Asynchronously mark a given trial as completed with results."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, ax_client.complete_trial, trial_index, result
    )


async def bayesian_optimizer_iterator_async(
    z_reader,
    param_names,
    param_bounds,
    num_initial_samples=10,
    trial_budget=20,
    batch_size=1,
    explore_ratio=0.1,
    threshold=None,
    epsilon=0.0,
    patience=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generation_strategy = GenerationStrategy(
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=num_initial_samples),
            GenerationStep(
                model=Models.GPEI, num_trials=-1, max_parallelism=batch_size
            ),
        ]
    )

    ax_client = AxClient(generation_strategy=generation_strategy, torch_device=device)
    ax_client.generation_strategy.trials_as_df
    ax_client.create_experiment(
        name="modular_iterator_optimization",
        parameters=[
            {"name": name, "type": "range", "bounds": bounds}
            for name, bounds in zip(param_names, param_bounds)
        ],
        objectives={"z": ObjectiveProperties(minimize=False)},
        parameter_constraints=[],
        tracking_metric_names=[],
    )

    best_value = float("-inf")
    no_improvement_count = 0

    # Generate initial Sobol samples
    trials = await get_next_batch_async(ax_client, num_initial_samples)
    for params, trial_index in trials:
        yield params  # yield parameter dictionary
        z_mean, z_sem = z_reader.read([params[name] for name in param_names])
        await complete_trial_async(ax_client, trial_index, {"z": (z_mean, z_sem)})
        if z_mean > best_value:
            best_value = z_mean
            no_improvement_count = 0

    last_best_value = best_value

    steps_taken = 0
    while steps_taken < trial_budget:
        current_batch_size = min(batch_size, trial_budget - steps_taken)

        # Decide if we explore (qNIPV) or exploit (GPEI)
        explore = np.random.rand() < explore_ratio
        if explore:
            print("~ Using qNIPV (Exploration)")
            ax_client.generation_strategy._steps[-1].model = Models.BOTORCH_MODULAR
            ax_client.generation_strategy._steps[-1].model_kwargs = {
                "surrogate": Surrogate(SingleTaskGP),
                "botorch_acqf_class": qNegIntegratedPosteriorVariance,
                "acquisition_options": {"mc_points": 50},
            }
        else:
            print("$ Using GPEI (Exploitation)")
            ax_client.generation_strategy._steps[-1].model = Models.GPEI
            ax_client.generation_strategy._steps[-1].model_kwargs = {}

        trials = await get_next_batch_async(ax_client, current_batch_size)
        for params, trial_index in trials:
            yield params  # yield parameter dictionary
            z_mean, z_sem = z_reader.read([params[name] for name in param_names])
            await complete_trial_async(ax_client, trial_index, {"z": (z_mean, z_sem)})

            # Check improvement
            if z_mean > best_value:
                best_value = z_mean
                # Reset patience only if we are in exploitation mode
                if not explore:
                    no_improvement_count = 0
            else:
                # If exploitation step and no improvement >= epsilon
                if not explore and best_value - last_best_value < epsilon:
                    no_improvement_count += 1

            last_best_value = best_value

            # Early stopping checks
            if threshold is not None and best_value > threshold:
                print("Stopping early: threshold exceeded.")
                # Yield the final model and stop
                yield {"final_model": ax_client.generation_strategy.model}
                return

            if not explore and no_improvement_count > patience:
                print(
                    "Stopping early: no sufficient improvement in exploitation steps."
                )
                # Yield the final model and stop
                yield {"final_model": ax_client.generation_strategy.model}
                return

        steps_taken += 1
        if steps_taken == trial_budget:
            print(f"Trial budget met")

    # If we exit the loop naturally, yield the final model
    yield {"final_model": ax_client.generation_strategy.model}


def bayesian_optimizer_iterator_sync(
    z_reader,
    param_names,
    param_bounds,
    num_initial_samples=10,
    trial_budget=20,
    batch_size=1,
    explore_ratio=0.1,
    threshold=None,
    epsilon=0.0,
    patience=10,
):
    """
    A synchronous version of the Bayesian optimization iterator with early stopping.
    Also yields the final model at the end.

    Args:
        z_reader: Object with a .read() method returning (mean, sem).
        param_names (list): Parameter names.
        param_bounds (list): Parameter bounds as (min, max).
        num_initial_samples (int): Number of initial Sobol samples.
        trial_budget (int): Maximum number of BO steps.
        batch_size (int): Trials per batch.
        explore_ratio (float): Probability of switching to qNIPV exploration.
        threshold (float or None): If best observed value > threshold, stop early.
        epsilon (float): Minimum absolute improvement to reset patience in exploitation steps.
        patience (int): Number of exploitation steps allowed without improvement.

    Yields:
        dict: Parameter sets suggested by the optimizer for evaluation.
              At the end, yields {"final_model": model_bridge} containing the final model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generation_strategy = GenerationStrategy(
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=num_initial_samples),
            GenerationStep(
                model=Models.GPEI, num_trials=-1, max_parallelism=batch_size
            ),
        ]
    )

    ax_client = AxClient(generation_strategy=generation_strategy, torch_device=device)
    ax_client.create_experiment(
        name="modular_iterator_optimization_sync",
        parameters=[
            {"name": name, "type": "range", "bounds": bounds}
            for name, bounds in zip(param_names, param_bounds)
        ],
        objectives={"z": ObjectiveProperties(minimize=False)},
        parameter_constraints=[],
        tracking_metric_names=[],
    )

    best_value = float("-inf")
    no_improvement_count = 0

    # Initial Sobol samples
    for _ in range(num_initial_samples):
        params, trial_index = ax_client.get_next_trial()
        yield params
        z_mean, z_sem = z_reader.read([params[name] for name in param_names])
        ax_client.complete_trial(trial_index, {"z": (z_mean, z_sem)})
        if z_mean > best_value:
            best_value = z_mean
            no_improvement_count = 0

    last_best_value = best_value

    # Main BO loop: up to trial_budget steps
    for step in range(trial_budget):
        current_batch_size = min(batch_size, trial_budget - step)

        explore = np.random.rand() < explore_ratio
        if explore:
            print("~ Using qNIPV (Exploration)")
            ax_client.generation_strategy._steps[-1].model = Models.BOTORCH_MODULAR
            ax_client.generation_strategy._steps[-1].model_kwargs = {
                "surrogate": Surrogate(SingleTaskGP),
                "botorch_acqf_class": qNegIntegratedPosteriorVariance,
                "acquisition_options": {"mc_points": 50},
            }
        else:
            print("$ Using GPEI (Exploitation)")
            ax_client.generation_strategy._steps[-1].model = Models.GPEI
            ax_client.generation_strategy._steps[-1].model_kwargs = {}

        for _ in range(current_batch_size):
            params, trial_index = ax_client.get_next_trial()
            yield params
            z_mean, z_sem = z_reader.read([params[name] for name in param_names])
            ax_client.complete_trial(trial_index, {"z": (z_mean, z_sem)})

            if z_mean > best_value:
                best_value = z_mean
                # Reset patience only if in exploitation
                if not explore:
                    no_improvement_count = 0
            else:
                # If exploitation step and no improvement >= epsilon
                if not explore:
                    if best_value - last_best_value < epsilon:
                        no_improvement_count += 1

            last_best_value = best_value

            # Early stopping checks
            if threshold is not None and best_value > threshold:
                print("Stopping early: threshold exceeded.")
                # Yield final model
                yield {"final_model": ax_client.generation_strategy.model}
                return

            if not explore and no_improvement_count > patience:
                print(
                    "Stopping early: no sufficient improvement in exploitation steps."
                )
                # Yield final model
                yield {"final_model": ax_client.generation_strategy.model}
                return
        if step == trial_budget - 1:
            print(f"Trial budget met")

    # If we exit the loop naturally, yield the final model
    yield {"final_model": ax_client.generation_strategy.model}
