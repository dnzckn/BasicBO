import numpy as np
from typing import List, Tuple, Dict, Union, Iterator
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import torch


class SyntheticGaussian:
    """A synthetic response object simulating a noisy Gaussian-like surface as the objective.
    Replace this with your actual objective function.
    """

    def __init__(self, centers: List[float], sigma: float = 0.1, n_samples: int = 1):
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

    def read(self, params: List[float]) -> Tuple[float, float]:
        """
        Compute the objective value (mean and SEM if multiple samples).

        Args:
            params (list): Parameter values at which to evaluate.
        Returns:
            tuple: (mean, sem) representing the evaluated objective and its SEM.
        """
        params = np.array(params)
        # Gaussian-like function
        objective = np.exp(-np.sum((params - self.centers) ** 2) / (2 * self.sigma**2))
        if self.n_samples > 1:
            # Add noise for multiple samples and compute mean and SEM
            noisy_objective = [
                objective + np.random.normal(0, self.sigma)
                for _ in range(self.n_samples)
            ]
            mean_objective = np.mean(noisy_objective)
            sem_objective = np.std(noisy_objective) / np.sqrt(self.n_samples)
            return mean_objective, sem_objective
        return objective, 0.0


class BayesianOptimizerIterator:
    def __init__(
        self,
        objective_function: SyntheticGaussian,
        param_names: List[str],
        param_bounds: List[Tuple[float, float]],
        num_sobol: int = 20,
        num_gpei: int = 30,
        batch_size: int = 2,
        threshold: Union[float, None] = None,
        epsilon: float = 0.001,
        patience: int = 20,
        maximize: bool = True,
    ):
        """
        Initialize the Bayesian Optimizer Iterator.

        Args:
            objective_function: Object with a .read() method returning (mean, sem).
            param_names (list): Parameter names.
            param_bounds (list): Parameter bounds as (min, max).
            num_sobol (int): Number of initial Sobol samples.
            num_gpei (int): Number of GPEI trials.
            batch_size (int): Trials per batch.
            threshold (float or None): If best observed value > threshold, stop early.
            epsilon (float): Minimum absolute improvement to reset patience in GPEI steps.
            patience (int): Number of GPEI steps allowed without improvement.
            maximize (bool): Whether to maximize or minimize the objective.
        """
        self.objective_function = objective_function
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.num_sobol = num_sobol
        self.num_gpei = num_gpei
        self.num_trials = num_sobol + num_gpei  # Calculate total number of trials
        self.batch_size = batch_size
        self.threshold = threshold
        self.epsilon = epsilon
        self.patience = patience
        self.maximize = maximize

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Define Generation Strategy with Sobol and GPEI steps
        steps = []
        if self.num_sobol > 0:
            steps.append(
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=self.num_sobol,
                    min_trials_observed=self.num_sobol,
                )
            )
        if self.num_gpei > 0:
            steps.append(
                GenerationStep(
                    model=Models.GPEI,
                    num_trials=self.num_gpei,
                    max_parallelism=self.batch_size,
                    model_kwargs={"torch_device": self.device},
                )
            )

        self.generation_strategy = GenerationStrategy(steps=steps)

        # Initialize AxClient
        self.ax_client = AxClient(generation_strategy=self.generation_strategy)
        self.ax_client.create_experiment(
            name="sobol_gpei_optimization",
            parameters=[
                {"name": name, "type": "range", "bounds": bounds}
                for name, bounds in zip(param_names, param_bounds)
            ],
            objectives={"objective": ObjectiveProperties(minimize=not maximize)},
            parameter_constraints=[],
            tracking_metric_names=[],
            overwrite_existing_experiment=True,  # Ensure fresh experiment
        )

        # Initialize tracking variables
        self.best_value = float("-inf") if maximize else float("inf")
        self.no_improvement_count = 0
        self.last_best_value = self.best_value
        self.steps_taken = 0

    def evaluate_objective(self, params: Dict[str, float]) -> Tuple[float, float]:
        """Evaluate the objective function."""
        return self.objective_function.read([params[name] for name in self.param_names])

    def get_next_batch(self) -> List[Tuple[Dict[str, float], int]]:
        """Request the next batch of trials from AxClient."""
        return [self.ax_client.get_next_trial() for _ in range(self.batch_size)]

    def complete_trial(
        self, trial_index: int, result: Dict[str, Tuple[float, float]]
    ) -> None:
        """Mark a given trial as completed with results."""
        self.ax_client.complete_trial(trial_index, result)

        # Update best value
        objective_mean = result["objective"][0]
        if (self.maximize and objective_mean > self.best_value) or (
            not self.maximize and objective_mean < self.best_value
        ):
            self.best_value = objective_mean
            self.no_improvement_count = 0
        else:
            current_model = self.generation_strategy.current_step.model
            if current_model == Models.GPEI:
                if abs(objective_mean - self.best_value) < self.epsilon:
                    self.no_improvement_count += 1

    def __iter__(self) -> Iterator[Dict[str, Union[Dict[str, float], str]]]:
        """Return an iterator."""
        return self._iterator()

    def _iterator(self) -> Iterator[Dict[str, Union[Dict[str, float], str]]]:
        """Iterator for Bayesian Optimization."""
        total_trials = 0

        while total_trials < self.num_trials:
            trials = self.get_next_batch()
            for params, trial_index in trials:
                yield params

                # Evaluate the objective function
                objective_mean, objective_sem = self.evaluate_objective(params)

                # Complete the trial
                self.complete_trial(
                    trial_index, {"objective": (objective_mean, objective_sem)}
                )

                # Check if threshold is exceeded
                if self.threshold is not None and (
                    (self.maximize and self.best_value > self.threshold)
                    or (not self.maximize and self.best_value < self.threshold)
                ):
                    print("Stopping early: threshold exceeded.")
                    yield {"final_model": self.ax_client.generation_strategy.model}
                    return

                # Check patience (only during GPEI phase)
                if (
                    self.generation_strategy.current_step.model == Models.GPEI
                    and self.no_improvement_count > self.patience
                ):
                    print("Stopping early: no sufficient improvement in GPEI phase.")
                    yield {"final_model": self.ax_client.generation_strategy.model}
                    return

                total_trials += 1
                if total_trials >= self.num_trials:
                    break

        yield {"final_model": self.ax_client.generation_strategy.model}
