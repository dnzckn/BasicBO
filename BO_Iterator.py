import asyncio
import numpy as np
import torch
from typing import List, Tuple, Dict, Union, AsyncIterator, Iterator
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.acquisition import qNegIntegratedPosteriorVariance
from botorch.models.gp_regression import SingleTaskGP


class SyntheticGaussian:
    """A synthetic response object simulating a noisy Gaussian-like surface as the objective.
    In reality you would replace with your actual objective.

    E.g., for solar cells, you could seek to optimize the cell's measured
    open-circuit voltage as a function of its chemistry.
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


class BayesianOptimizerIterator:
    """An iterator class for performing Bayesian Optimization with Ax.

    Example usage:
    --------------
    >>> param_names = ["a", "b"]
    >>> param_bounds = [[-1.0, 1.0], [-1.0, 1.0]]
    >>> objective_function = SyntheticGaussian(centers=[0.2, 0.1], sigma=0.1, n_samples=1)
    >>> suggested_params = []
    >>> observed_z = []
    >>> best_result = {"params": None, "value": float("-inf")}
    >>> run_async = True
    >>> num_initial_samples = 20
    >>> trial_budget = 20
    >>> batch_size = 2
    >>> explore_ratio = 0.1
    >>> threshold = 0.999
    >>> epsilon = 0.001
    >>> patience = 20
    >>> bo_iterator = BayesianOptimizerIterator(
    >>>     objective_function,
    >>>     param_names,
    >>>     param_bounds,
    >>>     num_initial_samples=num_initial_samples,
    >>>     trial_budget=trial_budget,
    >>>     batch_size=batch_size,
    >>>     explore_ratio=explore_ratio,
    >>>     threshold=threshold,
    >>>     epsilon=epsilon,
    >>>     patience=patience,
    >>>     run_async=run_async
    >>> )
    >>> async for output in bo_iterator:
    >>>     if "final_model" in output:
    >>>         output = output["final_model"]
    >>>     else:
    >>>         params = output
    >>>         z_mean, z_sem = objective_function.read([params[name] for name in param_names])
    >>>         suggested_params.append(params)
    >>>         observed_z.append(z_mean)
    >>>         if z_mean > best_result["value"]:
    >>>             best_result = {"params": params, "value": z_mean}
    >>> results = {
    >>>     "suggested_params": suggested_params,
    >>>     "observed_z": observed_z,
    >>>     "best_result": best_result,
    >>> }
    """

    def __init__(
        self,
        objective_function: SyntheticGaussian,
        param_names: List[str],
        param_bounds: List[Tuple[float, float]],
        num_initial_samples: int = 10,
        trial_budget: int = 20,
        batch_size: int = 1,
        explore_ratio: float = 0.1,
        threshold: Union[float, None] = None,
        epsilon: float = 0.0,
        patience: int = 10,
        run_async: bool = True,
        maximize: bool = True,
    ):
        """
        Initialize the BayesianOptimizerIterator.

        Args:
            objective_function: Object with a .read() method returning (mean, sem).
            param_names (list): Parameter names.
            param_bounds (list): Parameter bounds as (min, max).
            num_initial_samples (int): Number of initial Sobol samples.
            trial_budget (int): Maximum number of BO steps.
            batch_size (int): Trials per batch.
            explore_ratio (float): Probability of switching to qNIPV exploration.
            threshold (float or None): If best observed value > threshold, stop early.
            epsilon (float): Minimum absolute improvement to reset patience in exploitation steps.
            patience (int): Number of exploitation steps allowed without improvement.
            run_async (bool): Whether to run the optimization asynchronously.
            maximize (bool): Whether to maximize or minimize the objective.
        """
        self.objective_function = objective_function
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.num_initial_samples = num_initial_samples
        self.trial_budget = trial_budget
        self.batch_size = batch_size
        self.explore_ratio = explore_ratio
        self.threshold = threshold
        self.epsilon = epsilon
        self.patience = patience
        self.run_async = run_async
        self.maximize = maximize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=num_initial_samples),
                GenerationStep(
                    model=Models.GPEI, num_trials=-1, max_parallelism=batch_size
                ),
            ]
        )
        self.ax_client = AxClient(
            generation_strategy=self.generation_strategy, torch_device=self.device
        )
        self.ax_client.create_experiment(
            name="modular_iterator_optimization",
            parameters=[
                {"name": name, "type": "range", "bounds": bounds}
                for name, bounds in zip(param_names, param_bounds)
            ],
            objectives={"z": ObjectiveProperties(minimize=not maximize)},
            parameter_constraints=[],
            tracking_metric_names=[],
        )
        self.best_value = float("-inf") if maximize else float("inf")
        self.no_improvement_count = 0
        self.last_best_value = self.best_value
        self.steps_taken = 0

    async def get_next_batch_async(self) -> List[Tuple[Dict[str, float], int]]:
        """
        Asynchronously request the next batch of trials from AxClient.

        Returns:
            list: A list of (params, trial_index) tuples.
        """
        loop = asyncio.get_event_loop()
        return await asyncio.gather(
            *[
                loop.run_in_executor(None, self.ax_client.get_next_trial)
                for _ in range(self.batch_size)
            ]
        )

    async def complete_trial_async(
        self, trial_index: int, result: Dict[str, Tuple[float, float]]
    ) -> None:
        """
        Asynchronously mark a given trial as completed with results.

        Args:
            trial_index (int): The index of the trial to complete.
            result (dict): The result to record for the trial.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.ax_client.complete_trial, trial_index, result
        )

    def get_next_batch_sync(self) -> List[Tuple[Dict[str, float], int]]:
        """
        Synchronously request the next batch of trials from AxClient.

        Returns:
            list: A list of (params, trial_index) tuples.
        """
        return [self.ax_client.get_next_trial() for _ in range(self.batch_size)]

    def complete_trial_sync(
        self, trial_index: int, result: Dict[str, Tuple[float, float]]
    ) -> None:
        """
        Synchronously mark a given trial as completed with results.

        Args:
            trial_index (int): The index of the trial to complete.
            result (dict): The result to record for the trial.
        """
        self.ax_client.complete_trial(trial_index, result)

    def __iter__(self) -> Iterator[Dict[str, Union[Dict[str, float], str]]]:
        """
        Return a synchronous iterator.

        Returns:
            iterator: A synchronous iterator.
        """
        return self._sync_iterator()

    def __aiter__(self) -> AsyncIterator[Dict[str, Union[Dict[str, float], str]]]:
        """
        Return an asynchronous iterator.

        Returns:
            iterator: An asynchronous iterator.
        """
        return self._async_iterator()

    def _sync_iterator(self) -> Iterator[Dict[str, Union[Dict[str, float], str]]]:
        """
        Synchronous iterator for Bayesian Optimization.

        Yields:
            dict: Parameter sets suggested by the optimizer for evaluation.
                  At the end, yields {"final_model": model_bridge} containing the final model.
        """
        # Generate initial Sobol samples
        trials = self.get_next_batch_sync()
        for params, trial_index in trials:
            yield params  # yield parameter dictionary
            z_mean, z_sem = self.objective_function.read(
                [params[name] for name in self.param_names]
            )
            self.complete_trial_sync(trial_index, {"z": (z_mean, z_sem)})
            if (self.maximize and z_mean > self.best_value) or (
                not self.maximize and z_mean < self.best_value
            ):
                self.best_value = z_mean
                self.no_improvement_count = 0

        self.last_best_value = self.best_value

        while self.steps_taken < self.trial_budget:
            explore = np.random.rand() < self.explore_ratio
            if explore:
                print("~ Using qNIPV (Exploration)")
                self.ax_client.generation_strategy._steps[
                    -1
                ].model = Models.BOTORCH_MODULAR
                self.ax_client.generation_strategy._steps[-1].model_kwargs = {
                    "surrogate": Surrogate(SingleTaskGP),
                    "botorch_acqf_class": qNegIntegratedPosteriorVariance,
                    "acquisition_options": {"mc_points": 50},
                }
            else:
                print("$ Using GPEI (Exploitation)")
                self.ax_client.generation_strategy._steps[-1].model = Models.GPEI
                self.ax_client.generation_strategy._steps[-1].model_kwargs = {}

            trials = self.get_next_batch_sync()
            for params, trial_index in trials:
                yield params  # yield parameter dictionary
                z_mean, z_sem = self.objective_function.read(
                    [params[name] for name in self.param_names]
                )
                self.complete_trial_sync(trial_index, {"z": (z_mean, z_sem)})

                if (self.maximize and z_mean > self.best_value) or (
                    not self.maximize and z_mean < self.best_value
                ):
                    self.best_value = z_mean
                    if not explore:
                        self.no_improvement_count = 0
                else:
                    if (
                        not explore
                        and abs(self.best_value - self.last_best_value) < self.epsilon
                    ):
                        self.no_improvement_count += 1

                self.last_best_value = self.best_value

                if self.threshold is not None and (
                    (self.maximize and self.best_value > self.threshold)
                    or (not self.maximize and self.best_value < self.threshold)
                ):
                    print("Stopping early: threshold exceeded.")
                    yield {"final_model": self.ax_client.generation_strategy.model}
                    return

                if not explore and self.no_improvement_count > self.patience:
                    print(
                        "Stopping early: no sufficient improvement in exploitation steps."
                    )
                    yield {"final_model": self.ax_client.generation_strategy.model}
                    return

            self.steps_taken += 1
            if self.steps_taken == self.trial_budget:
                print("Trial budget met")

        yield {"final_model": self.ax_client.generation_strategy.model}

    async def _async_iterator(
        self,
    ) -> AsyncIterator[Dict[str, Union[Dict[str, float], str]]]:
        """
        Asynchronous iterator for Bayesian Optimization.

        Yields:
            dict: Parameter sets suggested by the optimizer for evaluation.
                  At the end, yields {"final_model": model_bridge} containing the final model.
        """
        # Generate initial Sobol samples
        trials = await self.get_next_batch_async()
        for params, trial_index in trials:
            yield params  # yield parameter dictionary
            z_mean, z_sem = self.objective_function.read(
                [params[name] for name in self.param_names]
            )
            await self.complete_trial_async(trial_index, {"z": (z_mean, z_sem)})
            if (self.maximize and z_mean > self.best_value) or (
                not self.maximize and z_mean < self.best_value
            ):
                self.best_value = z_mean
                self.no_improvement_count = 0

        self.last_best_value = self.best_value

        while self.steps_taken < self.trial_budget:
            explore = np.random.rand() < self.explore_ratio
            if explore:
                print("~ Using qNIPV (Exploration)")
                self.ax_client.generation_strategy._steps[
                    -1
                ].model = Models.BOTORCH_MODULAR
                self.ax_client.generation_strategy._steps[-1].model_kwargs = {
                    "surrogate": Surrogate(SingleTaskGP),
                    "botorch_acqf_class": qNegIntegratedPosteriorVariance,
                    "acquisition_options": {"mc_points": 50},
                }
            else:
                print("$ Using GPEI (Exploitation)")
                self.ax_client.generation_strategy._steps[-1].model = Models.GPEI
                self.ax_client.generation_strategy._steps[-1].model_kwargs = {}

            trials = await self.get_next_batch_async()
            for params, trial_index in trials:
                yield params  # yield parameter dictionary
                z_mean, z_sem = self.objective_function.read(
                    [params[name] for name in self.param_names]
                )
                await self.complete_trial_async(trial_index, {"z": (z_mean, z_sem)})

                if (self.maximize and z_mean > self.best_value) or (
                    not self.maximize and z_mean < self.best_value
                ):
                    self.best_value = z_mean
                    if not explore:
                        self.no_improvement_count = 0
                else:
                    if (
                        not explore
                        and abs(self.best_value - self.last_best_value) < self.epsilon
                    ):
                        self.no_improvement_count += 1

                self.last_best_value = self.best_value

                if self.threshold is not None and (
                    (self.maximize and self.best_value > self.threshold)
                    or (not self.maximize and self.best_value < self.threshold)
                ):
                    print("Stopping early: threshold exceeded.")
                    yield {"final_model": self.ax_client.generation_strategy.model}
                    return

                if not explore and self.no_improvement_count > self.patience:
                    print(
                        "Stopping early: no sufficient improvement in exploitation steps."
                    )
                    yield {"final_model": self.ax_client.generation_strategy.model}
                    return

            self.steps_taken += 1
            if self.steps_taken == self.trial_budget:
                print("Trial budget met")

        yield {"final_model": self.ax_client.generation_strategy.model}
