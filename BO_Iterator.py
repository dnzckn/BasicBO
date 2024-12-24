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
from torch.quasirandom import SobolEngine


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
    """An iterator class for performing Bayesian Optimization with Ax==0.4.3.

    Example usage:
    --------------
    >>> param_names = ["a", "b"]
    >>> param_bounds = [[-1.0, 1.0], [-1.0, 1.0]]
    >>> objective_function = SyntheticGaussian(centers=[0.2, 0.1], sigma=0.1, n_samples=1)
    >>> suggested_params = []
    >>> observed_objective = []
    >>> best_result = {"params": None, "value": float("-inf")}
    >>> run_async = True
    >>> num_initial_samples = 20
    >>> trial_budget = 20
    >>> batch_size = 2
    >>> explore_ratio = 0.1
    >>> threshold = 0.999
    >>> epsilon = 0.001
    >>> patience = 20
    >>> maximize = True
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
    >>>     run_async=run_async,
    >>>     maximize=maximize,
    >>> )
    >>> async for output in bo_iterator:
    >>>     if "final_model" in output:
    >>>         output = output["final_model"]
    >>>     else:
    >>>         params = output
    >>>         objective_mean, objective_sem = bo_iterator.evaluate_objective(params)
    >>>         suggested_params.append(params)
    >>>         observed_objective.append(objective_mean)
    >>>         if objective_mean > best_result["value"]:
    >>>             best_result = {"params": params, "value": objective_mean}
    >>> results = {
    >>>     "suggested_params": suggested_params,
    >>>     "observed_objective": observed_objective,
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
            objectives={"objective": ObjectiveProperties(minimize=not maximize)},
            parameter_constraints=[],
            tracking_metric_names=[],
        )
        self.best_value = float("-inf") if maximize else float("inf")
        self.no_improvement_count = 0
        self.last_best_value = self.best_value
        self.steps_taken = 0

    def evaluate_objective(self, params: Dict[str, float]) -> Tuple[float, float]:
        """
        Evaluate the objective function.

        Args:
            params (dict): Parameter values at which to evaluate.
        Returns:
            tuple: (mean, sem) representing the evaluated objective and its SEM.
        """
        # This is where you would adjust to your objective function with the expected output
        return self.objective_function.read([params[name] for name in self.param_names])

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
            objective_mean, objective_sem = self.evaluate_objective(params)
            self.complete_trial_sync(
                trial_index, {"objective": (objective_mean, objective_sem)}
            )
            if (self.maximize and objective_mean > self.best_value) or (
                not self.maximize and objective_mean < self.best_value
            ):
                self.best_value = objective_mean
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
                    "torch_device": self.device,
                }
            else:
                print("$ Using GPEI (Exploitation)")
                self.ax_client.generation_strategy._steps[-1].model = Models.GPEI
                self.ax_client.generation_strategy._steps[-1].model_kwargs = {
                    "torch_device": self.device
                }

            trials = self.get_next_batch_sync()
            for params, trial_index in trials:
                yield params  # yield parameter dictionary
                objective_mean, objective_sem = self.evaluate_objective(params)
                self.complete_trial_sync(
                    trial_index, {"objective": (objective_mean, objective_sem)}
                )

                if (self.maximize and objective_mean > self.best_value) or (
                    not self.maximize and objective_mean < self.best_value
                ):
                    self.best_value = objective_mean
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
            objective_mean, objective_sem = self.evaluate_objective(params)
            await self.complete_trial_async(
                trial_index, {"objective": (objective_mean, objective_sem)}
            )
            if (self.maximize and objective_mean > self.best_value) or (
                not self.maximize and objective_mean < self.best_value
            ):
                self.best_value = objective_mean
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
                    "torch_device": self.device,
                }
            else:
                print("$ Using GPEI (Exploitation)")
                self.ax_client.generation_strategy._steps[-1].model = Models.GPEI
                self.ax_client.generation_strategy._steps[-1].model_kwargs = {
                    "torch_device": self.device
                }

            trials = await self.get_next_batch_async()
            for params, trial_index in trials:
                yield params  # yield parameter dictionary
                objective_mean, objective_sem = self.evaluate_objective(params)
                await self.complete_trial_async(
                    trial_index, {"objective": (objective_mean, objective_sem)}
                )

                if (self.maximize and objective_mean > self.best_value) or (
                    not self.maximize and objective_mean < self.best_value
                ):
                    self.best_value = objective_mean
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


class SobolIterator:
    """
    Yields Sobol sample parameter dictionaries in any dimension.
    The user can externally evaluate them and record results.

    Example usage:
    >>> iterator = SobolIterator(
    >>>     param_names=["x0", "x1"],
    >>>     param_bounds=[(0, 1), (0, 1)],
    >>>     n_sobol=1000,
    >>>     objective_function=your_objective_function,
    >>>     threshold=0.9,
    >>>     maximize=True
    >>> )
    >>> for param_dict in iterator:
    >>>     x0 = param_dict["x0"]
    >>>     x1 = param_dict["x1"]
    >>>     objective = iterator.evaluate_objective(param_dict)
    >>>     iterator.record_result(param_dict, objective)
    >>> data = iterator.get_all_data()
    """

    def __init__(
        self,
        param_names: List[str],
        param_bounds: List[Tuple[float, float]],
        n_sobol: int = 30,
        objective_function=None,
        threshold: Union[float, None] = None,
        maximize: bool = True,
    ):
        """
        Initialize the SobolIterator.

        Args:
            param_names (list): List of parameter names.
            param_bounds (list): List of parameter bounds as (min, max) tuples.
            n_sobol (int): Number of Sobol samples to generate.
            objective_function: Function to evaluate the objective.
            threshold (float or None): If best observed value > threshold, stop early.
            maximize (bool): Whether to maximize or minimize the objective.
        """
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.n_sobol = n_sobol
        self.objective_function = objective_function
        self.threshold = threshold
        self.maximize = maximize

        if len(param_names) != len(param_bounds):
            raise ValueError("param_names and param_bounds must match in length.")
        self.dimension = len(param_names)

        # Prepare Sobol engine
        self.sobol_engine = SobolEngine(dimension=self.dimension, scramble=True)
        self.current_step = 0

        # We'll store (param_dict, result) so you can retrieve them
        self.trials = []

    def __iter__(self) -> "SobolIterator":
        """
        Return the iterator object itself.

        Returns:
            SobolIterator: The iterator object.
        """
        return self

    def __next__(self) -> Dict[str, float]:
        """
        Generate the next Sobol sample.

        Returns:
            dict: A dictionary of parameter names and their sampled values.

        Raises:
            StopIteration: If the number of Sobol samples exceeds n_sobol or threshold is met.
        """
        if self.current_step >= self.n_sobol:
            raise StopIteration

        sobol_pt = self.sobol_engine.draw(1).numpy()[0]  # shape=(dimension,)
        param_dict = {}
        for i, name in enumerate(self.param_names):
            low, high = self.param_bounds[i]
            val = low + sobol_pt[i] * (high - low)
            param_dict[name] = float(val)

        self.current_step += 1

        if self.objective_function is not None:
            objective = self.evaluate_objective(param_dict)
            self.record_result(param_dict, objective)
            if self.should_stop(objective):
                print("Stopping early: threshold exceeded.")
                self.current_step = self.n_sobol
                # raise StopIteration # doesnt seem to work

        return param_dict

    def evaluate_objective(self, params: Dict[str, float]) -> Tuple[float, float]:
        """
        Evaluate the objective function.

        Args:
            params (dict): Parameter values at which to evaluate.
        Returns:
            tuple: (mean, sem) representing the evaluated objective and its SEM.
        """
        if self.objective_function is None:
            raise ValueError("Objective function is not defined.")
        return self.objective_function.read([params[name] for name in self.param_names])

    def record_result(
        self,
        param_dict: Dict[str, float],
        result: Union[float, Tuple[float, float]],
    ) -> None:
        """
        Record the result of evaluating the parameters.

        Args:
            param_dict (dict): The parameter dictionary.
            result (float or tuple): The result of the evaluation.
        """
        self.trials.append({"params": param_dict, "result": result})

    def should_stop(self, result: Union[float, Tuple[float, float]]) -> bool:
        """
        Determine if the iterator should stop early based on the threshold.

        Args:
            result (float or tuple): The result of the evaluation.
        Returns:
            bool: True if the iterator should stop, False otherwise.
        """
        if self.threshold is None:
            return False

        if isinstance(result, tuple):
            value = result[0]
        else:
            value = result

        if self.maximize:
            return value >= self.threshold
        else:
            return value <= self.threshold

    def get_all_data(
        self,
    ) -> Dict[str, List[Union[Dict[str, float], Union[float, Tuple[float, float]]]]]:
        """
        Retrieve all recorded trials.

        Returns:
            dict: A dictionary containing lists of parameter sets and their results.
        """
        sobol_params = [trial["params"] for trial in self.trials]
        observed_objective = [trial["result"] for trial in self.trials]
        return {"sobol_params": sobol_params, "observed_objective": observed_objective}
