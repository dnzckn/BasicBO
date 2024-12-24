import numpy as np
from typing import List, Tuple, Dict, Union, Iterator
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import torch
from torch.quasirandom import SobolEngine


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
        Potentially remove this method if you have data recording outside of func.
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
