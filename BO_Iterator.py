import numpy as np
from typing import List, Tuple, Dict, Union, Iterator
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import torch
from torch.quasirandom import SobolEngine
from typing import List, Tuple, Dict, Union, Iterator, Callable, Any
from abc import ABC, abstractmethod


class BaseOptimizerIterator(ABC, Iterator):
    """
    A generic base class for optimization iterators.
    """

    def __init__(
        self,
        param_names: List[str],
        param_bounds: List[Tuple[float, float]],
        objective_function: Callable[[List[float]], Tuple[float, float]] = None,
        threshold: Union[float, None] = None,
        maximize: bool = True,
    ):
        if len(param_names) != len(param_bounds):
            raise ValueError("param_names and param_bounds must match in length.")

        self.param_names = param_names
        self.param_bounds = param_bounds
        self.objective_function = objective_function
        self.threshold = threshold
        self.maximize = maximize
        self.trials = (
            []
        )  # Store trial results as a list of {"params": ..., "result": ...}
        self.current_step = 0
        self.final_model = None  # To store the final model or state

    def evaluate_objective(self, params: Dict[str, float]) -> Tuple[float, float]:
        if self.objective_function is None:
            raise ValueError("Objective function is not defined.")
        return self.objective_function([params[name] for name in self.param_names])

    def record_result(
        self, param_dict: Dict[str, float], result: Union[float, Tuple[float, float]]
    ) -> None:
        self.trials.append({"params": param_dict, "result": result})

    def should_stop(self, result: Union[float, Tuple[float, float]]) -> bool:
        if self.threshold is None:
            return False

        value = result[0] if isinstance(result, tuple) else result
        return value >= self.threshold if self.maximize else value <= self.threshold

    def get_all_data(self) -> Dict[str, List[Any]]:
        params = [trial["params"] for trial in self.trials]
        results = [trial["result"] for trial in self.trials]
        return {"params": params, "results": results}

    def get_final_model(self) -> Any:
        """Retrieve the final model or state after optimization."""
        return self.final_model

    @abstractmethod
    def __next__(self) -> Dict[str, float]:
        pass


class BayesianOptimizerIterator(BaseOptimizerIterator):
    def __init__(
        self,
        objective_function: Callable[[List[float]], Tuple[float, float]],
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
        super().__init__(
            param_names, param_bounds, objective_function, threshold, maximize
        )

        self.num_sobol = num_sobol
        self.num_gpei = num_gpei
        self.num_trials = num_sobol + num_gpei
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.patience = patience

        self.best_value = float("-inf") if maximize else float("inf")
        self.best_params = None
        self.no_improvement_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=num_sobol),
                GenerationStep(
                    model=Models.GPEI,
                    num_trials=num_gpei,
                    max_parallelism=batch_size,
                    model_kwargs={"torch_device": self.device},
                ),
            ]
        )

        self.ax_client = AxClient(generation_strategy=self.generation_strategy)
        self.ax_client.create_experiment(
            name="sobol_gpei_optimization",
            parameters=[
                {"name": name, "type": "range", "bounds": bounds}
                for name, bounds in zip(param_names, param_bounds)
            ],
            objectives={"objective": ObjectiveProperties(minimize=not maximize)},
        )

    def __next__(self) -> Dict[str, float]:
        if self.current_step >= self.num_trials:
            # Stop iteration after all trials are completed
            self.final_model = self.ax_client.generation_strategy.model
            raise StopIteration

        trial_params, trial_index = self.ax_client.get_next_trial()
        objective_mean, objective_sem = self.evaluate_objective(trial_params)
        self.ax_client.complete_trial(
            trial_index, {"objective": (objective_mean, objective_sem)}
        )

        # Update the best value and params
        if (self.maximize and objective_mean > self.best_value) or (
            not self.maximize and objective_mean < self.best_value
        ):
            self.best_value = objective_mean
            self.best_params = trial_params
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # Check stopping conditions
        if (
            self.should_stop(self.best_value)
            or self.no_improvement_count > self.patience
        ):
            print("Stopping early: Optimization converged.")
            # Save the final model and stop the iteration
            self.final_model = self.ax_client.generation_strategy.model
            raise StopIteration

        self.current_step += 1
        return trial_params


class SobolIterator(BaseOptimizerIterator):
    def __init__(
        self,
        param_names: List[str],
        param_bounds: List[Tuple[float, float]],
        n_sobol: int = 30,
        **kwargs,
    ):
        """
        Initialize the Sobol Iterator.

        Args:
            param_names (list): List of parameter names.
            param_bounds (list): List of parameter bounds as (min, max) tuples.
            n_sobol (int): Number of Sobol samples to generate.
            kwargs: Additional arguments passed to BaseOptimizerIterator.
        """
        super().__init__(param_names, param_bounds, **kwargs)
        self.n_sobol = n_sobol
        self.sobol_engine = SobolEngine(dimension=len(param_names), scramble=True)
        self.best_value = float("-inf") if self.maximize else float("inf")
        self.best_params = None

    def __next__(self) -> Dict[str, float]:
        if self.current_step >= self.n_sobol:
            raise StopIteration

        sobol_pt = self.sobol_engine.draw(1).numpy()[0]
        param_dict = {
            name: low + sobol_pt[i] * (high - low)
            for i, (name, (low, high)) in enumerate(
                zip(self.param_names, self.param_bounds)
            )
        }
        self.current_step += 1

        if self.objective_function:
            result = self.evaluate_objective(param_dict)
            self.record_result(param_dict, result)

            # Update the best result if necessary
            value = result[0]  # Assuming result is a (mean, sem) tuple
            if (self.maximize and value > self.best_value) or (
                not self.maximize and value < self.best_value
            ):
                self.best_value = value
                self.best_params = param_dict

            if self.should_stop(result):
                print("Stopping early: threshold exceeded.")
                raise StopIteration

        return param_dict


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
