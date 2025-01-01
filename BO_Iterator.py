import numpy as np
from typing import List, Tuple, Dict, Union, Iterator, Callable, Any
import torch
from torch.quasirandom import SobolEngine
from abc import ABC, abstractmethod

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties


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
        record_data: bool = True,  # <-- Add a boolean flag here
    ):
        if len(param_names) != len(param_bounds):
            raise ValueError("param_names and param_bounds must match in length.")

        self.param_names = param_names
        self.param_bounds = param_bounds
        self.objective_function = objective_function
        self.threshold = threshold
        self.maximize = maximize

        # Whether to store observed data
        self.record_data = record_data

        # Store all observations in a list of {"params": ..., "objectives": ...}
        self.observations = []

        self.current_step = 0
        self.final_model = None  # To store the final model or state

        # Track best (for convenience)
        self.best_objectives = float("-inf") if maximize else float("inf")
        self.best_params = None

    def evaluate_objective(self, params_dict: Dict[str, float]) -> Tuple[float, float]:
        """
        Evaluate the objective_function (if defined) on the provided params_dict.
        """
        if self.objective_function is None:
            raise ValueError("Objective function is not defined.")
        return self.objective_function([params_dict[name] for name in self.param_names])

    def record_observation(
        self,
        params_dict: Dict[str, float],
        objectives_tuple: Union[float, Tuple[float, float]],
    ) -> None:
        """
        Store the newly observed params & objectives in self.observations,
        but only if record_data=True.
        """
        if self.record_data:
            self.observations.append(
                {"params": params_dict, "objectives": objectives_tuple}
            )

    def should_stop(self, objectives: Union[float, Tuple[float, float]]) -> bool:
        """
        Check whether we've exceeded (for maximize) or fallen below (for minimize)
        the threshold.
        """
        if self.threshold is None:
            return False

        # If objectives is a tuple, interpret the first element as the primary objective
        primary_obj = objectives[0] if isinstance(objectives, tuple) else objectives
        if self.maximize:
            return primary_obj >= self.threshold
        else:
            return primary_obj <= self.threshold

    def get_all_observations(self) -> Dict[str, Any]:
        """
        Return a dict containing all observed (params, objectives) pairs
        and also the best result found so far.
        """
        params_list = [obs["params"] for obs in self.observations]
        objectives_list = [obs["objectives"] for obs in self.observations]

        return {
            "params": params_list,
            "objectives": objectives_list,
            "best_result": {
                "params": self.best_params,
                "objectives": self.best_objectives,
            },
        }

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
        record_data: bool = True,
    ):
        super().__init__(
            param_names=param_names,
            param_bounds=param_bounds,
            objective_function=objective_function,
            threshold=threshold,
            maximize=maximize,
            record_data=record_data,
        )

        self.num_sobol = num_sobol
        self.num_gpei = num_gpei
        self.num_trials = num_sobol + num_gpei
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.patience = patience

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
            self.final_model = self.ax_client.generation_strategy.model
            raise StopIteration

        # Get next params & evaluate
        trial_params, trial_index = self.ax_client.get_next_trial()
        obj_mean, obj_sem = self.evaluate_objective(trial_params)
        self.ax_client.complete_trial(trial_index, {"objective": (obj_mean, obj_sem)})

        # Record in the iterator's internal observations list (only if record_data=True)
        self.record_observation(trial_params, (obj_mean, obj_sem))

        # Update best
        if (self.maximize and obj_mean > self.best_objectives) or (
            not self.maximize and obj_mean < self.best_objectives
        ):
            self.best_objectives = obj_mean
            self.best_params = trial_params
            self.no_improvement_count = 0
        else:
            # Only increment no_improvement_count in GPEI phase
            if self.current_step >= self.num_sobol:
                self.no_improvement_count += 1

        # Threshold check
        if self.should_stop(self.best_objectives):
            print("Stopping early: Threshold exceeded.")
            self.final_model = self.ax_client.generation_strategy.model
            raise StopIteration

        # Patience check (only in GPEI)
        if (
            self.current_step >= self.num_sobol
            and self.no_improvement_count > self.patience
        ):
            print("Stopping early: No improvement under GPEI for too long.")
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
        record_data: bool = True,  # <-- pass along to base if you want
        **kwargs,
    ):
        """
        Initialize the Sobol Iterator.
        """
        super().__init__(
            param_names=param_names,
            param_bounds=param_bounds,
            record_data=record_data,
            **kwargs,
        )
        self.n_sobol = n_sobol
        self.sobol_engine = SobolEngine(dimension=len(param_names), scramble=True)

    def __next__(self) -> Dict[str, float]:
        if self.current_step >= self.n_sobol:
            raise StopIteration

        sobol_pt = self.sobol_engine.draw(1).numpy()[0]
        params_dict = {
            name: low + sobol_pt[i] * (high - low)
            for i, (name, (low, high)) in enumerate(
                zip(self.param_names, self.param_bounds)
            )
        }
        self.current_step += 1

        if self.objective_function:
            objectives_tuple = self.evaluate_objective(params_dict)
            self.record_observation(params_dict, objectives_tuple)

            # Update best if needed
            primary_obj = objectives_tuple[0]  # (mean, sem)
            if (self.maximize and primary_obj > self.best_objectives) or (
                not self.maximize and primary_obj < self.best_objectives
            ):
                self.best_objectives = primary_obj
                self.best_params = params_dict

            if self.should_stop(objectives_tuple):
                print("Stopping early: threshold exceeded.")
                raise StopIteration

        return params_dict


class SyntheticGaussian:
    """
    A synthetic response object simulating a noisy Gaussian-like surface
    as the objective.
    """

    def __init__(self, centers: List[float], sigma: float = 0.1, n_samples: int = 1):
        self.centers = np.array(centers)
        self.sigma = sigma
        self.n_samples = n_samples

    def read(self, params: List[float]) -> Tuple[float, float]:
        """
        Compute the objective value (mean, sem).
        """
        params_arr = np.array(params)
        # Gaussian-like function
        base_obj = np.exp(
            -np.sum((params_arr - self.centers) ** 2) / (2 * self.sigma**2)
        )

        if self.n_samples > 1:
            # Multiple draws => add noise, average, compute sem
            noisy_objs = [
                base_obj + np.random.normal(0, self.sigma)
                for _ in range(self.n_samples)
            ]
            mean_obj = np.mean(noisy_objs)
            sem_obj = np.std(noisy_objs) / np.sqrt(self.n_samples)
            return mean_obj, sem_obj

        return base_obj, 0.0
