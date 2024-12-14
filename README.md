# Bayesian Optimization Demonstration

This repository contains a Jupyter notebook that demonstrates Bayesian Optimization (BO) using [`ax_platform==0.4.3`](https://github.com/facebook/Ax). We utilize a synthetic 2D Gaussian peak function as the objective and compare optimization strategies from brute-force grid search to guided Bayesian Optimization.

## Overview

### Traditional Grid Search vs. Bayesian Optimization

![Comparison](figures/output_1.png)

The figure above compares a traditional grid search (left) with Bayesian Optimization (right). Grid search exhaustively samples a dense grid across the parameter space, while Bayesian Optimization adaptively selects evaluation points using a probabilistic model to efficiently converge to the global optimum.

### Optimized Model

![Final Model](figures/final_model.png)

After completing BO, we plot the final model's mean and standard deviation, illustrating the optimized surrogate.

### Configuring the Optimization Mode

![Configuring Modes](figures/config_rates.png)

The tutorial showcases ax's ability to be ran in sync or async mode with varying batch sizes. The figure demonstrates the tradeoffs between rate and error on a short 20 shot on goal experiment. 

---

## Key Concepts of Bayesian Optimization (BO)

1. ### Objective Function: A Gaussian Peak

   We define our objective function f(x, y) as a 2D Gaussian centered at x=0.2, y=0.1 with a standard deviation of 0.1:

   ![Gaussian Response](figures/gaussian_response.png)

   This smooth, unimodal landscape is ideal for showcasing how an optimizer efficiently converges to a peak.

2. ### Parameter Space

   - **Parameters**: x, y
   - **Bounds**: Both x and y range from -1 to 1. 
   n.b. the params nor bounds need not be normalized.

3. ### Grid Sweep (Reference Baseline)

   We perform a grid search over a 51 by 51 grid, evaluating f(x, y) at each point. While this brute-force approach provides a comprehensive overview of the surface, it becomes computationally expensive for higher-dimensional or expensive-to-sample problems.

4. ### Bayesian Optimization Setup with Ax

Bayesian Optimization leverages a surrogate model, typically Gaussian Process Regression (GPR), to model the objective function and guide sampling. The Ax platform enables this process through a well-structured workflow:

4.1 **Initial Exploration ([Sobol-generated](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis) samples)**:
   - A batch of pseudo-random Sobol-generated samples is used to cover the parameter space evenly. This phase establishes a broad understanding of the landscape, providing a foundation for further refinement.

4.2 **Adaptive Sampling**:
   - Subsequent iterations adaptively alternate between:
     - **Exploitation**: Using Gaussian Process Expected Improvement (GPEI) to sample near known promising points and refine the model in areas of interest.
     - **Exploration**: Using Negative Integrated Posterior Variance (qNIPV) to sample in less-explored regions and reduce global uncertainty.

This adaptive strategy ensures efficient progress towards the global optimum while avoiding local maxima.

---

5. ### Adaptive Strategy: Balancing Exploitation and Exploration

The balance between exploitation and exploration is controlled probabilistically through a defined `explore_ratio`:

5.1 **Exploitation (GPEI)**:
  - Focuses on improving the current best-known result by sampling near high-performing areas.
  
5.2 **Exploration (qNIPV)**:
  - Samples in regions with high uncertainty to enhance the model's understanding of the entire parameter space.

This dynamic approach prevents the optimizer from prematurely converging on local optima and promotes thorough exploration of the search space.

6. ### Stopping Criteria

   To prevent unnecessary computations, we implement early stopping conditions:

   - **Threshold**: Stops early if the best observed value surpasses a predefined threshold.
   - **Epsilon and Patience**: Halts if consecutive exploitation steps do not improve the best value by at least epsilon for a certain number of steps (patience), avoiding endless searching when significant improvements are unlikely.

   These criteria make BO practical for scenarios with expensive evaluations.

---

## Key Concepts Summary

- **Objective Function**: A controlled 2D Gaussian peak used to understand the BO process on a known landscape.
- **Parameter Space**: Parameters x and y respective range for the search. Ax handles normalization and standardization in the back end.
- **Grid Sweep**: A brute-force baseline providing exhaustive coverage, useful for visualization and reference.
- **Bayesian Optimization Steps**: Involves sequential decision-making on point selection using a surrogate model (GP) and acquisition functions (EI, qNIPV).
- **Exploration vs. Exploitation**: Balances global search with local optimization to ensure convergence to the global optimum.
- **Stopping Conditions**: Efficiently terminates the search when further improvements are unlikely, conserving computational resources.

---

This notebook and codebase demonstrate the integration of these concepts with `ax_platform`. By following the workflow—from initial random sampling and model updates to adaptive exploration-exploitation trade-offs and early stopping—you gain practical insights into how Bayesian Optimization efficiently tackles real-world, black-box optimization tasks.

## Further Reading

Explore real-life Ax development demos where BO accelerates materials discovery:

- [PASCAL: The Perovskite Automated Spin Coat Assembly Line Accelerates Composition Screening in Triple-Halide Perovskite Alloys](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00075g)

- [Bayesian Optimization and Prediction of the Durability of Triple-Halide Perovskite Thin Films Under Light and Heat Stressors](https://pubs.rsc.org/en/content/articlelanding/2024/ma/d4ma00747f)
