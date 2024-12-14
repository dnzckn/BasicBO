# Overview
Working ipynb notebook demo of `ax_platform==0.4.3`

### Traditional Grid Search on left, BO on right

![top view](figures/output_1.png)

### Configuring Mode

![top view](figures/config_rates.png)


# Some flash cards on BO

1. **Gaussian Peak Function**:
   - The `gaussian_peak` function represents a 2D Gaussian distribution, centered at coordinates (0.2, 0.1). The standard deviation for both dimensions is set to 0.1, resulting in a peak centered at this point with a smooth, bell-shaped decline.
   - This function serves as the objective to be optimized by Bayesian optimization, simulating a scenario where we seek to find the peak value (maximum).

2. **Grid Sweep**:
   - A grid of points is generated from -1 to 1 for both `x` and `y` dimensions. The function is evaluated at each point in this grid to create a 2D representation of the objective surface.
   - This grid-based evaluation allows visualization of the entire parameter space, providing a baseline to compare against the Bayesian optimization process. It helps illustrate the quality of the optimization results.

3. **Bayesian Optimization Setup**:
   - The Bayesian optimization is implemented using the Ax framework, which allows for flexible experimentation with different optimization strategies.
   - **Generation Strategy**:
     - A **Sobol** sampling strategy is employed for initial exploration of the parameter space (`num_initial_samples` points). Sobol sequences are effective for sampling evenly across the parameter space, ensuring a good initial understanding of the landscape.
     - After Sobol sampling, a **Gaussian Process with Expected Improvement (GPEI)** model is used to refine the search for the maximum. The GPEI model leverages a surrogate model (Gaussian Process) to balance exploration and exploitation, optimizing the objective until convergence or exhaustion of allowed trials (`num_trials=-1` means it continues indefinitely until another stopping criterion is met).
     - The generation strategy helps control how the initial exploration and subsequent exploitation phases are managed.
   - **Objective Setup**:
     - The objective is defined as maximizing `z`, which represents the value of the Gaussian peak given the inputs `x` and `y`. The experiment is designed with two parameters (`x` and `y`), both ranging from -1 to 1.

4. **Optimization Loop**:
   - The optimization loop iterates for up to `num_bo_samples` times, or until a convergence criterion is met.
   - **Exploitation vs. Exploration**:
     - In each iteration, a decision is made on whether to exploit the known high-performing areas (`GPEI`) or to explore less certain regions (`qNIPV` - Negative Integrated Posterior Variance). This is determined based on the ratio defined by `exploit_ratio`.
     - **Exploitation** (`GPEI`): Focuses on sampling points that are expected to yield the highest improvement based on the current surrogate model.
     - **Exploration** (`qNIPV`): Samples points to reduce uncertainty in the model, helping to improve the quality of the surrogate model and prevent getting stuck in local optima.
     - The balance between exploitation and exploration is key to efficiently finding the global optimum while avoiding local traps.
   - **Convergence Check**:
     - After each iteration, the improvement in the objective function is checked. If the improvement falls below a defined threshold (`epsilon`) for a specified number of consecutive iterations (`patience`), the optimization is stopped early. This prevents unnecessary computation when further significant improvements are unlikely.
     - Refitting the GP model every `refit_every` iterations helps maintain model accuracy without excessive computational cost.

5. **Normalization and Standardization**:
   - **Normalization**: Inputs (`x` and `y`) are normalized to ensure they fall within a consistent range. This helps the optimization algorithm treat all inputs equally, regardless of their original scale, thereby improving numerical stability and the performance of the Gaussian Process model.
   - **Standardization**: Outputs (`z`) are standardized to have zero mean and unit variance. This ensures the objective values are easier for the model to learn and improves training efficiency. Standardizing outputs is particularly important for Gaussian Process models, which are sensitive to the scale of the data.

6. **Plotting**:
   - The visualization consists of two subplots:
     - **Grid Sweep Plot**: The first subplot shows the objective function evaluated over a dense grid of points. This gives a clear picture of the Gaussian peak and serves as the reference for assessing the performance of the optimizer.
     - **Bayesian Optimization Process Plot**: The second subplot illustrates the points sampled by the Bayesian optimization process. The sampled points are shown as black circles overlaid on a color map created via `scipy.interpolate.griddata`. This plot helps visualize how the optimizer explores and exploits the parameter space to converge on the peak of the Gaussian.
   - The distinction between the grid-based visualization and the Bayesian optimization plot helps in understanding the efficiency of the optimizer compared to a brute-force approach of evaluating all points in the parameter space.