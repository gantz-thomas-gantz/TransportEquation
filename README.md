# Transport Equation Solver

## Overview
This project implements numerical schemes for solving the one-dimensional transport equation:
$$ \frac{\partial u}{\partial t} + a \frac{\partial u}{\partial x} = 0 $$
using finite difference methods. It compares different discretization schemes and evaluates their accuracy and stability.

## Dependencies
The code requires the following Python libraries:
- `numpy` for numerical operations
- `plotly` for visualization
- `copy` for deep copying arrays

Install missing dependencies with:
```sh
pip install numpy plotly
```

## Implementation
### Initialization
- Defines the problem parameters (time interval, wave speed, and CFL condition).
- Specifies the initial condition $ u(x,0) = \sin(2 \pi x) $.
- Provides an analytical solution for validation.

### Spatial and Temporal Discretization
The function `espaceDiscretise(J, lam)` computes:
- `dx`: spatial step size
- `dt`: temporal step size
- `X`: array of discretized spatial points
- `M`: number of time steps

### Finite Difference Schemes
Several numerical schemes are implemented in a dictionary `Schema`:
- **Centered Difference (C)**
- **Upwind (DAG/DAD)**
- **Lax-Friedrichs (LF)**
- **Lax-Wendroff (LW)**

The function `schema(dx, dt, X, M, c_1, c0, c1)` computes the numerical solution for a given scheme.

### Numerical Solution and Visualization
- `numerical_solution(Schema, J)`: Computes and visualizes numerical solutions against the analytical solution.
- `plot_norm_Qn(Schema, J)`: Evaluates stability by plotting norms of matrix powers.
- `convergenceDeltaX(Schema, J)`: Analyzes error convergence with respect to spatial discretization.

## Usage
Run the script to visualize results for different schemes:
```sh
python transport_solver.py
```

## Results
- The numerical solutions are plotted and compared against the analytical solution.
- Stability analysis evaluates growth in matrix norms.
- Convergence studies confirm expected error behavior for different schemes.



