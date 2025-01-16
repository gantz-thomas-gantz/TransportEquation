import numpy as np              # Import NumPy for numerical operations
import plotly.graph_objs as go  # Import Plotly for visualizations
import copy                     # Import for deep copying of arrays

#### ----------------------------------------
### INITIALIZATION
#### ----------------------------------------

### Problem description: Transport equation

# Time interval
T = 0.75 

# Speed of the wave
a = 1    

# Initial condition
def uini(x: float) -> float:
    """Defines the initial condition of the system."""
    return np.sin(2 * np.pi * x)

### Analytical solution
def ubar(x: float, t: float) -> float:
    """Returns the analytical solution of the transport equation."""
    return uini(x - a * t)

### Constant parameter
lam = 0.8

#### ----------------------------------------
### EXERCISE 3
#### ----------------------------------------

def espaceDiscretise(J: int, lam: float) -> tuple[float, float, np.ndarray, int]:
    """
    Discretizes the spatial and temporal domains.

    Parameters:
    J (int): Number of spatial steps.
    lam (float): CFL condition parameter.

    Returns:
    tuple: 
        - dx (float): Spatial step size.
        - dt (float): Temporal step size.
        - X (np.ndarray): Array of spatial points in [0, 1).
        - M (int): Number of time steps.
    """
    dx = 1 / J
    dt = lam * dx
    X = dx * np.arange(0, J - 1)
    M = int(T / dt)
    return dx, dt, X, M

def schema(
    dx: float, dt: float, X: np.ndarray, M: int, c_1: float, c0: float, c1: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Computes the numerical solution using a given finite difference scheme.

    Parameters:
    dx (float): Spatial step size.
    dt (float): Temporal step size.
    X (np.ndarray): Array of spatial points.
    M (int): Number of time steps.
    c_1 (float): Coefficient for the left neighbor.
    c0 (float): Coefficient for the central point.
    c1 (float): Coefficient for the right neighbor.

    Returns:
    tuple:
        - Uh (np.ndarray): Numerical solution at the final time step.
        - Ub (np.ndarray): Analytical solution at the final time step.
        - errL2 (float): L2 norm of the error.
        - errInf (float): Infinity norm of the error.
    """
    J = len(X)

    # Temporary vectors for time integration
    Uhp = np.array([uini(x) for x in X])
    Uh = np.zeros(J)
    
    # Error measures
    errL2 = 0.0
    errInf = 0.0

    # Iterate over time steps
    for n in range(1, M + 1):

        # Analytical solution at t = n * dt
        Ub = np.array([ubar(x, n * dt) for x in X])  
        
        # (!) Use periodicity
        Uh[0] = c_1 * Uhp[J - 1] + c0 * Uhp[0] + c1 * Uhp[1]
        for j in range(1, J - 1):  # Inner points
            Uh[j] = c_1 * Uhp[j - 1] + c0 * Uhp[j] + c1 * Uhp[j + 1]
        Uh[J - 1] = c_1 * Uhp[J - 2] + c0 * Uhp[J - 1] + c1 * Uhp[0]

        Uhp = copy.deepcopy(Uh)

        # Update error
        errL2 = max(errL2, np.linalg.norm(Uh - Ub, 2) * np.sqrt(dx))
        errInf = max(errInf, np.linalg.norm(Uh - Ub, np.inf))
  
    return Uh, Ub, errL2, errInf

# Finite difference schemes
Schema = {
    "C": {  # Centre 
        "-1": (lam * a) / 2,   
        "0": 1.0, 
        "1": -(lam * a) / 2
    },
    "DAG": {  # Decentre a gauche
        "-1": lam * a,          
        "0": 1.0 - lam * a, 
        "1": 0.0
    },
    "DAD": {  # Decentre a droite
        "-1": 0.0,                  
        "0": 1.0 + lam * a, 
        "1": -lam * a
    },
    "LF": {  # Lax-Friedrichs
        "-1": 0.5 + (lam * a) / 2,  
        "0": 0.0, 
        "1": 0.5 - (lam * a) / 2
    },
    "LW": {  # Lax-Wendroff
        "-1": (lam * a) / 2 + (lam**2 * a**2) / 2, 
        "0": 1.0 - (lam**2 * a**2), 
        "1": -(lam * a) / 2 + (lam**2 * a**2) / 2
    }
}

def numerical_solution(Schema: dict, J: int) -> None:
    """
    Plots the analytical and numerical solutions for all defined schemes.

    Parameters:
    Schema (dict): Dictionary of schemes with coefficients.
    J (int): Number of spatial discretization points.

    Returns:
    None
    """
    dx, dt, X, M = espaceDiscretise(J, lam)
    fig = go.Figure()
    for schema_name, coeffs in Schema.items():
        Uh, Ub, errL2, errInf = schema(dx, dt, X, M, coeffs["-1"], coeffs["0"], coeffs["1"])
        fig.add_trace(go.Scatter(
            x=X, 
            y=Uh, 
            mode='lines+markers',  
            name=f"{schema_name} (Numerical Solution)",
            showlegend=True
        ))
    fig.add_trace(go.Scatter(
        x=X, 
        y=Ub, 
        mode='lines+markers',  
        name="Analytical Solution",
        showlegend=True
    ))
    fig.update_layout(
        title=r"Comparison of Analytical and Numerical Solutions",
        xaxis_title=r"$x$",
        yaxis_title=r"$u(x, t = \Delta t \cdot M)$",
        xaxis=dict(
            tickformat=".1e",  
            type="linear"       
        ),
        yaxis=dict(
            tickformat=".1e",  
            type="linear"       
        )
    )
    fig.show()

#### ----------------------------------------
### EXERCISE 4
#### ----------------------------------------

def determine_Q(c_1: float, c0: float, c1: float, J: int) -> np.ndarray:
    """
    Constructs the iteration matrix Q for a three-point finite difference scheme.

    Parameters:
    c_1 (float): Coefficient for the left neighbor.
    c0 (float): Coefficient for the central point.
    c1 (float): Coefficient for the right neighbor.
    J (int): Number of spatial discretization points.

    Returns:
    np.ndarray: Iteration matrix Q of size (J, J).
    """
    Q = np.diag([c0] * J)
    Q += np.diag([c1] * (J - 1), k=1)  # Upper off-diagonal
    Q += np.diag([c_1] * (J - 1), k=-1)  # Lower off-diagonal
    # Periodicity: wrap-around values
    Q[0, -1] = c_1
    Q[-1, 0] = c1
    return Q


def plot_norm_Qn(Schema: dict, J: int) -> None:
    """
    Plots the L2 and Infinity norms of the iteration matrix powers for all schemes.

    Parameters:
    Schema (dict): Dictionary of schemes with coefficients.
    J (int): Number of spatial discretization points.

    Returns:
    None
    """
    dx, dt, X, M = espaceDiscretise(J, lam)
    fig = go.Figure()
    time_steps = np.arange(0, M + 1)
    for schema_name, coeffs in Schema.items():
        norm_inf = []
        norm_2 = []
        Q = determine_Q(coeffs["-1"], coeffs["0"], coeffs["1"], J)
        for n in time_steps:
            Qn = np.linalg.matrix_power(Q, n)
            norm_inf.append(np.linalg.norm(Qn, ord=np.inf))
            norm_2.append(np.linalg.norm(Qn, ord=2))

        # Plot norms for each schema
        fig.add_trace(go.Scatter(
            x=time_steps, 
            y=norm_2, 
            mode='lines+markers',  
            name=f"{schema_name} (L2 Norm)",
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=time_steps, 
            y=norm_inf, 
            mode='lines+markers',  
            name=f"{schema_name} (Infinity Norm)",
            showlegend=True
        ))

    fig.update_layout(
        title=r"Stability Analysis",
        xaxis_title=r"$n$",
        yaxis_title=r"$\|Q^n\|$",
        xaxis=dict(
            tickformat=".1e",  
            type="linear"       
        ),
        yaxis=dict(
            tickformat=".1e",  
            type="linear"       
        )
    )
    fig.show()

#### ----------------------------------------
### EXERCISE 5
#### ----------------------------------------

def convergenceDeltaX(Schema: dict, J: list[int]) -> None:
    """
    Plots the error convergence with respect to spatial step size for all schemes.

    Parameters:
    Schema (dict): Dictionary of schemes with coefficients.
    J (list[int]): List of spatial discretization point counts.

    Returns:
    None
    """
    vec_errL2 = [[] for _ in range(len(Schema))]
    vec_errInf = [[] for _ in range(len(Schema))]
    vec_deltaX = np.array([])

    for j in J:
        dx, dt, X, M = espaceDiscretise(J=j, lam=lam)
        vec_deltaX = np.append(vec_deltaX, dx)
        for i, (schema_name, coeffs) in enumerate(Schema.items()):
            Uh, Ub, errL2, errInf = schema(dx, dt, X, M, coeffs["-1"], coeffs["0"], coeffs["1"])
            vec_errL2[i].append(errL2)
            vec_errInf[i].append(errInf)

    fig = go.Figure()

    for i, (schema_name, _) in enumerate(Schema.items()):
        fig.add_trace(go.Scatter(
            x=vec_deltaX, 
            y=vec_errL2[i], 
            mode='lines+markers',  
            name=f"{schema_name} (L2 Error)",
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=vec_deltaX, 
            y=vec_errInf[i], 
            mode='lines+markers',  
            name=f"{schema_name} (Infinity Error)",
            showlegend=True
        ))

    fig.add_trace(go.Scatter(
        x=vec_deltaX, 
        y=vec_deltaX, 
        mode='lines+markers',  
        name=r"$\Delta x$",
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=vec_deltaX, 
        y=vec_deltaX**2, 
        mode='lines+markers',  
        name=r"$\Delta x^2$",
        showlegend=True
    ))

    fig.update_layout(
        title=r"Error Convergence vs. Spatial Step Size",
        xaxis_title=r"$\Delta x$",
        yaxis_title=r"Error",
        xaxis=dict(
            tickformat=".1e",  
            type="log"       
        ),
        yaxis=dict(
            tickformat=".1e",  
            type="log"       
        )
    )
    fig.show()

#### ----------------------------------------
### MAIN
#### ----------------------------------------

numerical_solution(Schema=Schema, J=20)
plot_norm_Qn(Schema=Schema, J=20)
convergenceDeltaX(Schema=Schema, J=[25, 50, 100, 200])





