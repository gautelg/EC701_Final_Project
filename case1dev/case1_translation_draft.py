import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt


### SYSTEM/CONTROLLER/SOLVER

def hcw_matrices(n):
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [3*n**2, 0, 0, 0, 2*n, 0],
        [0, 0, 0, -2*n, 0, 0],
        [0, 0, -n**2, 0, 0, 0]
    ], dtype=float)

    B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float)

    return A, B

def discretize_system(A, B, dt):
    C = np.zeros((A.shape[0], A.shape[0]))
    D = np.zeros((A.shape[0], B.shape[1]))
    Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), dt)
    return Ad, Bd

def solve_mpc(x0, x_ref, Ad, Bd, Q, R, P, N, u_max):
    nx = Ad.shape[0]
    nu = Bd.shape[1]

    x = cp.Variable((nx, N+1))
    u = cp.Variable((nu, N))

    cost = 0
    constraints = [x[:, 0] == x0]

    for k in range(N):
        cost += cp.quad_form(x[:, k] - x_ref, Q)
        cost += cp.quad_form(u[:, k], R)
        constraints += [
            x[:, k+1] == Ad @ x[:, k] + Bd @ u[:, k],
            u[:, k] <= u_max,
            u[:, k] >= -u_max
        ]

    cost += cp.quad_form(x[:, N] - x_ref, P)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(
    solver=cp.OSQP,
    warm_start=True,
    verbose=False,
    eps_abs=1e-5,
    eps_rel=1e-5,
    max_iter=20000
    )

    if u.value is None:
        raise RuntimeError("MPC solve failed")

    return u.value[:, 0]

def simulate_closed_loop(x_init, waypoints, Ad, Bd, Q, R, P, N, u_max,
                         eps_r=0.1, eps_v=0.01, max_steps=1000):
    x = x_init.copy()
    history_x = [x.copy()]
    history_u = []
    wp_idx = 0

    for step in range(max_steps):
        r_ref = waypoints[wp_idx]
        x_ref = np.hstack([r_ref, np.zeros(3)])

        u = solve_mpc(x, x_ref, Ad, Bd, Q, R, P, N, u_max)
        x = Ad @ x + Bd @ u

        history_x.append(x.copy())
        history_u.append(u.copy())

        pos_err = np.linalg.norm(x[:3] - r_ref)
        vel_err = np.linalg.norm(x[3:])

        if pos_err < eps_r and vel_err < eps_v:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                break

    return np.array(history_x), np.array(history_u)








### MISSION PARAMETERS

n = 0.0011          # example mean motion [rad/s]
dt = 1.0            # sample time [s]
N = 20              # horizon length

A, B = hcw_matrices(n)
Ad, Bd = discretize_system(A, B, dt)

Q = np.diag([10, 10, 10, 5, 5, 5])
R = 0.1 * np.eye(3)
P = 20 * np.diag([10, 10, 10, 5, 5, 5])

u_max = np.array([0.01, 0.01, 0.01])   # accel bounds

R_circle = 10.0
waypoints = [
    np.array([ R_circle, 0.0, 0.0]),
    np.array([ 0.0, R_circle, 0.0]),
    np.array([-R_circle, 0.0, 0.0]),
    np.array([ 0.0,-R_circle, 0.0]),
]

x_init = np.array([30.0, -20.0, 5.0, 0.0, 0.0, 0.0])

X, U = simulate_closed_loop(x_init, waypoints, Ad, Bd, Q, R, P, N, u_max)






### PLOTTING

def plot_translation_results(X, U, waypoints, u_max=None, dt=1.0):
    t_x = np.arange(X.shape[0]) * dt
    t_u = np.arange(U.shape[0]) * dt

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 1) XY trajectory
    ax = axs[0, 0]
    ax.plot(X[:, 0], X[:, 1], label="trajectory")
    ax.scatter(X[0, 0], X[0, 1], marker='o', label="start")
    ax.scatter(0, 0, marker='x', s=80, label="target")
    wp = np.array(waypoints)
    ax.scatter(wp[:, 0], wp[:, 1], marker='s', label="waypoints")
    for i, r in enumerate(wp):
        ax.text(r[0], r[1], f"WP{i+1}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Relative trajectory in xy-plane")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    # 2) Position components
    ax = axs[0, 1]
    ax.plot(t_x, X[:, 0], label='x')
    ax.plot(t_x, X[:, 1], label='y')
    ax.plot(t_x, X[:, 2], label='z')
    ax.set_xlabel("time")
    ax.set_ylabel("position")
    ax.set_title("Position components")
    ax.grid(True)
    ax.legend()

    # 3) Velocity components
    ax = axs[1, 0]
    ax.plot(t_x, X[:, 3], label='xdot')
    ax.plot(t_x, X[:, 4], label='ydot')
    ax.plot(t_x, X[:, 5], label='zdot')
    ax.set_xlabel("time")
    ax.set_ylabel("velocity")
    ax.set_title("Velocity components")
    ax.grid(True)
    ax.legend()

    # 4) Control inputs
    ax = axs[1, 1]
    ax.plot(t_u, U[:, 0], label='ux')
    ax.plot(t_u, U[:, 1], label='uy')
    ax.plot(t_u, U[:, 2], label='uz')
    if u_max is not None:
        for i in range(3):
            ax.axhline(u_max[i], linestyle='--', linewidth=1)
            ax.axhline(-u_max[i], linestyle='--', linewidth=1)
    ax.set_xlabel("time")
    ax.set_ylabel("control")
    ax.set_title("Control inputs")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

plot_translation_results(X, U, waypoints, u_max=u_max, dt=dt)

