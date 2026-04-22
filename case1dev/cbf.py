import numpy as np
import cvxpy as cp

def hcw_drift_accel(r, v, n):
    x, y, z = r
    vx, vy, vz = v
    return np.array([
        3 * n**2 * x + 2 * n * vy,
        -2 * n * vx,
        -n**2 * z
    ], dtype=float)

def cbf_filter_translation(x_trans, u_nom, n, u_max, R_koz, k0=1.0, k1=2.0, rho=1e4, use_slack=True):
    r = x_trans[:3]
    v = x_trans[3:]

    a_drift = hcw_drift_accel(r, v, n)

    h = np.dot(r, r) - R_koz**2
    hdot = 2.0 * np.dot(r, v)

    rhs = (
        -2.0 * np.dot(v, v)
        -2.0 * np.dot(r, a_drift)
        -k1 * hdot
        -k0 * h
    )

    u = cp.Variable(3)

    constraints = [
        2.0 * r @ u >= rhs,
        u <= u_max,
        u >= -u_max,
    ]

    if use_slack:
        delta = cp.Variable(nonneg=True)
        constraints[0] = 2.0 * r @ u >= rhs - delta
        objective = cp.Minimize(cp.sum_squares(u - u_nom) + rho * cp.square(delta))
    else:
        objective = cp.Minimize(cp.sum_squares(u - u_nom))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)

    if u.value is None:
        return np.clip(u_nom, -u_max, u_max)

    return np.array(u.value).reshape(3,)