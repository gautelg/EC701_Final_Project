from dataclasses import dataclass
import time

import numpy as np

try:
    import casadi as ca
except ModuleNotFoundError:  # pragma: no cover - exercised once CasADi is installed
    ca = None

from sim.interface.sim_state import SimState
from sim.interface.control_command import ControlCommand
from control.base_controller import BaseController
from control.common.relative_safety_cbf import cbf_filter_translation


# ================================================
# NumPy quaternion helpers
# ================================================
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_normalize(q):
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-12 else q


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotate_vector(q, v):
    qv = np.array([0., v[0], v[1], v[2]])
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[1:]


def quat_to_dcm(q):
    q = quat_normalize(np.asarray(q, dtype=float))
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y)],
        [2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)],
        [2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)],
    ], dtype=float)


def dcm_to_quat(C):
    C = np.asarray(C, dtype=float)
    tr = np.trace(C)

    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (C[2, 1] - C[1, 2]) / s
        y = (C[0, 2] - C[2, 0]) / s
        z = (C[1, 0] - C[0, 1]) / s
    elif C[0, 0] > C[1, 1] and C[0, 0] > C[2, 2]:
        s = 2.0 * np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2])
        w = (C[2, 1] - C[1, 2]) / s
        x = 0.25 * s
        y = (C[0, 1] + C[1, 0]) / s
        z = (C[0, 2] + C[2, 0]) / s
    elif C[1, 1] > C[2, 2]:
        s = 2.0 * np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2])
        w = (C[0, 2] - C[2, 0]) / s
        x = (C[0, 1] + C[1, 0]) / s
        y = 0.25 * s
        z = (C[1, 2] + C[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1])
        w = (C[1, 0] - C[0, 1]) / s
        x = (C[0, 2] + C[2, 0]) / s
        y = (C[1, 2] + C[2, 1]) / s
        z = 0.25 * s

    return quat_normalize(np.array([w, x, y, z], dtype=float))


def _as_array(value, shape=None):
    arr = np.asarray(value, dtype=float)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _diag_matrix(value):
    arr = np.asarray(value, dtype=float)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (3,):
        return np.diag(arr)
    raise ValueError(f"Expected 3 values or a 3x3 matrix, got {arr.shape}")


def _scalar_last_to_first(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def _sim_state_to_case2_quaternion(state: SimState):
    q_inertial_to_body = _scalar_last_to_first(state.quaternion)
    R_BN = quat_to_dcm(q_inertial_to_body)

    if state.hill_to_inertial_dcm is not None:
        R_NH = np.asarray(state.hill_to_inertial_dcm, dtype=float)
    else:
        R_NH = np.eye(3)

    R_BH = R_BN @ R_NH
    return dcm_to_quat(R_BH)


def _pointing_error_deg(pos, q):
    dist = np.linalg.norm(pos)
    los = -pos / dist if dist > 1e-6 else np.array([0., 0., -1.])
    d_body = rotate_vector(q, los)
    cos_ang = np.clip(np.dot([0., 0., 1.], d_body), -1., 1.)
    return np.degrees(np.arccos(cos_ang))


def _require_casadi():
    if ca is None:
        raise ModuleNotFoundError(
            "CasADi is required for Case2MissionController. "
            "Install it first, e.g. `pip install casadi`."
        )


# ================================================
# Parameters
# ================================================
mu          = 3.986004418e14
alt         = 400e3
r_orbit     = 6378e3 + alt
n           = np.sqrt(mu / r_orbit**3)
m_chaser    = 500.0
J_np        = np.diag([100., 100., 100.])
J_inv_np    = np.diag([0.01, 0.01, 0.01])

u_max_thrust    = 5.0
tau_max         = 0.8
dt              = 4.0          # 4s timestep for speed
keep_out_radius = 8.0
circle_radius   = 12.0

# -------------------------------------------------------
# Waypoints — now with a pre-approach waypoint (WP_pre)
# that corrects x and z before the main approach burn.
#
# The root cause of x-drift: thrusting hard along y in
# HCW dynamics produces a Coriolis-induced x drift that
# overwhelms a short-horizon optimizer. The fix is to
# first correct x,z laterally (WP_pre), then approach
# purely along y (WP0) where no Coriolis coupling occurs.
#
# WP_pre: [ 0, -500,  0]  lateral correction only
# WP0:    [ 0,  -12,  0]  6 o'clock — pure y approach
# WP1:    [12,    0,  0]  3 o'clock
# WP2:    [ 0,   12,  0]  12 o'clock
# WP3:    [-12,   0,  0]  9 o'clock
# -------------------------------------------------------
v_circ = 2 * np.pi * circle_radius / 600.0

waypoints = np.array([
    [ 0.0,          -500.0,         0.0],   # WP_pre: lateral correction
    [ 0.0,          -circle_radius, 0.0],   # WP0: 6 o'clock
    [ circle_radius,  0.0,          0.0],   # WP1: 3 o'clock
    [ 0.0,           circle_radius, 0.0],   # WP2: 12 o'clock
    [-circle_radius,  0.0,          0.0],   # WP3: 9 o'clock
])

wp_tangent_vels = np.array([
    [ 0.0,    0.0,    0.0],   # WP_pre: come to rest laterally
    [ v_circ, 0.0,    0.0],   # WP0: tangent is [+1, 0]
    [ 0.0,    v_circ, 0.0],   # WP1: tangent is [ 0,+1]
    [-v_circ, 0.0,    0.0],   # WP2: tangent is [-1, 0]
    [ 0.0,   -v_circ, 0.0],   # WP3: tangent is [ 0,-1]
])

# Short horizon — keeps each solve tractable for IPOPT.
# N=30 * dt=4s = 120s lookahead, enough for local guidance.
N_leg = 30

# Arrival tolerances
arrival_tol_pre = 3.0   # looser for WP_pre (just needs x,z ~ 0)
arrival_tol     = 3.0   # for circle waypoints


# ================================================
# CasADi helpers
# ================================================
def ca_quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return ca.vertcat(w1*w2-x1*x2-y1*y2-z1*z2,
                      w1*x2+x1*w2+y1*z2-z1*y2,
                      w1*y2-x1*z2+y1*w2+z1*x2,
                      w1*z2+x1*y2-y1*x2+z1*w2)


def ca_quat_conj(q):
    return ca.vertcat(q[0], -q[1], -q[2], -q[3])


def ca_rotate_vector(q, v):
    qv = ca.vertcat(0, v)
    return ca_quat_multiply(ca_quat_multiply(q, qv), ca_quat_conj(q))[1:]


def make_physics_guess(x_cur, N, n_value, dt_value, J_np_value, J_inv_np_value):
    """
    Propagate forward with zero thrust to produce a dynamically
    feasible initial guess. Always satisfies the HCW dynamics
    constraints by construction, so IPOPT starts from a feasible
    point rather than spending iterations finding one.
    The waypoint target is handled by the cost function — the
    guess only needs to be feasible, not aimed at the waypoint.
    """
    g = np.zeros((13, N+1))
    x_sim = x_cur.copy()
    for i in range(N+1):
        g[:, i] = x_sim
        if i < N:
            pos = x_sim[0:3]
            vel = x_sim[3:6]
            # HCW with zero thrust
            dpos = vel
            dvel = np.array([
                3*n_value**2*pos[0] + 2*n_value*vel[1],
               -2*n_value*vel[0],
               -n_value**2*pos[2]
            ])
            x_sim[0:6] += dt_value * np.concatenate([dpos, dvel])
            q     = x_sim[6:10]
            omega = x_sim[10:13]
            q_dot = 0.5 * quat_multiply(q, np.concatenate([[0.], omega]))
            x_sim[6:10] = quat_normalize(q + dt_value * q_dot)
            # zero torque — omega evolves freely under gyroscopic terms
            omega_cross_J = np.cross(omega, J_np_value @ omega)
            x_sim[10:13] = omega + dt_value * J_inv_np_value @ (-omega_cross_J)
    return g


# ================================================
# Build MPC Opti problem (single version, reused for all legs)
# ================================================
def build_opti(
    N,
    n_value,
    m_chaser_value,
    J_value,
    u_max_thrust_value,
    tau_max_value,
    dt_value,
    weights=None,
):
    _require_casadi()

    J_ca = ca.DM(J_value)
    J_inv_ca = ca.DM(np.linalg.inv(J_value))

    # Coupled pointing cost: (1 - cos theta)^2
    # Depends on BOTH pos (sets LOS direction) and q (sets body frame)
    _p        = ca.MX.sym('p', 3)
    _q        = ca.MX.sym('q', 4)
    _d        = ca.sqrt(ca.dot(_p, _p) + 1.0)
    _los_lvlh = -_p / _d
    _los_body = ca_rotate_vector(_q, _los_lvlh)
    _cos_th   = ca.dot(_los_body, ca.vertcat(0., 0., 1.))
    pointing_cost_fn = ca.Function('pt_cost', [_p, _q], [(1.0 - _cos_th)**2])

    # Coupled pointing rate cost: penalise omega perpendicular to LOS
    _om             = ca.MX.sym('om', 3)
    _los_rate_cost  = ca.dot(_om, _om) - ca.dot(_om, _los_lvlh)**2
    pointing_rate_cost_fn = ca.Function('pt_rate_cost', [_p, _om], [_los_rate_cost])

    opti  = ca.Opti()
    X     = opti.variable(13, N+1)
    U     = opti.variable(6,  N)
    x0_p  = opti.parameter(13)
    wp_p  = opti.parameter(3)
    wpv_p = opti.parameter(3)

    weight_cfg = {
        "Q_vel": 50.0,
        "Q_omega": 300.0,
        "R_thrust": 80.0,
        "R_torque": 150.0,
        "Q_pointing": 25000.0,
        "Q_pointing_rate": 10000.0,
        "Q_att_reg": 10.0,
        "Q_qnorm": 1500.0,
        "Q_du": 80.0,
        "Q_dtau": 400.0,
        "Q_wp_pos": 800000.0,
        "Q_wp_vel": 300000.0,
        "Q_pos_lateral": 2000.0,
        "Q_pos_axial": 200.0,
    }
    if weights:
        weight_cfg.update(weights)

    Q_vel = float(weight_cfg["Q_vel"])
    Q_omega = float(weight_cfg["Q_omega"])
    R_thrust = float(weight_cfg["R_thrust"])
    R_torque = float(weight_cfg["R_torque"])
    Q_pointing = float(weight_cfg["Q_pointing"])
    Q_pointing_rate = float(weight_cfg["Q_pointing_rate"])
    Q_att_reg = float(weight_cfg["Q_att_reg"])
    Q_qnorm = float(weight_cfg["Q_qnorm"])
    Q_du = float(weight_cfg["Q_du"])
    Q_dtau = float(weight_cfg["Q_dtau"])
    Q_wp_pos = float(weight_cfg["Q_wp_pos"])
    Q_wp_vel = float(weight_cfg["Q_wp_vel"])

    # -------------------------------------------------------
    # Lateral vs axial position weights — split to prevent
    # Coriolis-induced x drift when thrusting along y.
    # x and z (lateral) are weighted 10x heavier than y
    # (axial) to keep the chaser on a tight lateral corridor.
    # -------------------------------------------------------
    Q_pos_lateral = float(weight_cfg["Q_pos_lateral"])   # x and z tracking (heavy)
    Q_pos_axial   = float(weight_cfg["Q_pos_axial"])     # y tracking (lighter — allow free approach)

    cost = 0

    for k in range(N):
        pos_k = X[0:3, k]
        vel_k = X[3:6, k]
        q_k   = X[6:10, k]
        om_k  = X[10:13, k]

        alpha   = k / N
        pos_ref = x0_p[0:3] * (1 - alpha) + wp_p * alpha
        vel_ref = x0_p[3:6] * (1 - alpha) + wpv_p * alpha

        # Split lateral/axial position cost
        cost += Q_pos_lateral * (pos_k[0] - pos_ref[0])**2   # x
        cost += Q_pos_axial   * (pos_k[1] - pos_ref[1])**2   # y
        cost += Q_pos_lateral * (pos_k[2] - pos_ref[2])**2   # z
        cost += Q_vel * ca.dot(vel_k - vel_ref, vel_k - vel_ref)

        # Coupled pointing (dominant term — attitude + translation linked)
        cost += Q_pointing      * pointing_cost_fn(pos_k, q_k)
        cost += Q_pointing_rate * pointing_rate_cost_fn(pos_k, om_k)

        # Attitude regularisation
        cost += Q_att_reg * ca.dot(q_k[1:4], q_k[1:4])
        cost += Q_qnorm   * (ca.dot(q_k, q_k) - 1.0)**2
        cost += Q_omega   * ca.dot(om_k, om_k)

        # Control effort
        cost += R_thrust * ca.dot(U[0:3, k], U[0:3, k])
        cost += R_torque * ca.dot(U[3:6, k], U[3:6, k])

        # Rate smoothness
        if k > 0:
            du = U[:, k] - U[:, k-1]
            cost += Q_du   * ca.dot(du[0:3], du[0:3])
            cost += Q_dtau * ca.dot(du[3:6], du[3:6])

    # Terminal costs
    cost += Q_wp_pos * ca.dot(X[0:3, -1] - wp_p,  X[0:3, -1] - wp_p)
    cost += Q_wp_vel * ca.dot(X[3:6, -1] - wpv_p, X[3:6, -1] - wpv_p)
    cost += 5.0 * Q_pointing * pointing_cost_fn(X[0:3, -1], X[6:10, -1])
    cost += Q_qnorm * (ca.dot(X[6:10, -1], X[6:10, -1]) - 1.0)**2
    cost += 3.0 * Q_omega * ca.dot(X[10:13, -1], X[10:13, -1])

    opti.minimize(cost)

    # Dynamics
    for k in range(N):
        pos   = X[0:3, k]
        vel   = X[3:6, k]
        q     = X[6:10, k]
        omega = X[10:13, k]
        accel  = U[0:3, k]
        torque = U[3:6, k]

        dpos = vel
        dvel = ca.vertcat(
            3*n_value**2*pos[0] + 2*n_value*vel[1] + accel[0],
           -2*n_value*vel[0]                             + accel[1],
           -n_value**2*pos[2]                            + accel[2]
        )
        omega_quat    = ca.vertcat(0, omega)
        q_dot         = 0.5 * ca_quat_multiply(q, omega_quat)
        q_next_raw    = q + dt_value * q_dot
        omega_cross_J = ca.cross(omega, J_ca @ omega)
        omega_dot     = J_inv_ca @ (torque - omega_cross_J)

        opti.subject_to(X[0:6,  k+1] == X[0:6,  k] + dt_value * ca.vertcat(dpos, dvel))
        opti.subject_to(X[6:10, k+1] == q_next_raw)
        opti.subject_to(X[10:13,k+1] == omega + dt_value * omega_dot)

    # Initial condition
    opti.subject_to(X[:, 0] == x0_p)

    # Control bounds
    accel_bound = np.asarray(u_max_thrust_value, dtype=float) / m_chaser_value
    for k in range(N):
        opti.subject_to(U[0:3, k] <= accel_bound)
        opti.subject_to(U[0:3, k] >= -accel_bound)
        opti.subject_to(U[3:6, k] <= tau_max_value)
        opti.subject_to(U[3:6, k] >= -tau_max_value)

    opti.solver('ipopt', {
        'print_time': False,
        'ipopt.print_level': 0,
        'ipopt.tol': 1e-4,
        'ipopt.acceptable_tol': 5e-3,
        'ipopt.constr_viol_tol': 1e-4,
        'ipopt.acceptable_constr_viol_tol': 1e-3,
        'ipopt.max_iter': 3000,
        'ipopt.acceptable_iter': 20,
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.hessian_approximation': 'limited-memory',
    })

    return opti, X, U, x0_p, wp_p, wpv_p


def _solve_case2_step(
    opti,
    X_var,
    U_var,
    x0_p,
    wp_p,
    wpv_p,
    x,
    wp_pos,
    wp_vel,
    N_leg_value,
    n_value,
    dt_value,
    J_np_value,
    J_inv_np_value,
    u_warm,
    x_warm,
    verbose=False,
    current_wp=0,
):
    solve_start = time.perf_counter()
    solve_status = "not_run"
    iter_count = None

    opti.set_value(x0_p,  x)
    opti.set_value(wp_p,  wp_pos)
    opti.set_value(wpv_p, wp_vel)

    if x_warm is not None:
        opti.set_initial(X_var, x_warm)
        opti.set_initial(U_var, u_warm)
    else:
        opti.set_initial(
            X_var,
            make_physics_guess(x, N_leg_value, n_value, dt_value, J_np_value, J_inv_np_value)
        )
        opti.set_initial(U_var, np.zeros((6, N_leg_value)))

    valid = True
    warm_start_ok = True

    try:
        sol     = opti.solve()
        solve_status = "solve_succeeded"
        if hasattr(sol, "stats"):
            try:
                stats = sol.stats()
                solve_status = str(stats.get("return_status", solve_status))
                iter_count = stats.get("iter_count")
            except Exception:
                pass
        u_opt   = sol.value(U_var)
        x_sol   = sol.value(X_var)
        u_apply = u_opt[:, 0]
    except RuntimeError as e:
        solve_status = f"warm_start_failed: {str(e)[:80]}"
        if verbose:
            print(f"  Warning: warm-start failed at WP{current_wp}: {str(e)[:80]}")
        try:
            opti.set_initial(
                X_var,
                make_physics_guess(x, N_leg_value, n_value, dt_value, J_np_value, J_inv_np_value)
            )
            opti.set_initial(U_var, np.zeros((6, N_leg_value)))
            sol     = opti.solve()
            solve_status = "solve_succeeded_after_cold_start"
            if hasattr(sol, "stats"):
                try:
                    stats = sol.stats()
                    solve_status = str(stats.get("return_status", solve_status))
                    iter_count = stats.get("iter_count")
                except Exception:
                    pass
            u_opt   = sol.value(U_var)
            x_sol   = sol.value(X_var)
            u_apply = u_opt[:, 0]
            warm_start_ok = False
        except RuntimeError as e2:
            solve_status = f"failed: {str(e2)[:80]}"
            if verbose:
                print(f"  Physics-guess also failed: {str(e2)[:80]} — zero input")
            u_opt   = np.zeros((6, N_leg_value))
            x_sol   = None
            u_apply = np.zeros(6)
            valid = False
            warm_start_ok = False

    solve_seconds = time.perf_counter() - solve_start

    return u_apply, u_opt, x_sol, valid, warm_start_ok, {
        "solve_seconds": solve_seconds,
        "solver_status": solve_status,
        "iter_count": iter_count,
    }


def _waypoint_arrived(
    x,
    current_wp,
    waypoints_value,
    arrival_tol_pre_value,
    arrival_tol_value,
    pre_approach_waypoint_index=0,
):
    # Optionally allow a dedicated pre-approach waypoint that only needs x/z
    # convergence before the controller transitions to the next leg.
    if pre_approach_waypoint_index is not None and current_wp == pre_approach_waypoint_index:
        lateral_err = np.sqrt(x[0]**2 + x[2]**2)
        return lateral_err < arrival_tol_pre_value
    return np.linalg.norm(x[0:3] - waypoints_value[current_wp]) < arrival_tol_value


@dataclass
class Case2MissionStatus:
    waypoint_index: int
    done: bool
    current_waypoint: np.ndarray | None
    current_waypoint_velocity: np.ndarray | None
    last_valid: bool
    pointing_error_deg: float
    last_solve_seconds: float
    last_solver_status: str
    last_iter_count: int | None


class Case2MissionController(BaseController):
    """
    Integrated translational + attitudinal Case 2 MPC controller.

    The underlying optimization remains the teammate-authored 13-state,
    6-input nonlinear MPC. This adapter packages it to the common
    BaseController -> ControlCommand sim interface.
    """

    def __init__(self, config):
        _require_casadi()

        self.config = dict(config)
        self.mass = float(self.config.get("mass", m_chaser))
        self.n = float(self.config.get("mean_motion", n))
        self.dt = float(self.config.get("controller_dt", dt))
        self.N_leg = int(self.config.get("horizon", N_leg))

        inertia_cfg = self.config.get("inertia_diag", np.diag(J_np))
        self.J_np = _diag_matrix(inertia_cfg)
        self.J_inv_np = np.linalg.inv(self.J_np)

        self.u_max_thrust = np.asarray(self.config.get("u_max_thrust", u_max_thrust), dtype=float)
        if self.u_max_thrust.shape == ():
            self.u_max_thrust = np.full(3, float(self.u_max_thrust))
        elif self.u_max_thrust.shape != (3,):
            raise ValueError(f"Expected scalar or 3-vector u_max_thrust, got {self.u_max_thrust.shape}")

        self.tau_max = np.asarray(self.config.get("tau_max", tau_max), dtype=float)
        if self.tau_max.shape == ():
            self.tau_max = np.full(3, float(self.tau_max))
        elif self.tau_max.shape != (3,):
            raise ValueError(f"Expected scalar or 3-vector tau_max, got {self.tau_max.shape}")

        self.keep_out_radius = float(self.config.get("keep_out_radius", keep_out_radius))
        self.use_cbf = bool(self.config.get("use_cbf", True))
        self.R_koz = float(self.config.get("R_koz", self.keep_out_radius))
        self.cbf_k0 = float(self.config.get("cbf_k0", 1.0))
        self.cbf_k1 = float(self.config.get("cbf_k1", 2.0))
        self.cbf_rho = float(self.config.get("cbf_rho", 1e4))
        self.cbf_use_slack = bool(self.config.get("cbf_use_slack", True))
        self.arrival_tol_pre = float(self.config.get("arrival_tol_pre", arrival_tol_pre))
        self.arrival_tol = float(self.config.get("arrival_tol", arrival_tol))
        self.pre_approach_waypoint_index = self.config.get("pre_approach_waypoint_index", 0)
        if self.pre_approach_waypoint_index is not None:
            self.pre_approach_waypoint_index = int(self.pre_approach_waypoint_index)
        self.verbose = bool(self.config.get("verbose", False))
        self.weights = dict(self.config.get("weights", {}))

        self.waypoints = _as_array(self.config.get("waypoints", waypoints))
        self.wp_tangent_vels = _as_array(self.config.get("wp_tangent_vels", wp_tangent_vels))

        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ValueError("Case2MissionController waypoints must have shape (N, 3)")
        if self.wp_tangent_vels.shape != self.waypoints.shape:
            raise ValueError("wp_tangent_vels must have the same shape as waypoints")
        if len(self.waypoints) == 0:
            raise ValueError("Case2MissionController requires at least one waypoint")

        self.opti, self.X_var, self.U_var, self.x0_p, self.wp_p, self.wpv_p = build_opti(
            self.N_leg,
            self.n,
            self.mass,
            self.J_np,
            self.u_max_thrust,
            self.tau_max,
            self.dt,
            weights=self.weights,
        )

        self.current_wp = 0
        self.done_flag = False
        self.u_warm = None
        self.x_warm = None

        self.last_valid = True
        self.last_force = np.zeros(3)
        self.last_torque = np.zeros(3)
        self.last_pointing_error_deg = np.nan
        self.last_solve_seconds = 0.0
        self.last_solver_status = "not_run"
        self.last_iter_count = None

    @property
    def done(self):
        return self.done_flag

    def current_waypoint(self):
        if self.current_wp >= len(self.waypoints):
            return None
        return self.waypoints[self.current_wp]

    def current_waypoint_velocity(self):
        if self.current_wp >= len(self.wp_tangent_vels):
            return None
        return self.wp_tangent_vels[self.current_wp]

    def status(self):
        return Case2MissionStatus(
            waypoint_index=self.current_wp,
            done=self.done_flag,
            current_waypoint=self.current_waypoint(),
            current_waypoint_velocity=self.current_waypoint_velocity(),
            last_valid=self.last_valid,
            pointing_error_deg=self.last_pointing_error_deg,
            last_solve_seconds=self.last_solve_seconds,
            last_solver_status=self.last_solver_status,
            last_iter_count=self.last_iter_count,
        )

    def _state_to_x(self, state: SimState):
        q_case2 = _sim_state_to_case2_quaternion(state)
        return np.concatenate([state.rel_pos, state.rel_vel, q_case2, state.omega])

    def step(self, state: SimState) -> ControlCommand:
        if self.done_flag:
            return ControlCommand(force=np.zeros(3), torque=np.zeros(3), valid=True)

        wp_pos = self.current_waypoint()
        wp_vel = self.current_waypoint_velocity()
        if wp_pos is None or wp_vel is None:
            self.done_flag = True
            return ControlCommand(force=np.zeros(3), torque=np.zeros(3), valid=True)

        x = self._state_to_x(state)

        u_apply, u_opt, x_sol, valid, warm_start_ok, solve_info = _solve_case2_step(
            self.opti,
            self.X_var,
            self.U_var,
            self.x0_p,
            self.wp_p,
            self.wpv_p,
            x,
            wp_pos,
            wp_vel,
            self.N_leg,
            self.n,
            self.dt,
            self.J_np,
            self.J_inv_np,
            self.u_warm,
            self.x_warm,
            verbose=self.verbose,
            current_wp=self.current_wp,
        )

        accel_cmd = np.asarray(u_apply[:3], dtype=float)
        if self.use_cbf:
            accel_cmd = cbf_filter_translation(
                np.hstack([state.rel_pos, state.rel_vel]),
                accel_cmd,
                self.n,
                self.u_max_thrust / self.mass,
                self.R_koz,
                k0=self.cbf_k0,
                k1=self.cbf_k1,
                rho=self.cbf_rho,
                use_slack=self.cbf_use_slack,
            )

        force = self.mass * accel_cmd
        torque = u_apply[3:]

        self.last_valid = valid
        self.last_force = force
        self.last_torque = torque
        self.last_pointing_error_deg = _pointing_error_deg(x[0:3], x[6:10])
        self.last_solve_seconds = float(solve_info["solve_seconds"])
        self.last_solver_status = str(solve_info["solver_status"])
        self.last_iter_count = solve_info["iter_count"]

        if warm_start_ok and x_sol is not None:
            self.u_warm = np.hstack([u_opt[:, 1:], u_opt[:, -1:]])
            self.x_warm = np.hstack([x_sol[:, 1:], x_sol[:, -1:]])
        else:
            self.u_warm = None
            self.x_warm = None

        if _waypoint_arrived(
            x,
            self.current_wp,
            self.waypoints,
            self.arrival_tol_pre,
            self.arrival_tol,
            self.pre_approach_waypoint_index,
        ):
            self.current_wp += 1
            self.u_warm = None
            self.x_warm = None
            if self.current_wp >= len(self.waypoints):
                self.done_flag = True

        return ControlCommand(force=force, torque=torque, valid=valid)

    def pointing_error(self, state: SimState):
        x = self._state_to_x(state)
        return _pointing_error_deg(x[0:3], x[6:10])


def run_standalone_demo():
    import matplotlib.pyplot as plt

    # ================================================
    # Initial conditions
    # ================================================
    pos0   = np.array([20., -500., 10.])
    vel0   = np.array([0., 0., 0.])
    q0     = quat_normalize(np.array([0.7071, 0., 0., 0.7071]))
    omega0 = np.array([0.01, -0.02, 0.03])
    x      = np.concatenate([pos0, vel0, q0, omega0])

    # ================================================
    # Build MPC problem (single problem reused for all legs)
    # ================================================
    print("Building MPC opti problem...")
    opti, X_var, U_var, x0_p, wp_p, wpv_p = build_opti(
        N_leg,
        n,
        m_chaser,
        J_np,
        np.full(3, u_max_thrust),
        np.full(3, tau_max),
        dt,
    )
    print("Done. Starting simulation.\n")

    # ================================================
    # Simulation loop
    # ================================================
    max_steps_total = 1200

    pos_hist       = []
    vel_hist       = []
    angle_hist     = []
    omega_hist     = []
    thrust_hist    = []
    torque_hist    = []
    wp_visit_steps = []

    current_wp = 0
    u_warm     = None
    x_warm     = None

    print(f"{'Step':>5} | {'x':>7} {'y':>7} {'z':>7} m | "
          f"{'phase':>8} | {'point':>6} | {'WP':>6} | "
          f"{'Fx':>6} {'Fy':>6} {'Fz':>6} N | "
          f"{'tx':>6} {'ty':>6} {'tz':>6} Nm")
    print("-" * 118)

    for step in range(max_steps_total):
        wp_pos = waypoints[current_wp]
        wp_vel = wp_tangent_vels[current_wp]

        u_apply, u_opt, x_sol, valid, warm_start_ok, _solve_info = _solve_case2_step(
            opti,
            X_var,
            U_var,
            x0_p,
            wp_p,
            wpv_p,
            x,
            wp_pos,
            wp_vel,
            N_leg,
            n,
            dt,
            J_np,
            J_inv_np,
            u_warm,
            x_warm,
            verbose=True,
            current_wp=current_wp,
        )

        thrust = m_chaser * u_apply[:3]
        torque = u_apply[3:]

        # Record
        pos_hist.append(x[0:3].copy())
        vel_hist.append(x[3:6].copy())
        omega_hist.append(x[10:13].copy())
        thrust_hist.append(thrust.copy())
        torque_hist.append(torque.copy())

        pos  = x[0:3]
        pointing_err = _pointing_error_deg(pos, x[6:10])
        angle_hist.append(pointing_err)

        if step % 10 == 0:
            phase    = np.degrees(np.arctan2(x[1], x[0]))
            wp_label = f'WP_pre' if current_wp == 0 else f'WP{current_wp-1}'
            print(f"{step:5d} | {x[0]:7.1f} {x[1]:7.1f} {x[2]:7.1f} | "
                  f"{phase:7.1f}° | {pointing_err:5.1f}° | {wp_label:>6} | "
                  f"{thrust[0]:6.2f} {thrust[1]:6.2f} {thrust[2]:6.2f} | "
                  f"{torque[0]:6.3f} {torque[1]:6.3f} {torque[2]:6.3f}")

        # Propagate state
        accel = u_apply[:3]
        dpos  = x[3:6]
        dvel  = np.array([
            3*n**2*pos[0] + 2*n*x[4] + accel[0],
           -2*n*x[3]                 + accel[1],
           -n**2*pos[2]              + accel[2]
        ])
        x[0:6] = x[0:6] + dt * np.concatenate([dpos, dvel])

        q     = x[6:10]
        omega = x[10:13]
        q_dot  = 0.5 * quat_multiply(q, np.concatenate([[0.], omega]))
        q_next = q + dt * q_dot
        x[6:10] = q_next / np.linalg.norm(q_next)

        omega_cross_J = np.cross(omega, J_np @ omega)
        omega_dot     = J_inv_np @ (torque - omega_cross_J)
        x[10:13] = omega + dt * omega_dot

        # Warm-start update — null after any cold-start recovery
        if warm_start_ok and x_sol is not None:
            u_warm = np.hstack([u_opt[:, 1:], u_opt[:, -1:]])
            x_warm = np.hstack([x_sol[:, 1:], x_sol[:, -1:]])
        else:
            u_warm = None
            x_warm = None

        # Waypoint switching
        if _waypoint_arrived(
            x,
            current_wp,
            waypoints,
            arrival_tol_pre,
            arrival_tol,
            pre_approach_waypoint_index=0,
        ):
            wp_visit_steps.append(step)
            wp_label = 'WP_pre' if current_wp == 0 else f'WP{current_wp-1}'
            print(f"\n  ✅ {wp_label} reached at step {step} "
                  f"(pointing={pointing_err:.1f}°)\n")
            current_wp += 1
            u_warm = None
            x_warm = None
            if current_wp >= len(waypoints):
                print(f"\n🎉 Full revolution complete at step {step}!")
                break

    print("\nSimulation finished.\n")

    # ================================================
    # Arrays
    # ================================================
    pos_hist    = np.array(pos_hist)
    vel_hist    = np.array(vel_hist)
    angle_hist  = np.array(angle_hist)
    omega_hist  = np.array(omega_hist)
    thrust_hist = np.array(thrust_hist)
    torque_hist = np.array(torque_hist)
    t_vec       = np.arange(len(pos_hist)) * dt

    # ================================================
    # Plots
    # ================================================
    plt.close('all')
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Waypoint-Guided Rendezvous Loop — Coupled Translational + Attitudinal MPC',
                 fontsize=13)

    wp_times  = [s * dt for s in wp_visit_steps]
    wp_labels = ['WP_pre'] + [f'WP{i}' for i in range(len(wp_visit_steps)-1)]

    def mark_wps(ax):
        ylo, yhi = ax.get_ylim()
        for i, t in enumerate(wp_times):
            ax.axvline(t, color='purple', lw=1.0, ls=':', alpha=0.7)
            if i < len(wp_labels):
                ax.text(t+2, ylo + 0.92*(yhi-ylo), wp_labels[i],
                        fontsize=7, color='purple')

    axs[0,0].plot(t_vec, pos_hist[:,0], label='x', lw=2)
    axs[0,0].plot(t_vec, pos_hist[:,1], label='y', lw=2)
    axs[0,0].plot(t_vec, pos_hist[:,2], label='z', lw=2)
    axs[0,0].set_ylabel('Position [m]'); axs[0,0].legend(); axs[0,0].grid(True)
    axs[0,0].set_title('Relative Position')

    axs[0,1].plot(t_vec, vel_hist[:,0], label='vx', lw=2)
    axs[0,1].plot(t_vec, vel_hist[:,1], label='vy', lw=2)
    axs[0,1].plot(t_vec, vel_hist[:,2], label='vz', lw=2)
    axs[0,1].set_ylabel('Velocity [m/s]'); axs[0,1].legend(); axs[0,1].grid(True)
    axs[0,1].set_title('Relative Velocity')

    axs[0,2].plot(t_vec, angle_hist, 'r', lw=2)
    axs[0,2].axhline(5,  color='orange', ls='--', lw=1.5, label='5° goal')
    axs[0,2].axhline(10, color='k',      ls='--', lw=1.0, label='10° limit')
    axs[0,2].set_ylabel('Pointing error [deg]'); axs[0,2].legend(); axs[0,2].grid(True)
    axs[0,2].set_title('Pointing Error (to Target)')

    axs[1,0].plot(t_vec, omega_hist[:,0], label='ωx', lw=2)
    axs[1,0].plot(t_vec, omega_hist[:,1], label='ωy', lw=2)
    axs[1,0].plot(t_vec, omega_hist[:,2], label='ωz', lw=2)
    axs[1,0].set_ylabel('Angular rate [rad/s]'); axs[1,0].legend(); axs[1,0].grid(True)
    axs[1,0].set_title('Angular Rates')

    axs[1,1].plot(t_vec, thrust_hist[:,0], label='Fx', lw=2)
    axs[1,1].plot(t_vec, thrust_hist[:,1], label='Fy', lw=2)
    axs[1,1].plot(t_vec, thrust_hist[:,2], label='Fz', lw=2)
    axs[1,1].set_ylabel('Thrust [N]'); axs[1,1].legend(); axs[1,1].grid(True)
    axs[1,1].set_title('Translational Thrust')

    axs[1,2].plot(t_vec, torque_hist[:,0], label='τx', lw=2)
    axs[1,2].plot(t_vec, torque_hist[:,1], label='τy', lw=2)
    axs[1,2].plot(t_vec, torque_hist[:,2], label='τz', lw=2)
    axs[1,2].set_ylabel('Torque [Nm]'); axs[1,2].legend(); axs[1,2].grid(True)
    axs[1,2].set_title('Attitude Torque')

    for ax in axs.flat:
        ax.set_xlabel('Time [s]')
        mark_wps(ax)

    plt.tight_layout()
    plt.savefig('mpc_waypoint_plots.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 3D Trajectory
    fig3d = plt.figure(figsize=(9, 7))
    ax3d  = fig3d.add_subplot(111, projection='3d')

    ax3d.plot(pos_hist[:,0], pos_hist[:,1], pos_hist[:,2],
              'b-', lw=2.5, label='Chaser trajectory')
    ax3d.plot([0], [0], [0], 'r*', markersize=14, label='Target')

    theta_plt = np.linspace(0, 2*np.pi, 200)
    ax3d.plot(circle_radius * np.cos(theta_plt),
              circle_radius * np.sin(theta_plt),
              np.zeros_like(theta_plt),
              'g--', lw=2, label=f'Nominal circle (r={circle_radius} m)')

    # Plot circle waypoints only (skip WP_pre)
    wp_colors = ['yellow', 'magenta', 'lime', 'cyan']
    for i, wp in enumerate(waypoints[1:]):
        ax3d.scatter(*wp, color=wp_colors[i], s=80, zorder=5)
        ax3d.text(wp[0]+0.5, wp[1]+0.5, wp[2]+0.5, f'WP{i}',
                  fontsize=9, color=wp_colors[i])

    # Keep-out sphere (for reference in plot even though not enforced)
    u_s = np.linspace(0, 2*np.pi, 40)
    v_s = np.linspace(0, np.pi, 40)
    ax3d.plot_wireframe(
        keep_out_radius * np.outer(np.cos(u_s), np.sin(v_s)),
        keep_out_radius * np.outer(np.sin(u_s), np.sin(v_s)),
        keep_out_radius * np.outer(np.ones_like(u_s), np.cos(v_s)),
        color='red', alpha=0.15, linewidth=0.6,
        label=f'Keep-out sphere (r={keep_out_radius} m)')

    ax3d.set_xlabel('x radial [m]')
    ax3d.set_ylabel('y along-track [m]')
    ax3d.set_zlabel('z cross-track [m]')
    ax3d.set_title('3D Relative Trajectory — Waypoint Loop (Close-up)')
    ax3d.set_xlim(-20, 20)
    ax3d.set_ylim(-20, 20)
    ax3d.set_zlim(-15, 15)
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.legend(fontsize=8)
    ax3d.grid(True)

    plt.savefig('mpc_waypoint_3d.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================
    # Summary statistics
    # ================================================
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total sim time:        {t_vec[-1]:.1f} s")
    print(f"Total steps:           {len(t_vec)}")
    print(f"Waypoints reached:     {len(wp_visit_steps)} / {len(waypoints)}")
    if wp_visit_steps:
        labels = ['WP_pre'] + [f'WP{i}' for i in range(len(wp_visit_steps)-1)]
        for i, (lbl, s) in enumerate(zip(labels, wp_visit_steps)):
            print(f"  {lbl}: step {s} ({s*dt:.0f} s)")

    if len(angle_hist) > 50:
        steady = np.array(angle_hist[50:])
        print(f"\nPointing error (after step 50):")
        print(f"  Mean:          {np.mean(steady):.2f} deg")
        print(f"  Median:        {np.median(steady):.2f} deg")
        print(f"  Max:           {np.max(steady):.2f} deg")
        print(f"  pct < 5 deg:   {100*np.mean(steady < 5):.1f}%")
        print(f"  pct < 10 deg:  {100*np.mean(steady < 10):.1f}%")

    print(f"\nMin keep-out clearance: "
          f"{min(np.linalg.norm(p) for p in pos_hist):.2f} m "
          f"(limit: {keep_out_radius} m)")
    print("=" * 60)


if __name__ == "__main__":
    run_standalone_demo()
