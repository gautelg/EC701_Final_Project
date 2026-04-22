# Case 1 Controller Design and Implementation Report

## 1. Purpose and System Overview

The Case 1 simulation models a chaser spacecraft that visits a sequence of relative-position waypoints around a target and, at each waypoint, rotates so that its body-frame +x axis points toward the target. The implementation separates the problem into two lower-level controllers:

- a translational model predictive controller (MPC), implemented in `case1_translation_draft.py`;
- a quaternion-based attitude proportional-derivative (PD) controller, implemented in `case1_attitude_draft.py`.

The integrated mission in `main_sim.py` uses `MissionManager` from `mission_manager.py` as a discrete supervisor. The supervisor alternates between a TRANSLATE mode and a ROTATE mode. In the nominal implementation, only one subsystem is actively controlled at a time: the translational controller moves the chaser to the current waypoint, then the attitude controller points the spacecraft toward the target before the next waypoint is assigned.

## 2. Translational Dynamics Model

The translational controller is based on the Hill-Clohessy-Wiltshire (HCW) relative motion equations. The relative state is

```text
x = [rx, ry, rz, vx, vy, vz]^T
```

where `r = [rx, ry, rz]^T` is the chaser position relative to the target, and `v = [vx, vy, vz]^T` is the corresponding relative velocity. The continuous-time linear system is

```text
xdot = A x + B u
```

where the control input `u = [ux, uy, uz]^T` is interpreted as commanded relative acceleration. The matrix `A` in `hcw_matrices(n)` is

```text
A = [[0,      0,   0,  1,    0,  0],
     [0,      0,   0,  0,    1,  0],
     [0,      0,   0,  0,    0,  1],
     [3n^2,   0,   0,  0,   2n,  0],
     [0,      0,   0, -2n,   0,  0],
     [0,      0, -n^2, 0,    0,  0]]
```

and

```text
B = [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]].
```

The parameter `n` is the target orbit mean motion. In the integrated simulation, `n = 0.0011 rad/s`. The HCW model is appropriate for small relative distances near a circular reference orbit, so the simulation treats the target as the origin of a local orbital frame and the chaser as a nearby spacecraft.

The continuous model is converted to a discrete-time system using SciPy's `cont2discrete`:

```text
x[k+1] = Ad x[k] + Bd u[k].
```

In `main_sim.py`, the sampling time is `dt = 1.0 s`.

## 3. Translational MPC Design

For each waypoint, the translational reference is formed as

```text
x_ref = [r_wp_x, r_wp_y, r_wp_z, 0, 0, 0]^T.
```

This means the controller is not only asked to reach the waypoint position, but also to arrive with near-zero relative velocity. At every time step, `solve_mpc(...)` solves a finite-horizon quadratic program over predicted states and control inputs:

```text
minimize sum_{k=0}^{N-1} [(x_k - x_ref)^T Q (x_k - x_ref) + u_k^T R u_k]
         + (x_N - x_ref)^T P (x_N - x_ref)
```

subject to

```text
x_0 = current state
x_{k+1} = Ad x_k + Bd u_k
-u_max <= u_k <= u_max.
```

The cost weights used by the integrated simulation are

```text
Q = diag(10, 10, 10, 5, 5, 5)
R = 0.1 I_3
P = 10 Q
N = 40
u_max = [0.01, 0.01, 0.01]^T.
```

The `Q` matrix penalizes position and velocity tracking error, with position weighted more strongly than velocity. The `R` matrix penalizes control effort, preventing the optimizer from using unnecessarily aggressive acceleration commands. The terminal matrix `P` increases the penalty on the final predicted state so the horizon ends close to the waypoint. The hard input bound `u_max` models actuator acceleration limits and makes the MPC more physically realistic than an unconstrained linear-quadratic regulator.

Only the first optimal control input is applied:

```text
u_trans = u_0*
x_trans <- Ad x_trans + Bd u_trans.
```

The optimization is then solved again at the next sample. This receding-horizon structure is the main benefit of MPC: it repeatedly replans using the latest state while respecting actuator bounds.

The implementation uses CVXPY with the OSQP solver. Since the model is linear, the cost is quadratic, and the constraints are linear box constraints, the MPC problem is a convex quadratic program.

## 4. Attitude Kinematics and Dynamics

The attitude model uses scalar-first unit quaternions

```text
q = [q0, q1, q2, q3]^T = [q0, qv^T]^T,
```

where `q0` is the scalar component and `qv` is the vector component. Quaternion multiplication, conjugation, normalization, and vector rotation are implemented in `case1_attitude_draft.py`.

The quaternion kinematics are

```text
qdot = 1/2 Omega(omega) q,
```

where `omega = [wx, wy, wz]^T` is body angular velocity and

```text
Omega(omega) =
[[0,   -wx, -wy, -wz],
 [wx,   0,   wz, -wy],
 [wy,  -wz,  0,   wx],
 [wz,   wy, -wx,  0 ]].
```

The rigid-body rotational dynamics are Euler's equation:

```text
J omegadot + omega x (J omega) = tau,
```

or equivalently

```text
omegadot = J^{-1} [tau - omega x (J omega)].
```

The code computes this with `np.linalg.solve(J, ...)` rather than explicitly forming `J^{-1}`, which is numerically preferable. In the integrated simulation,

```text
J = diag(8, 6, 5).
```

The Euler integration update used by the simulation is

```text
q <- normalize(q + dt qdot)
omega <- omega + dt omegadot.
```

Quaternion normalization is important because numerical integration can otherwise cause the quaternion norm to drift away from one.

## 5. Desired Pointing Quaternion

During ROTATE mode, the desired attitude is recomputed from the current chaser position and the fixed target position. The desired line-of-sight vector is

```text
ell = target_pos - chaser_pos
ell_hat = ell / ||ell||.
```

The desired attitude is the unit quaternion that rotates the body +x axis,

```text
b_x = [1, 0, 0]^T,
```

onto `ell_hat`. The helper `quaternion_from_two_vectors(a, b)` constructs this rotation. It handles the special cases where the two vectors are already aligned or exactly opposite. In the integrated mission, this setup means the spacecraft's boresight or body +x direction points toward the target after each translational waypoint is reached.

## 6. Quaternion Error and Attitude PD Control

The attitude error quaternion is computed as

```text
q_e = q_des * conj(q).
```

The code normalizes `q_e` and flips its sign if the scalar component is negative:

```text
if q_e0 < 0, q_e <- -q_e.
```

This enforces the shortest-rotation representation. Since `q` and `-q` represent the same physical attitude, choosing the error quaternion with nonnegative scalar part avoids commanding the spacecraft along the longer rotation path.

The control law is

```text
tau = Kq q_ev - Kw omega,
```

where `q_ev` is the vector part of `q_e`. The proportional term commands torque in the direction that reduces the pointing error. The derivative term damps angular velocity. The integrated simulation uses

```text
Kq = diag(1, 1, 1)
Kw = diag(6, 6, 6)
tau_max = [0.05, 0.05, 0.05]^T.
```

After computing the torque, the command is clipped componentwise:

```text
-tau_max <= tau <= tau_max.
```

This saturation approximates actuator torque limits. For small attitude errors, the vector part of the error quaternion is approximately half the rotation error vector:

```text
q_ev approx 1/2 theta e_hat.
```

Thus the controller behaves like a classical rotational spring-damper system near the desired attitude. The `Kq` gain sets the restoring torque sensitivity to pointing error, while `Kw` sets angular-rate damping.

## 7. Sequential Mission Logic

The integrated simulation in `main_sim.py` is intentionally sequential. In TRANSLATE mode, MPC drives the chaser toward the active waypoint while attitude is held fixed unless optional drift propagation is enabled. In ROTATE mode, translation is held fixed while the attitude controller aligns the body +x axis with the target line of sight.

The supervisor uses four tolerances:

```text
eps_r: waypoint position tolerance
eps_v: waypoint velocity tolerance
eps_q: quaternion error-vector tolerance
eps_w: angular-rate tolerance
```

In the integrated setup,

```text
eps_r = 0.5 m
eps_v = 0.05 m/s
eps_q = 0.05
eps_w = 0.02 rad/s
required_count = 5.
```

Translation is considered complete when

```text
||r - r_wp|| < eps_r and ||v|| < eps_v.
```

Rotation is considered complete when

```text
||q_ev|| < eps_q and ||omega|| < eps_w.
```

The `required_count` parameter prevents chatter by requiring the completion condition to remain true for several consecutive simulation steps before switching modes. After ROTATE mode completes, the waypoint index increments. When all waypoints have completed translation and pointing, the mission is marked done.

## 8. Implementation Flow

The main simulation flow is:

1. Build the HCW continuous-time matrices from `n`.
2. Discretize the translational model with `dt`.
3. Define MPC weights, horizon, and acceleration bounds.
4. Define attitude inertia, PD gains, and torque bounds.
5. Instantiate `MissionManager` with waypoint and completion tolerances.
6. At each time step:
   - get the active waypoint;
   - if in TRANSLATE mode, solve the MPC and propagate the translational state;
   - if in ROTATE mode, compute the line-of-sight desired quaternion, apply saturated quaternion PD control, and propagate attitude;
   - log state, control, and error histories;
   - ask the mission manager whether to switch modes.

This creates a modular architecture: the translational MPC, attitude PD controller, and mission-level switching logic are independent enough to test separately but are combined in `run_sequential_mission(...)`.

## 9. Assumptions and Limitations

The main modeling assumptions are:

- HCW dynamics assume small relative motion about a circular reference orbit.
- Translational inputs are treated as direct acceleration commands.
- Attitude control uses a diagonal inertia matrix and direct body torque commands.
- The translational and attitude controllers are sequenced rather than fully coupled.
- Euler integration is simple and readable but less accurate than higher-order integrators for long or stiff simulations.
- Torque and acceleration limits are modeled as componentwise saturation or box constraints.

These assumptions are reasonable for a first controller-design study because they keep the mathematical structure clear: translation is a constrained linear MPC problem, attitude is a nonlinear quaternion rigid-body control problem, and the mission manager is a deterministic hybrid supervisor.

## 10. Source Mapping

- `case1_translation_draft.py`: HCW matrices, discretization, MPC problem construction, and closed-loop waypoint simulation.
- `case1_attitude_draft.py`: quaternion algebra, desired pointing quaternion, quaternion error, rigid-body attitude dynamics, and saturated PD torque control.
- `main_sim.py`: integrated sequential mission loop, controller parameters, logging, plotting, effort metrics, and animation export.
- `mission_manager.py`: mode-switching logic, waypoint indexing, completion tolerances, and chatter-prevention counters.

