import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from case1_translation_draft import solve_mpc, hcw_matrices, discretize_system
from case1_attitude_draft import (
    quat_norm,
    quat_error,
    attitude_dynamics,
    attitude_pd_control,
    compute_desired_pointing_quaternion,
    rotate_vector_by_quaternion
)
from mission_manager import MissionManager

def run_sequential_mission(
    x_trans_0,
    q_0,
    omega_0,
    target_pos,
    manager,
    Ad,
    Bd,
    Q,
    R,
    P,
    N,
    u_max,
    J,
    Kq,
    Kw,
    tau_max,
    dt,
    drift_translation_during_rotate=False,
    drift_attitude_during_translate=False,
):
    x_trans = x_trans_0.copy()
    q = quat_norm(q_0.copy())
    omega = omega_0.copy()

    log = {
        "time": [],
        "mode": [],
        "wp_idx": [],
        "x_trans": [],
        "u_trans": [],
        "q": [],
        "omega": [],
        "tau": [],
        "pos_err": [],
        "att_err": [],
    }

    for k in range(manager.max_steps):
        t = k * dt

        if manager.done:
            break

        wp = manager.current_waypoint()

        # defaults for inactive subsystem
        u_trans = np.zeros(3)
        tau = np.zeros(3)
        q_err = np.array([1.0, 0.0, 0.0, 0.0])

        if manager.mode == "TRANSLATE":
            x_ref = np.hstack([wp, np.zeros(3)])
            u_trans = solve_mpc(x_trans, x_ref, Ad, Bd, Q, R, P, N, u_max)
            x_trans = Ad @ x_trans + Bd @ u_trans

            tau = np.zeros(3)   # hold attitude fixed during translation

            if drift_attitude_during_translate:
                qdot, wdot = attitude_dynamics(q, omega, tau, J)
                q = q + dt * qdot
                q = quat_norm(q)
                omega = omega + dt * wdot

        elif manager.mode == "ROTATE":
            q_des = compute_desired_pointing_quaternion(x_trans[:3], target_pos)
            tau, q_err = attitude_pd_control(q, omega, q_des, Kq, Kw, tau_max)

            qdot, wdot = attitude_dynamics(q, omega, tau, J)
            q = q + dt * qdot
            q = quat_norm(q)
            omega = omega + dt * wdot

            u_trans = np.zeros(3)   # hold translation fixed during rotation

            if drift_translation_during_rotate:
                x_trans = Ad @ x_trans + Bd @ u_trans

        # compute logging errors
        pos_err = np.linalg.norm(x_trans[:3] - wp)

        if manager.mode == "ROTATE":
            att_err = np.degrees(2.0 * np.arccos(np.clip(q_err[0], -1.0, 1.0)))
        else:
            att_err = np.nan

        # log
        log["time"].append(t)
        log["mode"].append(manager.mode)
        log["wp_idx"].append(manager.wp_idx)
        log["x_trans"].append(x_trans.copy())
        log["u_trans"].append(u_trans.copy())
        log["q"].append(q.copy())
        log["omega"].append(omega.copy())
        log["tau"].append(tau.copy())
        log["pos_err"].append(pos_err)
        log["att_err"].append(att_err)

        # update supervisor after propagation
        manager.update_mode(x_trans, q_err, omega)

    # convert list logs to arrays where appropriate
    log["x_trans"] = np.array(log["x_trans"])
    log["u_trans"] = np.array(log["u_trans"])
    log["q"] = np.array(log["q"])
    log["omega"] = np.array(log["omega"])
    log["tau"] = np.array(log["tau"])
    log["pos_err"] = np.array(log["pos_err"])
    log["att_err"] = np.array(log["att_err"])

    return log



if __name__ == "__main__":

    ############################
    # SYSTEM / CONTROLLER PARAMS
    ############################

    # --- translation
    n = 0.0011
    dt = 1.0
    N = 40   # slightly longer horizon helps

    A, B = hcw_matrices(n)
    Ad, Bd = discretize_system(A, B, dt)

    Q = np.diag([10, 10, 10, 5, 5, 5])
    R = 0.1 * np.eye(3)
    P = 10 * Q

    u_max = np.array([0.01, 0.01, 0.01])

    # --- attitude
    J = np.diag([8.0, 6.0, 5.0])
    Kq = np.diag([1.0, 1.0, 1.0])
    Kw = np.diag([6.0, 6.0, 6.0])
    tau_max = np.array([0.05, 0.05, 0.05])

    ############################
    # MISSION SETUP
    ############################

    R_circle = 10.0
    waypoints = [
        np.array([ R_circle, 0.0, 0.0]),
        np.array([ 0.0, R_circle, 0.0]),
        np.array([-R_circle, 0.0, 0.0]),
        np.array([ 0.0,-R_circle, 0.0]),
    ]

    target_pos = np.array([0.0, 0.0, 0.0])

    manager = MissionManager(
        waypoints=waypoints,
        eps_r=0.5,
        eps_v=0.05,
        eps_q=0.05,
        eps_w=0.02,
        max_steps=2000,
        required_count=5,
    )

    ############################
    # INITIAL CONDITIONS
    ############################

    x_trans_0 = np.array([30.0, -20.0, 5.0, 0.0, 0.0, 0.0])

    q_0 = np.array([1.0, 0.0, 0.0, 0.0])
    omega_0 = np.array([0.0, 0.0, 0.0])

    ############################
    # RUN SIMULATION
    ############################

    log = run_sequential_mission(
        x_trans_0,
        q_0,
        omega_0,
        target_pos,
        manager,
        Ad,
        Bd,
        Q,
        R,
        P,
        N,
        u_max,
        J,
        Kq,
        Kw,
        tau_max,
        dt
    )

    ############################
    # MAIN MISSION PLOTS
    ############################

    t = np.array(log["time"])
    X = log["x_trans"]
    Q_hist = log["q"]
    Tau = log["tau"]
    pos_err = log["pos_err"]
    att_err = log["att_err"]        # NaN outside ROTATE mode

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # XY trajectory
    ax = axs[0, 0]
    ax.plot(X[:, 0], X[:, 1], label="trajectory", linewidth=1)
    wp = np.array(waypoints)
    ax.scatter(wp[:, 0], wp[:, 1], marker='s', label="waypoints")
    ax.scatter(0, 0, marker='x', label="target")

    colors = ["blue" if m == "TRANSLATE" else "red" for m in log["mode"]]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=8, alpha=0.7)

    ax.set_title("Trajectory (xy-plane)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    # Position error
    ax = axs[0, 1]
    ax.plot(t, pos_err)
    ax.set_title("Position error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [m]")
    ax.grid(True)

    # Attitude error (ROTATE mode only)
    ax = axs[1, 0]
    ax.plot(t, att_err)
    ax.set_title("Attitude error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angle error [deg]")
    ax.grid(True)

    # mark mode switches
    for k in range(1, len(t)):
        if log["mode"][k] != log["mode"][k - 1]:
            ax.axvline(t[k], linestyle='--', alpha=0.25)

    # Mode over time
    ax = axs[1, 1]
    mode_id = [0 if m == "TRANSLATE" else 1 for m in log["mode"]]
    ax.step(t, mode_id, where="post")
    ax.set_title("Mode")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mode ID")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["TRANSLATE", "ROTATE"])
    ax.grid(True)

    plt.tight_layout()
    plt.show()

############################
# INTEGRATED COST / EFFORT PLOTS
############################

t = np.array(log["time"])
u_trans = log["u_trans"]
tau = log["tau"]
pos_err = log["pos_err"]
att_err = log["att_err"]

# norms
u_norm = np.linalg.norm(u_trans, axis=1)
tau_norm = np.linalg.norm(tau, axis=1)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Translational effort
axs[0, 0].plot(t, u_norm)
axs[0, 0].set_title("Translational control magnitude")
axs[0, 0].set_xlabel("time [s]")
axs[0, 0].set_ylabel("acceleration magnitude [m/s²]")
axs[0, 0].grid(True)

# Rotational effort
axs[0, 1].plot(t, tau_norm)
axs[0, 1].set_title("Rotational control magnitude")
axs[0, 1].set_xlabel("time [s]")
axs[0, 1].set_ylabel("torque magnitude [N·m]")
axs[0, 1].grid(True)

# Position error
axs[1, 0].plot(t, pos_err)
axs[1, 0].set_title("Position error")
axs[1, 0].set_xlabel("time [s]")
axs[1, 0].set_ylabel("error [m]")
axs[1, 0].grid(True)

# Attitude error
axs[1, 1].plot(t, att_err)
axs[1, 1].set_title("Attitude error")
axs[1, 1].set_xlabel("time [s]")
axs[1, 1].set_ylabel("angle error [deg]")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

############################
# INTEGRATED COMPONENT PLOTS
############################

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Translational control components
axs[0, 0].plot(t, u_trans[:, 0], label="u_x")
axs[0, 0].plot(t, u_trans[:, 1], label="u_y")
axs[0, 0].plot(t, u_trans[:, 2], label="u_z")
axs[0, 0].set_title("Translational control components")
axs[0, 0].set_xlabel("time [s]")
axs[0, 0].set_ylabel("acceleration command [m/s²]")
axs[0, 0].grid(True)
axs[0, 0].legend()

# Rotational torque components
axs[0, 1].plot(t, tau[:, 0], label="tau_x")
axs[0, 1].plot(t, tau[:, 1], label="tau_y")
axs[0, 1].plot(t, tau[:, 2], label="tau_z")
axs[0, 1].set_title("Rotational control components")
axs[0, 1].set_xlabel("time [s]")
axs[0, 1].set_ylabel("torque [N·m]")
axs[0, 1].grid(True)
axs[0, 1].legend()

# Translational velocity components
axs[1, 0].plot(t, X[:, 3], label="v_x")
axs[1, 0].plot(t, X[:, 4], label="v_y")
axs[1, 0].plot(t, X[:, 5], label="v_z")
axs[1, 0].set_title("Relative velocity components")
axs[1, 0].set_xlabel("time [s]")
axs[1, 0].set_ylabel("velocity [m/s]")
axs[1, 0].grid(True)
axs[1, 0].legend()

# Angular velocity components
axs[1, 1].plot(t, log["omega"][:, 0], label="omega_x")
axs[1, 1].plot(t, log["omega"][:, 1], label="omega_y")
axs[1, 1].plot(t, log["omega"][:, 2], label="omega_z")
axs[1, 1].set_title("Angular velocity components")
axs[1, 1].set_xlabel("time [s]")
axs[1, 1].set_ylabel("angular velocity [rad/s]")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.show()



############################
# SUMMARY METRICS
############################

u_l1_raw = np.sum(np.abs(u_trans))
tau_l1_raw = np.sum(np.abs(tau))

u_l1_dt = dt * u_l1_raw     # delta-v-like proxy
tau_l1_dt = dt * tau_l1_raw # integrated rotational control effort

mission_time = t[-1] if len(t) > 0 else 0.0

print("Mission complete:", manager.done)
print("Final waypoint index:", manager.wp_idx)
print("Mission time [s]:", mission_time)
print("Mission time [min]:", mission_time / 60.0)

print("Translational effort raw L1:", u_l1_raw)
print("Rotational effort raw L1:", tau_l1_raw)

print("Integrated translational effort dt*L1 [m/s proxy]:", u_l1_dt)
print("Integrated rotational effort dt*L1 [N·m·s proxy]:", tau_l1_dt)

print("Max translational control magnitude [m/s²]:", np.max(np.linalg.norm(u_trans, axis=1)))
print("Max rotational torque magnitude [N·m]:", np.max(np.linalg.norm(tau, axis=1)))


# ###########################
# LIGHTWEIGHT ANIMATION
# ###########################

# NOTE: primarily use this animation for debugging ---

plt.ion()

fig, ax = plt.subplots(figsize=(7, 7))

X = log["x_trans"]
Q_hist = log["q"]
wp = np.array(waypoints)

for k in range(len(log["time"])):
    ax.clear()

    # trajectory so far
    ax.plot(X[:k+1, 0], X[:k+1, 1], linewidth=1, label="trajectory")

    # waypoints and target
    ax.scatter(wp[:, 0], wp[:, 1], marker='s', label="waypoints")
    ax.scatter(target_pos[0], target_pos[1], marker='x', s=80, label="target")

    # current spacecraft position
    r = X[k, :3]
    q = Q_hist[k]
    ax.scatter(r[0], r[1], marker='o', s=40, label="chaser")

    # body +x axis in inertial frame
    body_x = np.array([1.0, 0.0, 0.0])
    dir_vec = rotate_vector_by_quaternion(q, body_x)

    ax.arrow(
        r[0], r[1],
        2.0 * dir_vec[0], 2.0 * dir_vec[1],   # scale for visibility
        head_width=0.5,
        length_includes_head=True,
        color='green'
    )

    # optional LOS vector to target
    los = target_pos - r
    los_norm = np.linalg.norm(los)
    if los_norm > 1e-12:
        los = los / los_norm
        ax.arrow(
            r[0], r[1],
            2.0 * los[0], 2.0 * los[1],
            head_width=0.5,
            length_includes_head=True,
            color='orange'
        )

    mode = log["mode"][k]
    ax.set_title(f"t = {log['time'][k]:.1f} s, mode = {mode}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="upper right")

    plt.pause(0.01)

plt.ioff()
plt.show()


# NOTE: saveable FuncAnimation animation; use this primarily just to export

fig, ax = plt.subplots(figsize=(7,7))

X = log["x_trans"]
Q_hist = log["q"]
wp = np.array(waypoints)

def update(k):
    ax.clear()

    r = X[k, :3]
    q = Q_hist[k]

    # trajectory so far
    ax.plot(X[:k+1, 0], X[:k+1, 1], linewidth=1)

    # waypoints and target
    ax.scatter(wp[:, 0], wp[:, 1], marker='s')
    ax.scatter(target_pos[0], target_pos[1], marker='x', s=80)

    # current spacecraft position
    ax.scatter(r[0], r[1], marker='o')

    # green arrow: body +x axis / boresight
    body_x = np.array([1.0, 0.0, 0.0])
    dir_vec = rotate_vector_by_quaternion(q, body_x)

    ax.arrow(
        r[0], r[1],
        2.0 * dir_vec[0], 2.0 * dir_vec[1],
        head_width=0.5,
        length_includes_head=True,
        color='green'
    )

    # orange arrow: line of sight to target
    los = target_pos - r
    los_norm = np.linalg.norm(los)
    if los_norm > 1e-12:
        los_unit = los / los_norm
        ax.arrow(
            r[0], r[1],
            2.0 * los_unit[0], 2.0 * los_unit[1],
            head_width=0.5,
            length_includes_head=True,
            color='orange'
        )

    ax.set_title(f"t = {log['time'][k]:.1f} s, mode = {log['mode'][k]}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)

anim = FuncAnimation(fig, update, frames=len(X), interval=20)
anim.save("mission.mp4", fps=20)

plt.show()