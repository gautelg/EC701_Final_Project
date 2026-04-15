# less realistic behavior, currently troubleshooting. ~JW 4.14.26

import numpy as np
import matplotlib.pyplot as plt

def skew(v):
    x, y, z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ], dtype=float)


# DEFINE QUATERNION PROPERTIES/OPERATIONS

def quat_mul(q1, q2):
    """Hamilton product. q = [q0, q1, q2, q3] with scalar-first convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def quat_norm(q):
    return q / np.linalg.norm(q)

def rotate_vector_by_quaternion(q, v):
    """
    Rotate vector v (3D) by quaternion q (scalar-first).
    """
    q_conj = quat_conj(q)
    v_quat = np.hstack([0.0, v])
    return quat_mul(quat_mul(q, v_quat), q_conj)[1:]

def quaternion_from_two_vectors(a, b):
    """
    Returns scalar-first unit quaternion rotating vector a onto vector b.
    Both a and b are 3-vectors.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    dot_ab = np.dot(a, b)

    # Same direction
    if np.isclose(dot_ab, 1.0):
        return np.array([1.0, 0.0, 0.0, 0.0])

    # Opposite direction: need any axis perpendicular to a
    if np.isclose(dot_ab, -1.0):
        # choose a convenient perpendicular axis
        if not np.isclose(a[0], 0.0):
            axis = np.array([-a[1], a[0], 0.0])
        else:
            axis = np.array([0.0, -a[2], a[1]])
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])  # 180 deg rotation

    axis = np.cross(a, b)
    q = np.array([1.0 + dot_ab, axis[0], axis[1], axis[2]])
    return quat_norm(q)

def compute_desired_pointing_quaternion(chaser_pos, target_pos):
    """
    Desired quaternion such that body +x axis points toward target.
    """
    chaser_pos = np.asarray(chaser_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)

    los = target_pos - chaser_pos
    los_norm = np.linalg.norm(los)

    if los_norm < 1e-12:
        # Degenerate case: target coincident with chaser
        return np.array([1.0, 0.0, 0.0, 0.0])

    los_unit = los / los_norm
    body_x = np.array([1.0, 0.0, 0.0])

    q_des = quaternion_from_two_vectors(body_x, los_unit)
    return quat_norm(q_des)

def omega_matrix(omega):
    wx, wy, wz = omega
    return np.array([
        [0,   -wx, -wy, -wz],
        [wx,   0,   wz, -wy],
        [wy,  -wz,  0,   wx],
        [wz,   wy, -wx,  0 ]
    ], dtype=float)

def quat_derivative(q, omega):
    return 0.5 * omega_matrix(omega) @ q

def attitude_dynamics(q, omega, tau, J):
    qdot = quat_derivative(q, omega)
    # wdot = np.linalg.inv(J) @ (tau - np.cross(omega, J @ omega))
    wdot = np.linalg.solve(J, tau - np.cross(omega, J @ omega))     # slightly more numerically stable
    return qdot, wdot

def quat_error(q_des, q):
    q_e = quat_mul(q_des, quat_conj(q))
    q_e = quat_norm(q_e)
    if q_e[0] < 0:  # enforce shortest-rotation convention
        q_e = -q_e
    return q_e

def attitude_pd_control(q, omega, q_des, Kq, Kw, tau_max):
    q_e = quat_error(q_des, q)
    e_vec = q_e[1:]  # vector part
    tau = Kq @ e_vec - Kw @ omega
    tau = np.clip(tau, -tau_max, tau_max)
    return tau, q_e

def simulate_attitude(q0, w0, q_des, J, Kq, Kw, tau_max, dt=0.05, T=20.0):
    steps = int(T / dt)
    q = quat_norm(q0.copy())
    w = w0.copy()

    Q_hist = [q.copy()]
    W_hist = [w.copy()]
    Tau_hist = []
    Err_hist = []

    for _ in range(steps):
        tau, q_e = attitude_pd_control(q, w, q_des, Kq, Kw, tau_max)

        qdot, wdot = attitude_dynamics(q, w, tau, J)

        q = q + dt * qdot
        q = quat_norm(q)
        w = w + dt * wdot

        Q_hist.append(q.copy())
        W_hist.append(w.copy())
        Tau_hist.append(tau.copy())
        Err_hist.append(q_e.copy())

    return (
        np.array(Q_hist),
        np.array(W_hist),
        np.array(Tau_hist),
        np.array(Err_hist)
    )

def plot_attitude_results(Q, W, Tau, Err, dt):
    t_q = np.arange(Q.shape[0]) * dt
    t_u = np.arange(Tau.shape[0]) * dt
    t_e = np.arange(Err.shape[0]) * dt

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # quaternion components
    ax = axs[0, 0]
    ax.plot(t_q, Q[:, 0], label='q0')
    ax.plot(t_q, Q[:, 1], label='q1')
    ax.plot(t_q, Q[:, 2], label='q2')
    ax.plot(t_q, Q[:, 3], label='q3')
    ax.set_title("Quaternion components")
    ax.set_xlabel("time")
    ax.grid(True)
    ax.legend()

    # angular velocity
    ax = axs[0, 1]
    ax.plot(t_q, W[:, 0], label='wx')
    ax.plot(t_q, W[:, 1], label='wy')
    ax.plot(t_q, W[:, 2], label='wz')
    ax.set_title("Angular velocity")
    ax.set_xlabel("time")
    ax.grid(True)
    ax.legend()

    # control torque
    ax = axs[1, 0]
    ax.plot(t_u, Tau[:, 0], label='tau_x')
    ax.plot(t_u, Tau[:, 1], label='tau_y')
    ax.plot(t_u, Tau[:, 2], label='tau_z')
    ax.set_title("Control torque")
    ax.set_xlabel("time")
    ax.grid(True)
    ax.legend()

    # quaternion error vector part
    ax = axs[1, 1]
    ax.plot(t_e, Err[:, 1], label='e1')
    ax.plot(t_e, Err[:, 2], label='e2')
    ax.plot(t_e, Err[:, 3], label='e3')
    ax.set_title("Quaternion error vector part")
    ax.set_xlabel("time")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()




    #######

if __name__ == "__main__":
    J = np.diag([8.0, 6.0, 5.0])

    q0 = np.array([1.0, 0.0, 0.0, 0.0])     # identity attitude
    w0 = np.array([0.0, 0.0, 0.0])

    # Example desired orientation: 90 deg about z-axis
    theta = np.pi / 2
    q_des = np.array([np.cos(theta/2), 0.0, 0.0, np.sin(theta/2)])

    Kq = np.diag([1.0, 1.0, 1.0])
    Kw = np.diag([6.0, 6.0, 6.0])
    tau_max = np.array([0.05, 0.05, 0.05])

    dt = 0.05
    T = 40

    Q, W, Tau, Err = simulate_attitude(
        q0, w0, q_des, J, Kq, Kw, tau_max, dt=dt, T=T
    )

    plot_attitude_results(Q, W, Tau, Err, dt)