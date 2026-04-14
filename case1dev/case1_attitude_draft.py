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
    wdot = np.linalg.inv(J) @ (tau - np.cross(omega, J @ omega))
    return qdot, wdot

def quat_error(q_des, q):
    q_inv = quat_conj(q)    # unit quaternion inverse
    q_e = quat_mul(q_inv, q_des)   # <-- flipped order
    q_e = quat_norm(q_e)

    if q_e[0] < 0:
        q_e = -q_e
    return q_e

    # enforce shortest-rotation convention
    if q_e[0] < 0:
        q_e = -q_e
    return q_e

def attitude_pd_control(q, omega, q_des, Kq, Kw, tau_max):
    q_e = quat_error(q_des, q)
    e_vec = q_e[1:]  # vector part
    tau = -Kq @ e_vec - Kw @ omega
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


J = np.diag([8.0, 6.0, 5.0])

q0 = np.array([1.0, 0.0, 0.0, 0.0])     # identity attitude
w0 = np.array([0.0, 0.0, 0.0])

# Example desired orientation: 90 deg about z-axis
theta = np.pi / 2
q_des = np.array([np.cos(theta/2), 0.0, 0.0, np.sin(theta/2)])

Kq = np.diag([2.0, 2.0, 2.0])
Kw = np.diag([3.0, 3.0, 3.0])
tau_max = np.array([0.05, 0.05, 0.05])

dt = 0.05
T = 15.0

Q, W, Tau, Err = simulate_attitude(
    q0, w0, q_des, J, Kq, Kw, tau_max, dt=dt, T=T
)

plot_attitude_results(Q, W, Tau, Err, dt)