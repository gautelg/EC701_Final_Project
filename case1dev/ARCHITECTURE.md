# Autonomous Rendezvous, Proximity Operations, and Docking (ARPOD) -- Sequential Hierarchical Controller

This codebase implements a modular simulation of spacecraft proximity operations subject to Hill-Clohessy-Wiltshire (HCW) orbital dynamics using a sequential guidance and control strategy, specifically to command a chase craft to safely approach, maneuver around, and observe a target craft. The hierarchical system separates translational control (MPC), quaternion-based attitude control, hybrid supervisory mode switching, and safety filtering (second-order CBF) into distinct components for clarity and extensibility.


## Simulation Demo

https://github.com/user-attachments/assets/7ed397fd-a6b4-4f7f-9457-98def522e20e

> *Simulation of an autonomous inspection trajectory around a target spacecraft in the Local Vertical Local Horizontal (LVLH) frame. While this scenario executes a coplanar waypoint sequence, the underlying controller tracks full 6-DOF state vectors, i.e., $[x, y, z, \dot{x}, \dot{y}, \dot{z}]^T$ and supports arbitrary 3D relative trajectories.*


---

## Control Flow

At each simulation timestep:

1. `main_sim.py` queries `mission_manager.py` for current mode and active waypoint.
2. If mode is `TRANSLATE`:
   - `translation_controller.py` computes nominal MPC control `u_nom`
   - `cbf.py` filters `u_nom` into safe control `u_trans`
   - `main_sim.py` propagates translational dynamics (and togglable attitude drift)
3. If mode is `ROTATE`:
   - `attitude_controller.py` computes desired pointing quaternion `q_des` from current relative position and torque command `τ`
   - `main_sim.py` propagates rigid-body attitude dynamics (and togglable orbital drift)
4. `main_sim.py` logs all states, controls, errors, and mode history
5. `mission_manager.py` updates waypoint index and mode transitions based on state tolerances

---

## Module details

### 1. `translation_controller.py`
Defines relative translational dynamics and computes nominal optimal control using Hill-Clohessy-Wiltshire (HCW) equations.

**Responsibilities:**
- Define continuous-time HCW dynamics in the LVLH frame
- Discretize system dynamics for prediction and simulation
- Solve a finite-horizon MPC problem (quadratic program via CVXPY + OSQP)
- Provide nominal translational control input `u_nom`
- (includes a `__main__` block for standalone unit testing)

**Key Interface:**
- Input: current state `x_trans`, reference state `x_ref`
- Output: nominal control `u_nom` (and discrete state-space matrices `Ad`, `Bd`)

---

### 2. `attitude_controller.py`
Defines spacecraft rigid-body attitude dynamics and computes quaternion-based pointing control.

**Responsibilities:**
- Implement scalar-first Hamilton quaternion algebra and normalization
- Compute desired pointing quaternion `q_des` based on line-of-sight (LOS) to target
- Compute shortest-path quaternion error `q_e`
- Compute bounded PD control torque `τ` (subject to actuator limits `±τ_max`)
- Define rigid-body Euler/quaternion state derivatives `(qdot, wdot)`
- (includes a `__main__` block for standalone unit testing)

**Key Interface:**
- Input: current attitude `(q, ω)`, desired attitude `q_des`
- Output: control torque `τ`, quaternion error `q_e`, and state derivatives `(qdot, wdot)`

---

### 3. `mission_manager.py`
Implements a supervisory finite-state machine for sequential mission execution.

**Responsibilities:**
- Manage mode switching between:
  - `TRANSLATE` (move to waypoint)
  - `ROTATE` (orient toward target)
- Track current waypoint index
- Evaluate completion conditions:
  - Translation: position and velocity tolerances
  - Rotation: attitude and angular rate tolerances
- Prevent mode-switching chatter using persistence counters

**Key Interface:**
- Input: current state, errors
- Output: mode, active waypoint, completion status

---

### 4. `cbf.py`
Implements a Control Barrier Function (CBF)-based safety filter for translation.

**Responsibilities:**
- Enforce a spherical keep-out zone around the target
- Modify nominal control input `u_nom` minimally
- Solve a QP to produce safe control `u_trans`
- Enforce actuator bounds and optional slack for feasibility

**Key Interface:**
- Input: current state `x_trans`, nominal control `u_nom`
- Output: safe control `u_trans`

---

### 5. `main_sim.py`
Top-level closed-loop simulation orchestrator.

**Responsibilities:**
- Initialize system parameters, gains, and initial conditions
- Run closed-loop sequential mission over time
- Coordinate all modules:
  - Query `mission_manager` for current mode
  - Call appropriate controller (`translation_controller` or `attitude_controller`)
  - Apply safety filter (`cbf`) in translation mode
- Integrate system states (including optional natural drift during inactive sequential phases)
- Log all relevant data (states, controls, errors, mode)
- Generate plots and animations

---

## Design Principles

- **Modularity:** Each subsystem (translation, attitude, safety, supervision) is independent and reusable
- **Separation of concerns:**
  - MPC → performance
  - CBF → safety
  - Mission manager → logic
- **Extendability:** Components can be replaced (e.g., nonlinear MPC, Basilisk dynamics) without restructuring the system
- **Realism progression:** Current model uses linear HCW + rigid-body dynamics, but architecture supports higher-fidelity simulation

---

## Intended Use

This implementation serves as:
- A prototype for spacecraft guidance and control logic
- A baseline for comparison with higher-fidelity simulators (e.g., Basilisk)
- A framework for testing:
  - sequential vs. coupled control strategies
  - safety constraints (CBFs)
  - robustness to modeling assumptions

---

## Notes for Integration (e.g., Basilisk)

- Replace HCW propagation with full orbital dynamics
- Replace simple PD attitude control with reaction-wheel or torque models
- Keep:
  - mission manager (mode logic)
  - high-level guidance structure
  - safety filtering concept

The architecture is designed so that only the dynamics layer needs to change, while control and supervision logic remain intact.
