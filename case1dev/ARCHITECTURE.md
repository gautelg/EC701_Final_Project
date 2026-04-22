# System Architecture Overview

This codebase implements a modular simulation of spacecraft proximity operations using a sequential guidance and control strategy. The system separates translational control, attitude control, supervisory logic, and safety filtering into distinct components for clarity and extensibility.

---

## Overall control/data flow

1. `main_sim.py` queries `mission_manager.py` for current mode and active waypoint.
2. If mode is `TRANSLATE`:
   - `case1_translation_draft.py` computes nominal MPC control `u_nom`
   - `cbf.py` filters `u_nom` into safe control `u_trans`
   - `main_sim.py` propagates translational dynamics
3. If mode is `ROTATE`:
   - `case1_attitude_draft.py` computes desired pointing quaternion from current relative position
   - `case1_attitude_draft.py` computes torque command and propagates attitude dynamics
4. `main_sim.py` logs all states, controls, errors, and mode history
5. `mission_manager.py` updates waypoint index / mode transitions based on thresholds

---

## Module details

### 1. `translation_controller.py`
Handles relative translational motion using Hill-Clohessy-Wiltshire (HCW) dynamics.

**Responsibilities:**
- Define continuous-time HCW dynamics in the LVLH frame
- Discretize dynamics for control
- Solve a finite-horizon MPC problem (quadratic program via CVXPY + OSQP)
- Provide nominal translational control input `u_nom`

**Key Interface:**
- Input: current state `x_trans`, reference state `x_ref`
- Output: nominal control `u_nom`

---

### 2. `attitude_controller.py`
Handles spacecraft orientation using quaternion-based rigid-body dynamics.

**Responsibilities:**
- Implement quaternion algebra and normalization
- Compute desired pointing quaternion based on line-of-sight (LOS)
- Compute quaternion error
- Apply PD-based torque control
- Propagate attitude dynamics

**Key Interface:**
- Input: current attitude `(q, ω)`, desired attitude `q_des`
- Output: control torque `τ`, updated `(q, ω)`

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
Top-level simulation orchestrator.

**Responsibilities:**
- Initialize system parameters, gains, and initial conditions
- Run closed-loop simulation over time
- Coordinate all modules:
  - Query `mission_manager` for current mode
  - Call appropriate controller (translation or attitude)
  - Apply safety filter (CBF) in translation mode
- Propagate system states
- Log all relevant data (states, controls, errors, mode)
- Generate plots and animations

---

## Control Flow

At each simulation timestep:

1. **Supervisor**
   - `mission_manager` determines current mode and active waypoint

2. **Translation Mode**
   - Compute nominal control via MPC (`u_nom`)
   - Apply CBF safety filter → `u_trans`
   - Propagate translational dynamics

3. **Rotation Mode**
   - Compute desired pointing quaternion (`q_des`)
   - Compute torque via PD controller (`τ`)
   - Propagate attitude dynamics

4. **Logging**
   - Store time, mode, states, inputs, and errors

5. **Mode Update**
   - `mission_manager` updates mode based on thresholds

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