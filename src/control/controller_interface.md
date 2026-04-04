# Controller Interface Guide

How to plug a new controller into the sim infrastructure.

---

## One-sentence summary

Subclass `BaseController` in `src/control/`, implement `step()`, pass the instance to
`ControllerAdapter`. Nothing in `src/sim/` needs to change.

---

## Step 1 — Subclass BaseController

```python
# src/control/my_controller.py
import numpy as np
from sim.interface.sim_state import SimState
from sim.interface.control_command import ControlCommand
from control.base_controller import BaseController

class MyController(BaseController):

    def __init__(self, ...):
        ...

    def step(self, state: SimState) -> ControlCommand:
        # compute force and torque
        return ControlCommand(force=f, torque=tau, valid=True)
```

---

## Step 2 — Wire it up

```python
from control.my_controller import MyController
from adapter.controller_adapter import ControllerAdapter

ctrl    = MyController(...)
adapter = ControllerAdapter(ctrl, {"max_force": 50.0, "max_torque": 10.0})
```

Pass `adapter` (not `ctrl` directly) into the closed-loop. The adapter handles
force/torque clipping and sets `valid=False` if a limit is hit.

---

## SimState — what the controller receives

| Field | Shape | Frame | Units |
|---|---|---|---|
| `time` | scalar | — | s |
| `rel_pos` | (3,) | Hill (x=R-bar, y=V-bar, z=H-bar) | m |
| `rel_vel` | (3,) | Hill | m/s |
| `quaternion` | (4,) | scalar-last `[qx, qy, qz, qw]` | — |
| `omega` | (3,) | chaser body frame | rad/s |

`rel_pos` and `rel_vel` are chaser minus target, projected into the Hill frame
defined by the **target's** current position and velocity.

## ControlCommand — what the controller must return

| Field | Shape | Frame | Units |
|---|---|---|---|
| `force` | (3,) | Hill | N |
| `torque` | (3,) | chaser body frame | N·m |
| `valid` | bool | — | — |

Set `valid=False` if the solver fails or returns an infeasible solution.
The sim records this flag but does not stop; it applies whatever force/torque
was returned (clipped to limits).

---

## Known caveats

- **`rel_vel` is approximate.** The `ω × Δr` frame-rotation term is not subtracted
  (~0.1 m/s error for 100 m separation on ISS orbit). Flagged in `bsk_interface.py`.
- **Force is expressed in Hill frame.** `BskInterface` rotates it to inertial before
  applying to the spacecraft — the controller should not do this itself.
- **Torque is expressed in body frame.** Applied directly; no rotation is performed.

---

## Reference files

| File | Role |
|---|---|
| `src/control/base_controller.py` | ABC to subclass |
| `src/control/pd_controller.py` | Example implementation |
| `src/sim/interface/sim_state.py` | `SimState` definition |
| `src/sim/interface/control_command.py` | `ControlCommand` definition |
| `src/sim/adapter/controller_adapter.py` | Clipping + dispatch |
| `src/sim/core/bsk_interface.py` | Basilisk ↔ interface bridge |
| `src/sim/tests/test_closed_loop_pd.py` | Full closed-loop example |
