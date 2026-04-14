"""
i.e.,

run translational MPC until ||r - r_wp|| < \epsilon_r, ||v|| < \epsilon_v

switch to attitude mode

rotate until pointing error below threshold, bodyrate below threshold

switch back to translational mode and move on to next waypoint

"""



mode = "TRANSLATE"

if mode == "TRANSLATE":
    run translation MPC
    if near waypoint and slow enough:
        mode = "ROTATE"

elif mode == "ROTATE":
    set q_des for current waypoint
    run attitude controller
    if attitude error small and angular rate small:
        advance waypoint
        mode = "TRANSLATE"