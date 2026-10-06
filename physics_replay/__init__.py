"""Physics pick-and-place replay: Panda source demo -> UR5e + Robotiq 2F-85 target.

Frame convention: ``T_A_B`` is a 4x4 homogeneous transform that maps coordinates
expressed in frame B into frame A. Quaternions follow MuJoCo order (w, x, y, z).
Missing poses are stored as NaN, never as plausible coordinates.
"""
