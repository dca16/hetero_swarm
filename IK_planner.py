import numpy as np
import mujoco
from OpenX_Cube_simulator_mujoco2 import OpenX_Simulator_Cube
import modern_robotics as mr

def calc_IK_step(R_d, p_d, R_cur, p_cur, Jb_cur):
    # convert Rp to trans
    T_d = mr.RpToTrans(np.array(R_d),np.array(p_d))
    T_cur = mr.RpToTrans(np.array(R_cur),np.array(p_cur))

    # convert to body frame
    T_bd = np.linalg.inv(T_cur) @ T_d

    # get twist
    twist_se3 = mr.MatrixLog6(T_bd)
    twist = mr.se3ToVec(twist_se3)

    # compute and return change in theta
    pseudo_inverse_jac = np.linalg.pinv(Jb_cur)
    theta_step = pseudo_inverse_jac @ twist
    return theta_step