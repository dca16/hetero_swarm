import numpy as np
from OpenX_Cube_simulator_mujoco2 import OpenX_Simulator_Cube
import mujoco
import modern_robotics as mr

class IKTorqueControl:
    def __init__(self, sim):
        self.sim = sim

    def get_torques(self, theta_d):
        # initialise K_p and K_d values
        K_p = 100.0
        K_d = 20.0
        
        # get M and h_bias
        M = self.sim.get_mass_matrix()[:-1, :-1]
        h_bias = self.sim.get_h_bias()[:-1]

        # get current pos and vel
        pos = self.sim.get_robot_joint_state()
        vel = self.sim.get_robot_jointvel_state()

        # compute desired theta_dd using equation from assignment description
        theta_dd_des = K_p*(theta_d - pos) - K_d*vel

        # compute and return tau
        tau = M @ theta_dd_des[:-1] + h_bias
        return tau