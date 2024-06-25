import numpy as np
import mujoco
import mujoco_viewer

class OpenX_Simulator_Cube(): 

    def __init__(self, max_step=500, model_xml="./green_cube_verify.xml", render=True):
        
        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.data  = mujoco.MjData(self.model)

        self.render = render
        if self.render: self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        if self.render: self.viewer.render()

        self.t = 0. 
        self.dt = self.model.opt.timestep
        self.stepcount = 0
        self.max_step = max_step
        self.render_step = 20
 
    def step(self, ctrl):
        self.data.ctrl[:] = ctrl# np.random.normal(0., 1.0, model.nu)
        mujoco.mj_step(self.model, self.data)
        if self.render:
            if self.viewer.is_alive: 
                if self.stepcount % self.render_step == 0: self.viewer.render()
            else: return
        self.stepcount += 1
        self.t += self.dt

    def close_sim(self):
        if self.render: self.viewer.close()
        print("Finished Simulation, step count: ", self.stepcount)

    def get_robot_joint_state(self): 
        return np.copy(self.data.qpos[:6])

    def get_robot_jointvel_state(self):
        return np.copy(self.data.qvel[:6])

    def get_robot_jointacc_state(self):
        return np.copy(self.data.qacc[:6])

    def get_robot_ee_state(self):
        return (np.copy(self.data.site_xmat[0].reshape((3,3))), np.copy(self.data.site_xpos[0])) 
        
    def get_box_state(self):
        box_pos = np.copy(self.data.qpos[6:9])
        box_orn = np.copy(self.data.qpos[9:13])
        return np.copy(self.data.qpos[6:])#, box_pos, box_orn

    def get_box_vel_state(self):
        box_pos = np.copy(self.data.qvel[6:9])
        box_orn = np.copy(self.data.qvel[9:12])
        return np.copy(self.data.qvel[6:])#, box_pos, box_orn

    def get_box_acc_state(self):
        box_pos = np.copy(self.data.qacc[6:9])
        box_orn = np.copy(self.data.qacc[9:12])
        return np.copy(self.data.qacc[6:])#, box_pos, box_orn

    def get_mass_matrix(self):
        mass_matrix = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, mass_matrix, self.data.qM)
        return mass_matrix[:6, :6]#for arm only

    def get_h_bias(self): 
        return self.data.qfrc_bias[:6] # for arm only

    def get_jacSite(self, site_name):
        # To match Modern Robotics textbook, Jacobian is rotation then position (i.e., [J_r,J_p])
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name) #site for end-effector
        jac_p = np.zeros((3,self.model.nv)) #position jacobian (wrt spatial twist position)
        jac_r = np.zeros((3,self.model.nv)) # rotation jacobian (wrt spatial twist orn)
        mujoco.mj_jacSite(self.model, self.data, jac_p, jac_r, site_id)
        # print(jac_r)
        return np.concatenate([jac_r, jac_p], axis=0)[:,:6] #for arm only, add [:,:6]

    def get_site_pose(self, site_name): 
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name) #site for end-effector
        p = self.data.site_xpos[site_id]
        R = self.data.site_xmat[site_id].reshape((3,3))
        return (R, p)
 
        