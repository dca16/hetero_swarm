import numpy as np
import mujoco
import mujoco_viewer
import glfw
from OpenX_Cube_simulator_mujoco2 import OpenX_Simulator_Cube

# Initialize simulator and controller
sim = OpenX_Simulator_Cube(render=True)

# Fixed z-height for the spotlight
fixed_z_height = 0.5

# Function to handle key presses
def handle_key_presses(sim):
    # Handle sphere movement
    sphere_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'sphere_free_joint')
    qpos_addr_sphere = sim.model.jnt_qposadr[sphere_joint_id]
    pos_sphere = sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3].copy()  # Get the position part of the free joint (first 3 components)
    if glfw.get_key(sim.viewer.window, glfw.KEY_W) == glfw.PRESS:
        pos_sphere[1] += 0.0005  # Move forward
    if glfw.get_key(sim.viewer.window, glfw.KEY_S) == glfw.PRESS:
        pos_sphere[1] -= 0.0005  # Move backward
    if glfw.get_key(sim.viewer.window, glfw.KEY_A) == glfw.PRESS:
        pos_sphere[0] -= 0.0005  # Move left
    if glfw.get_key(sim.viewer.window, glfw.KEY_D) == glfw.PRESS:
        pos_sphere[0] += 0.0005  # Move right
    if glfw.get_key(sim.viewer.window, glfw.KEY_Q) == glfw.PRESS:
        pos_sphere[2] += 0.0005  # Move up
    if glfw.get_key(sim.viewer.window, glfw.KEY_E) == glfw.PRESS:
        pos_sphere[2] -= 0.0005  # Move down
    sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3] = pos_sphere  # Set the position part of the free joint

    # Handle spotlight movement
    spotlight_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'spotlight_free_joint')
    qpos_addr_spotlight = sim.model.jnt_qposadr[spotlight_joint_id]
    pos_spotlight = sim.data.qpos[qpos_addr_spotlight:qpos_addr_spotlight + 3].copy()  # Get the position part of the free joint (first 3 components)
    if glfw.get_key(sim.viewer.window, glfw.KEY_I) == glfw.PRESS:
        pos_spotlight[1] += 0.01  # Move forward
    if glfw.get_key(sim.viewer.window, glfw.KEY_K) == glfw.PRESS:
        pos_spotlight[1] -= 0.01  # Move backward
    if glfw.get_key(sim.viewer.window, glfw.KEY_J) == glfw.PRESS:
        pos_spotlight[0] -= 0.01  # Move left
    if glfw.get_key(sim.viewer.window, glfw.KEY_L) == glfw.PRESS:
        pos_spotlight[0] += 0.01  # Move right

    # Ensure the z-height of the spotlight remains fixed
    pos_spotlight[2] = fixed_z_height

    # Ensure the spotlight remains within visible bounds
    pos_spotlight = np.clip(pos_spotlight, -1, 1)
    sim.data.qpos[qpos_addr_spotlight:qpos_addr_spotlight + 3] = pos_spotlight  # Set the position part of the free joint

    # Log spotlight position for debugging
    #print(f"Spotlight position: {pos_spotlight}")

    mujoco.mj_forward(sim.model, sim.data)

# Function to check contacts and print contact info
def check_contacts(sim):
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        geom1 = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if (geom1 == 'green_cube' and geom2 == 'white_sphere') or (geom1 == 'white_sphere' and geom2 == 'green_cube'):
            print("Contact detected between sphere and cube")

# Function to zero out the rotational velocity of the sphere
def zero_rotational_velocity(sim, sphere_joint_id):
    qvel_addr_sphere = sim.model.jnt_dofadr[sphere_joint_id]
    sim.data.qvel[qvel_addr_sphere+3:qvel_addr_sphere+6] = 0  # Zero out the rotational part of the velocity

# Run the simulation
sim_time = 10.0
step_count = 0
render_interval = 10  # Render every 10 steps to reduce performance impact
sphere_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'sphere_free_joint')

while sim.t < sim_time:
    if sim.viewer.is_alive:
        handle_key_presses(sim)
        check_contacts(sim)
        zero_rotational_velocity(sim, sphere_joint_id)
        ctrl = np.zeros(sim.model.nu)
        sim.step(ctrl)
        
        if step_count % render_interval == 0:
            sim.viewer.render()
        
        print(f"Step: {step_count}, Simulation time: {sim.t}")
        step_count += 1
    else:
        break

sim.close_sim()