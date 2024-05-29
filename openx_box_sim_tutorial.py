import numpy as np
import mujoco
import mujoco_viewer
import glfw
from OpenX_Cube_simulator_mujoco2 import OpenX_Simulator_Cube
import matplotlib.pyplot as plt

# Initialize simulator and controller
sim = OpenX_Simulator_Cube(render=True)

# Fixed z-height for the spotlight
fixed_z_height = 0.5

# Cube dimensions (half extents)
cube_half_extent = 0.05

# Initiate list to store contact locations
contacts = []  # Initialize as an empty list
cube_edges = []  # List to store the edges of the cube

# Function to handle key presses
def handle_key_presses(sim):
    # Handle sphere movement
    sphere_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'sphero_free_joint')
    qpos_addr_sphere = sim.model.jnt_qposadr[sphere_joint_id]
    pos_sphere = sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3].copy()  # Get the position part of the free joint (first 3 components)
    if glfw.get_key(sim.viewer.window, glfw.KEY_2) == glfw.PRESS:
        pos_sphere[1] += 0.0005  # Move forward
    if glfw.get_key(sim.viewer.window, glfw.KEY_W) == glfw.PRESS:
        pos_sphere[1] -= 0.0005  # Move backward
    if glfw.get_key(sim.viewer.window, glfw.KEY_Q) == glfw.PRESS:
        pos_sphere[0] -= 0.0005  # Move left
    if glfw.get_key(sim.viewer.window, glfw.KEY_E) == glfw.PRESS:
        pos_sphere[0] += 0.0005  # Move right
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

    mujoco.mj_forward(sim.model, sim.data)

# Function to check contacts and print contact info
def check_contacts(sim, cont_arr):
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        geom1 = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if (geom1 == 'white_sphero' and geom2 != 'floor') or (geom1 != 'floor' and geom2 == 'white_sphero'):
            print("Contact detected between sphero and", geom1 if geom1 != 'white_sphero' else geom2)
            pos_contact = contact.pos[:2].copy()  # Get only the x and y coordinates and copy the array
            cont_arr.append(pos_contact)  # Append to the list
            print(f"Contact position: {pos_contact}")

# Function to calculate cube edges
def calculate_cube_edges(cube_pos):
    # Calculate the positions of the edges of the cube
    edges = [
        cube_pos + np.array([cube_half_extent, cube_half_extent]),
        cube_pos + np.array([-cube_half_extent, cube_half_extent]),
        cube_pos + np.array([cube_half_extent, -cube_half_extent]),
        cube_pos + np.array([-cube_half_extent, -cube_half_extent])
    ]
    return edges

# Function to zero out the rotational velocity of the sphere
def zero_rotational_velocity(sim, sphere_joint_id):
    qvel_addr_sphere = sim.model.jnt_dofadr[sphere_joint_id]
    sim.data.qvel[qvel_addr_sphere+3:qvel_addr_sphere+6] = 0  # Zero out the rotational part of the velocity

# Run the simulation
sim_time = 15.0
step_count = 0
render_interval = 10  # Render every 10 steps to reduce performance impact
sphere_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'sphero_free_joint')

# IDs of all green cube bodies
cube_body_ids = [
    mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'cube'),
    mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'cube2'),
    mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'cube3')
]

while sim.t < sim_time:
    if sim.viewer.is_alive:
        handle_key_presses(sim)
        check_contacts(sim, contacts)
        zero_rotational_velocity(sim, sphere_joint_id)
        ctrl = np.zeros(sim.model.nu)
        sim.step(ctrl)
        
        if step_count % render_interval == 0:
            sim.viewer.render()

        # Get the cube positions and calculate their edges
        for cube_body_id in cube_body_ids:
            cube_pos = sim.data.xpos[cube_body_id][:2]
            cube_edges.extend(calculate_cube_edges(cube_pos))
        
        print(f"Step: {step_count}, Simulation time: {sim.t}")
        step_count += 1
    else:
        break

sim.close_sim()

# Convert the contacts and edges lists to NumPy arrays
contacts = np.array(contacts)
cube_edges = np.array(cube_edges)
print("Contacts:", contacts)
print("Cube edges:", cube_edges)

# Plot the contact positions and the edges of the cube
if contacts.size > 0:
    plt.scatter(contacts[:, 0], contacts[:, 1], c='red', marker='o', label='Contact Points', s=10)
if cube_edges.size > 0:
    plt.scatter(cube_edges[:, 0], cube_edges[:, 1], c='green', marker='o', label='Cube Edges', s=10)
plt.title('Contact Positions and Cube Edges')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.legend()
plt.show()