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

# Grid for ergodic exploration
grid_size = 100
visit_count = np.zeros((grid_size, grid_size))
contact_influence_steps = 100  # Number of steps to stay influenced by contact

# Function to move sphere randomly
def move_sphere_randomly(sim):
    sphere_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'sphero_free_joint')
    qpos_addr_sphere = sim.model.jnt_qposadr[sphere_joint_id]
    pos_sphere = sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3].copy()  # Get the position part of the free joint (first 3 components)

    # Random movement within a small range
    pos_sphere[:2] += np.random.uniform(-0.01, 0.01, size=2)  # Move in x and y directions randomly

    # Ensure the sphere remains within bounds
    pos_sphere[:2] = np.clip(pos_sphere[:2], -1, 1)
    pos_sphere[2] = np.clip(pos_sphere[2], 0.05, 0.5)

    sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3] = pos_sphere  # Set the position part of the free joint

# Function to move the sphere efficiently
def move_sphere_efficiently(sim):
    sphere_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'sphero_free_joint')
    qpos_addr_sphere = sim.model.jnt_qposadr[sphere_joint_id]
    pos_sphere = sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3].copy()  # Get the position part of the free joint (first 3 components)

    # Structured movement pattern to cover the space
    step_size = 0.01  # Adjust step size as needed
    directions = [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size)]
    direction = directions[np.random.choice(len(directions))]

    pos_sphere[:2] += direction

    # Ensure the sphere remains within bounds
    pos_sphere[0] = np.clip(pos_sphere[0], -1, 1)
    pos_sphere[1] = np.clip(pos_sphere[1], -0.75, 0.75)

    sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3] = pos_sphere  # Set the position part of the free joint

# Function to move the sphere using an ergodic algorithm
def move_sphere_ergodically(sim, step_count):
    sphere_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'sphero_free_joint')
    qpos_addr_sphere = sim.model.jnt_qposadr[sphere_joint_id]
    pos_sphere = sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3].copy()  # Get the position part of the free joint (first 3 components)

    # Update visit count
    x_idx = int((pos_sphere[0] + 1) / 2 * (grid_size - 1))
    y_idx = int((pos_sphere[1] + 0.75) / 1.5 * (grid_size - 1))
    visit_count[x_idx, y_idx] += 1

    # Calculate movement direction based on visit count
    move_directions = [
        (0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005)
    ]
    np.random.shuffle(move_directions)  # Randomize the order of directions
    best_direction = None
    min_visit = np.inf
    for direction in move_directions:
        new_x_pos = pos_sphere[0] + direction[0]
        new_y_pos = pos_sphere[1] + direction[1]

        if -1 <= new_x_pos <= 1 and -0.75 <= new_y_pos <= 0.75:
            new_x_idx = int((new_x_pos + 1) / 2 * (grid_size - 1))
            new_y_idx = int((new_y_pos + 0.75) / 1.5 * (grid_size - 1))
            if visit_count[new_x_idx, new_y_idx] < min_visit:
                min_visit = visit_count[new_x_idx, new_y_idx]
                best_direction = direction

    # Move in the best direction
    if best_direction is not None:
        pos_sphere[:2] += best_direction

    # Ensure the sphere remains within bounds
    pos_sphere[0] = np.clip(pos_sphere[0], -1, 1)
    pos_sphere[1] = np.clip(pos_sphere[1], -0.75, 0.75)

    sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3] = pos_sphere  # Set the position part of the free joint

    # Reset attraction to contact points after a certain number of steps
    if step_count % contact_influence_steps == 0:
        visit_count[:] = np.maximum(visit_count, 0)  # Ensure visit count does not go negative


    # Move in the best direction
    if best_direction is not None:
        pos_sphere[:2] += best_direction

    # Ensure the sphere remains within bounds
    pos_sphere[0] = np.clip(pos_sphere[0], -1, 1)
    pos_sphere[1] = np.clip(pos_sphere[1], -0.75, 0.75)

    sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3] = pos_sphere  # Set the position part of the free joint

# Function to handle key presses for the spotlight
def handle_key_presses(sim):
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
        if (geom1 == 'white_sphero' and geom2 != 'floor' and not geom2.startswith('wall')) or (geom2 == 'white_sphero' and geom1 != 'floor' and not geom1.startswith('wall')):
            print("Contact detected between sphero and", geom1 if geom1 != 'white_sphero' else geom2)
            pos_contact = contact.pos[:2].copy()  # Get only the x and y coordinates and copy the array
            cont_arr.append(pos_contact)  # Append to the list
            print(f"Contact position: {pos_contact}")

            # Calculate grid indices for the contact position
            x_idx = int((pos_contact[0] + 1) / 2 * (grid_size - 1))
            y_idx = int((pos_contact[1] + 0.75) / 1.5 * (grid_size - 1))

            # Increase the value around the contact point
            contact_radius = 5
            for dx in range(-contact_radius, contact_radius + 1):
                for dy in range(-contact_radius, contact_radius + 1):
                    new_x_idx = x_idx + dx
                    new_y_idx = y_idx + dy
                    if 0 <= new_x_idx < grid_size and 0 <= new_y_idx < grid_size:
                        visit_count[new_x_idx, new_y_idx] -= 5  # Decrease visit count to increase attractiveness

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
        move_sphere_ergodically(sim, step_count)  # Move the sphere using an ergodic algorithm
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

# At the end of the simulation, plot the "visit_count" as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(visit_count.T, cmap='hot', origin='lower', extent=[-1, 1, -0.75, 0.75])
plt.colorbar(label='Visit Count')
plt.title('Heatmap of Visit Count')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()