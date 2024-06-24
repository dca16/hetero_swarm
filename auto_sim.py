import numpy as np
import mujoco
import mujoco_viewer
import glfw
import sys  # Added for command-line arguments
import argparse
from OpenX_Cube_simulator_mujoco2 import OpenX_Simulator_Cube
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil

# Initialize simulator and controller
sim = OpenX_Simulator_Cube(render=True)

# Fixed z-height for the spotlight
fixed_z_height = 0.4

# Cube dimensions (half extents)
cube_half_extent = 0.05

# Initiate list to store contact locations
contacts = []  # Initialize as an empty list
cube_edges = []  # List to store the edges of the cube

# List to store captured images
captured_images = []

# Directory to save images
output_dir = "captured_images"

# Clear the captured images folder
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Grid for ergodic exploration
grid_size = 100
visit_count = np.zeros((grid_size, grid_size))
contact_influence_steps = 100  # Number of steps to stay influenced by contact

# Variable to keep track of the current camera
current_camera = 'fixed'
camera_switched = False  # Flag to debounce camera switching

# Function to move sphere randomly
def move_sphere_randomly(sim, step_count):  # Added step_count parameter
    sphere_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'sphero_free_joint')
    qpos_addr_sphere = sim.model.jnt_qposadr[sphere_joint_id]
    pos_sphere = sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3].copy()

    # Random movement within a small range
    pos_sphere[:2] += np.random.uniform(-0.01, 0.01, size=2)  # Move in x and y directions randomly

    # Ensure the sphere remains within bounds
    pos_sphere[:2] = np.clip(pos_sphere[:2], -1, 1)
    pos_sphere[2] = np.clip(pos_sphere[2], 0.05, 0.5)

    sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3] = pos_sphere  # Set the position part of the free joint

# Function to move the sphere efficiently
def move_sphere_efficiently(sim, step_count):  # Added step_count parameter
    sphere_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'sphero_free_joint')
    qpos_addr_sphere = sim.model.jnt_qposadr[sphere_joint_id]
    pos_sphere = sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3].copy()

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
def move_sphere_ergodically(sim, step_count):  # Added step_count parameter
    sphere_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'sphero_free_joint')
    qpos_addr_sphere = sim.model.jnt_qposadr[sphere_joint_id]
    pos_sphere = sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3].copy()

    # Update visit count
    x_idx = int((pos_sphere[0] + 1) / 2 * (grid_size - 1))
    y_idx = int((pos_sphere[1] + 0.75) / 1.5 * (grid_size - 1))
    visit_count[x_idx, y_idx] += 1

    # Evaluate the entries in each direction and pick the direction that moves towards the entries with the lowest values
    move_directions = [
        (0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005)
    ]
    direction_values = []
    for direction in move_directions:
        new_x_pos = pos_sphere[0] + direction[0]
        new_y_pos = pos_sphere[1] + direction[1]

        if -1 <= new_x_pos <= 1 and -0.75 <= new_y_pos <= 0.75:
            new_x_idx = int((new_x_pos + 1) / 2 * (grid_size - 1))
            new_y_idx = int((new_y_pos + 0.75) / 1.5 * (grid_size - 1))
            direction_values.append((visit_count[new_x_idx, new_y_idx], direction))
        else:
            direction_values.append((np.inf, direction))  # Assign a high value if out of bounds

    # Find the directions with the minimum visit count values
    min_value = min(direction_values, key=lambda x: x[0])[0]
    best_directions = [direction for value, direction in direction_values if value == min_value]

    # Pick a direction randomly from the best directions
    best_direction = best_directions[np.random.choice(len(best_directions))]

    # Move in the best direction
    pos_sphere[:2] += best_direction

    # Ensure the sphere remains within bounds
    pos_sphere[0] = np.clip(pos_sphere[0], -1, 1)
    pos_sphere[1] = np.clip(pos_sphere[1], -0.75, 0.75)

    sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3] = pos_sphere  # Set the position part of the free joint

    # Reset attraction to contact points after a certain number of steps
    if step_count % contact_influence_steps == 0:
        visit_count[:] = np.maximum(visit_count, 0)  # Ensure visit count does not go negative
# Function to handle key presses for the spotlight
def handle_key_presses(sim):
    # Use function attribute to initialize camera_switched only once
    if not hasattr(handle_key_presses, 'camera_switched'):
        handle_key_presses.camera_switched = False
        handle_key_presses.current_camera = 'fixed'

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

    # Ensure the spotlight remains within visible bounds
    pos_spotlight = np.clip(pos_spotlight, -1, 1)
    sim.data.qpos[qpos_addr_spotlight:qpos_addr_spotlight + 3] = pos_spotlight  # Set the position part of the free joint

    # Handle camera toggling
    if glfw.get_key(sim.viewer.window, glfw.KEY_T) == glfw.PRESS:
        if not handle_key_presses.camera_switched:
            handle_key_presses.current_camera = 'spotlight' if handle_key_presses.current_camera == 'fixed' else 'fixed'
            handle_key_presses.camera_switched = True
    else:
        handle_key_presses.camera_switched = False

    if handle_key_presses.current_camera == 'spotlight':
        sim.viewer.cam.fixedcamid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, 'spotlight_camera')
    else:
        sim.viewer.cam.fixedcamid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, 'fixed')

    sim.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    mujoco.mj_forward(sim.model, sim.data)

# Function to check contacts and print contact info
def check_contacts(sim, cont_arr, step_count):
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        geom1 = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if (geom1 == 'white_sphero' and geom2 != 'floor' and not geom2.startswith('wall')) or (geom2 == 'white_sphero' and geom1 != 'floor' and not geom1.startswith('wall')):
            print("Contact detected between sphero and", geom1 if geom1 != 'white_sphero' else geom2)
            pos_contact = contact.pos[:2].copy()  # Get only the x and y coordinates and copy the array
            normal_contact = contact.frame[:3].copy()  # Get the normal direction of the contact
            cont_arr.append((pos_contact, step_count, normal_contact))  # Append to the list
            print(f"Contact position: {pos_contact}, time step: {step_count}, normal: {normal_contact}")

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

# Function to capture images from the spotlight camera
def capture_image(sim, viewer, step_count):
    width, height = viewer.viewport.width, viewer.viewport.height
    rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.zeros((height, width, 1), dtype=np.float32)

    # Get the spotlight camera ID
    spotlight_cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, 'spotlight_camera')

    # Set the camera to the spotlight camera temporarily
    original_camid = viewer.cam.fixedcamid
    original_camtype = viewer.cam.type

    viewer.cam.fixedcamid = spotlight_cam_id
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    # Update the scene with the spotlight camera
    mujoco.mjv_updateScene(sim.model, sim.data, viewer.vopt, None, viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, viewer.scn)

    # Render the scene
    mujoco.mjr_render(viewer.viewport, viewer.scn, viewer.ctx)

    # Read the pixels from the buffer
    mujoco.mjr_readPixels(rgb_buffer, depth_buffer, viewer.viewport, viewer.ctx)

    # Save the image
    image = Image.fromarray(rgb_buffer)
    image.save(os.path.join(output_dir, f"step_{step_count}.png"))

    # Restore the original camera
    viewer.cam.fixedcamid = original_camid
    viewer.cam.type = original_camtype

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

def plot_contacts_and_edges(contacts_array, cube_edges_array):
    # Check if contacts_array is defined and has data
    if contacts_array.size > 0:
        plt.scatter(contacts_array['position'][:, 0], contacts_array['position'][:, 1], c='red', marker='o', label='Contact Points', s=10)
    else:
        print("No contact points to plot")

    # Check if cube_edges_array is defined and has data
    if cube_edges_array.size > 0:
        plt.scatter(cube_edges_array[:, 0], cube_edges_array[:, 1], c='green', marker='o', label='Cube Edges', s=10)
    else:
        print("No cube edges to plot")

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

# Function to zero out the rotational velocity of the sphere
def zero_rotational_velocity(sim, sphere_joint_id):
    qvel_addr_sphere = sim.model.jnt_dofadr[sphere_joint_id]
    sim.data.qvel[qvel_addr_sphere+3:qvel_addr_sphere+6] = 0  # Zero out the rotational part of the velocity

# Run the simulation
def run_simulation(movement_function):
    # Simulation settings
    sim_time = 3.0
    step_count = 0
    render_interval = 10  # Render every 10 steps to reduce performance impact
    sphere_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'sphero_free_joint')

    # IDs of all green cube bodies
    cube_body_ids = [
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'cube'),
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'cube2'),
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'cube3')
    ]
    
    contacts = []
    cube_edges = []

    while sim.t < sim_time:
        if sim.viewer.is_alive:
            movement_function(sim, step_count)  # Use the chosen movement function
            handle_key_presses(sim)
            check_contacts(sim, contacts, step_count)  # Pass the current step count to the check_contacts function
            zero_rotational_velocity(sim, sphere_joint_id)
            ctrl = np.zeros(sim.model.nu)
            sim.step(ctrl)
            
            if step_count % render_interval == 0:
                sim.viewer.render()
                capture_image(sim, sim.viewer, step_count)  # Capture image at current time step

            # Get the cube positions and calculate their edges
            for cube_body_id in cube_body_ids:
                cube_pos = sim.data.xpos[cube_body_id][:2]
                cube_edges.extend(calculate_cube_edges(cube_pos))  # Ensure edges are added correctly
            
            print(f"Step: {step_count}, Simulation time: {sim.t}")
            step_count += 1
        else:
            break

    sim.close_sim()

    # Convert the contacts list to a structured NumPy array
    if contacts:  # Ensure contacts are not empty
        contacts_array = np.array(contacts, dtype=[('position', float, (2,)), ('time_step', int), ('normal', float, (3,))])
        for c in contacts:
            print(c)
    else:
        print("No contacts recorded")
        contacts_array = np.array([], dtype=[('position', float, (2,)), ('time_step', int), ('normal', float, (3,))])

    # Convert the cube_edges list to a structured NumPy array
    if cube_edges:  # Ensure cube_edges are not empty
        cube_edges_array = np.array(cube_edges)
        print("Cube edges:", cube_edges_array)
    else:
        print("No cube edges recorded")
        cube_edges_array = np.array([])

    return contacts_array, cube_edges_array

# Convert the contacts list to a structured NumPy array
contacts_array = np.array(contacts, dtype=[('position', float, (2,)), ('time_step', int), ('normal', float, (3,))])
for c in contacts:
    print(c)

# Convert the cube_edges list to a NumPy array
cube_edges = np.array(cube_edges)
print("Cube edges:", cube_edges)

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose the movement function for the simulation.")
    parser.add_argument("movement", choices=["random", "efficient", "ergodic"], help="Movement function to use: 'random', 'efficient', or 'ergodic'")
    args = parser.parse_args()

    if args.movement == "random":
        contacts_array, cube_edges_array = run_simulation(move_sphere_randomly)
    elif args.movement == "efficient":
        contacts_array, cube_edges_array = run_simulation(move_sphere_efficiently)
    elif args.movement == "ergodic":
        contacts_array, cube_edges_array = run_simulation(move_sphere_ergodically)

    plot_contacts_and_edges(contacts_array, cube_edges_array)  # Call the plotting function after the simulation