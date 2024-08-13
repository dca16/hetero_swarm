import numpy as np
import mujoco
import mujoco_viewer
import glfw
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
from OpenX_Cube_simulator_mujoco2 import OpenX_Simulator_Cube

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
output_dir = "rcnn_images"

# Clear the captured images folder
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Variable to keep track of the current camera
current_camera = 'fixed'
camera_switched = False  # Flag to debounce camera switching

# Set grid size
gridsize_x = 10
gridsize_y = 8

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
    pos_spotlight = np.clip(pos_spotlight, -0.4, 0.4)
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

# Add this function to generate the raster path for a grid of any size
def generate_grid_path(x_range, y_range, gs_x, gs_y):
    x_values = np.linspace(x_range[0], x_range[1], gs_x)
    y_values = np.linspace(y_range[0], y_range[1], gs_y)
    path = []
    for y in y_values:
        for x in x_values:
            path.append((x, y))
    return path

# Generate the grid path
x_range = (-0.9, 0.9)
y_range = (-0.65, 0.65)
grid_path = generate_grid_path(x_range, y_range, gridsize_x, gridsize_y)

# Function to move the spotlight along the 4x4 grid path
def move_spotlight_grid(sim, step_count, grid_path):
    spotlight_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'spotlight_free_joint')
    qpos_addr_spotlight = sim.model.jnt_qposadr[spotlight_joint_id]

    path_index = step_count % len(grid_path)
    pos_spotlight = np.array(grid_path[path_index] + (fixed_z_height,))

    # Ensure the spotlight remains within visible bounds
    pos_spotlight = np.clip(pos_spotlight, [-0.9, -0.65, fixed_z_height], [0.9, 0.65, fixed_z_height])
    sim.data.qpos[qpos_addr_spotlight:qpos_addr_spotlight + 3] = pos_spotlight  # Set the position part of the free joint

    mujoco.mj_forward(sim.model, sim.data)

# Function to run the simulation
def run_simulation():
    sim_time = 5.0
    step_count = 0
    render_interval = 1  # Capture image at every step to ensure all positions are recorded

    while step_count < len(grid_path):  # Stop simulation after covering the 4x4 grid
        if sim.viewer.is_alive:
            handle_key_presses(sim)
            move_spotlight_grid(sim, step_count, grid_path)  # Move the spotlight along the 4x4 grid path
            ctrl = np.zeros(sim.model.nu)
            sim.step(ctrl)

            if step_count % render_interval == 0:
                sim.viewer.render()
                capture_image(sim, sim.viewer, step_count)

            step_count += 1
        else:
            break

    sim.close_sim()

# Run the simulation
if __name__ == "__main__":
    run_simulation()