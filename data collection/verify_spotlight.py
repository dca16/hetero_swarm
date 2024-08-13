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

# Argument parsing for shape and color
parser = argparse.ArgumentParser(description="Choose the shape and color of the object.")
parser.add_argument("shape", choices=["cube", "cylinder", "none"], help="Shape of the object: 'cube', 'cylinder', or 'none'")
parser.add_argument("color", choices=["red", "green", "blue", "none"], help="Color of the object: 'red', 'green', 'blue', or 'none'")
args = parser.parse_args()

# Generate object-specific XML
def generate_object_xml(object_type, object_color):
    if object_type == "none" and object_color == "none":
        return ""  # Return an empty string if no object is to be added
    
    color_dict = {
        'red': '1 0 0 1',
        'green': '0 1 0 1',
        'blue': '0 0 1 1'
    }
    type_dict = {
        'cube': 'box',
        'cylinder': 'cylinder'
    }
    object_xml = f"""
    <body name="object" pos="0 0 0.05">
        <geom name="{object_type}" type="{type_dict[object_type]}" size="0.05 0.05 0.05" rgba="{color_dict[object_color]}" contype="1" conaffinity="1"/>
        <joint name="object_free_joint" type="free" />
        <inertial pos="0 0 0" mass="100.0" diaginertia="10 10 10"/>
    </body>
    """
    return object_xml

# Generate the full XML
def generate_xml(object_type, object_color):
    object_xml = generate_object_xml(object_type, object_color)
    xml = f"""
    <mujoco model="multicube_simulation">
        <compiler angle="degree" coordinate="local"/>
        <option timestep="0.001" gravity="0 0 0"/>

        <worldbody>
            <!-- Ground plane -->
            <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

            <!-- Walls to enclose the space -->
            <!-- Wall at x = -0.4 -->
            <body name="wall_x_neg" pos="-0.4 0 0.025">
                <geom name="wall_x_neg_geom" type="box" size="0.01 0.4 0.05" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
                <inertial pos="0 0 0" mass="100.0" diaginertia="10 10 10"/>
            </body>

            <!-- Wall at x = 0.4 -->
            <body name="wall_x_pos" pos="0.4 0 0.025">
                <geom name="wall_x_pos_geom" type="box" size="0.01 0.4 0.05" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
                <inertial pos="0 0 0" mass="100.0" diaginertia="10 10 10"/>
            </body>

            <!-- Wall at y = -0.4 -->
            <body name="wall_y_neg" pos="0 -0.4 0.025">
                <geom name="wall_y_neg_geom" type="box" size="0.4 0.01 0.05" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
                <inertial pos="0 0 0" mass="100.0" diaginertia="10 10 10"/>
            </body>

            <!-- Wall at y = 0.4 -->
            <body name="wall_y_pos" pos="0 0.4 0.025">
                <geom name="wall_y_pos_geom" type="box" size="0.4 0.01 0.05" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
                <inertial pos="0 0 0" mass="100.0" diaginertia="10 10 10"/>
            </body>

            {object_xml}

            <!-- Floating spotlight -->
            <body name="spotlight_body" pos="0.35 0.35 0.5">
                <joint name="spotlight_free_joint" type="free" />
                <light name="spotlight" pos="0 0 0" dir="0 0 -1" diffuse="1 1 1" specular="0.5 0.5 0.5" cutoff="45" exponent="10" directional="false"/>
                <camera name="spotlight_camera" pos="0 0 0.1" mode="trackcom"/>
                <inertial pos="0 0 0" mass="0.01" diaginertia="0.001 0.001 0.001"/>
            </body>

            <!-- Camera -->
            <camera name="fixed" pos="0 0 3" zaxis="0 0 1"/>
        </worldbody>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
            markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
        </asset>

        <visual>
            <map znear="0.1" zfar="50"/>
        </visual>
    </mujoco>
    """
    return xml

# Write the generated XML to a file
xml = generate_xml(args.shape, args.color)
with open('generated_env.xml', 'w') as f:
    f.write(xml)

# Initialize simulator with the generated XML
sim = OpenX_Simulator_Cube(render=True, model_xml='generated_env.xml')

# Fixed z-height for the spotlight
fixed_z_height = 0.4

# List to store captured images
captured_images = []

# Directory to save images
output_dir = "/Users/domalberts/Documents/GitHub/hetero_swarm/verification_images/red_cyl"

# Clear the captured images folder
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Variable to keep track of the current camera
current_camera = 'fixed'
camera_switched = False  # Flag to debounce camera switching

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

# Add this function to generate the raster path for a 4x4 grid
def generate_4x4_grid_path(x_range, y_range):
    x_values = np.linspace(x_range[0], x_range[1], 30)
    y_values = np.linspace(y_range[0], y_range[1], 30)
    path = []
    for y in y_values:
        for x in x_values:
            path.append((x, y))
    return path

# Generate the 4x4 grid path
x_range = (-0.2, 0.2)
y_range = (-0.2, 0.2)
grid_path = generate_4x4_grid_path(x_range, y_range)

# Function to move the spotlight along the 4x4 grid path
def move_spotlight_grid(sim, step_count, grid_path):
    spotlight_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'spotlight_free_joint')
    qpos_addr_spotlight = sim.model.jnt_qposadr[spotlight_joint_id]

    path_index = step_count % len(grid_path)
    pos_spotlight = np.array(grid_path[path_index] + (fixed_z_height,))

    # Ensure the spotlight remains within visible bounds
    pos_spotlight = np.clip(pos_spotlight, [-0.4, -0.4, fixed_z_height], [0.4, 0.4, fixed_z_height])
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