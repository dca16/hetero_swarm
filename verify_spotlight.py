import numpy as np
import mujoco
import mujoco_viewer
import glfw
import argparse
from OpenX_Cube_simulator_mujoco2 import OpenX_Simulator_Cube
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import argparse

# Argument parsing for shape and color
parser = argparse.ArgumentParser(description="Choose the shape and color of the object.")
parser.add_argument("shape", choices=["cube", "cylinder"], help="Shape of the object: 'cube' or 'cylinder'")
parser.add_argument("color", choices=["red", "green", "blue"], help="Color of the object: 'red', 'green', or 'blue'")
args = parser.parse_args()

# Generate object-specific XML
def generate_object_xml(object_type, object_color):
    color_dict = {
        'red': '1 0 0 1',
        'green': '0 1 0 1',
        'blue': '0 0 1 1'
    }
    object_xml = f"""
    <body name="object" pos="0 0 0.05">
        <geom name="{object_type}" type="{object_type}" size="0.05 0.05 0.05" rgba="{color_dict[object_color]}" contype="1" conaffinity="1"/>
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
sim = OpenX_Simulator_Cube(render=True, xml_path='generated_env.xml')

# Fixed z-height for the spotlight
fixed_z_height = 0.4

# List to store captured images
captured_images = []

# Directory to save images
output_dir = "captured_images"

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

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose the shape and color of the central object.")
    parser.add_argument("--shape", choices=["cube", "cylinder"], required=True, help="Shape of the object: 'cube' or 'cylinder'")
    parser.add_argument("--color", choices=["red", "green", "blue"], required=True, help="Color of the object: 'red', 'green', or 'blue'")
    args = parser.parse_args()

    # Initialize simulator and controller with chosen shape and color
    sim = OpenX_Simulator_Cube(render=True, shape=args.shape, color=args.color)

    run_simulation(movement_function)