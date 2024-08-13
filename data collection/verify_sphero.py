import numpy as np
import mujoco
import mujoco_viewer
import glfw
import os
import shutil
from PIL import Image
from agents import Sphero
from OpenX_Cube_simulator_mujoco2 import OpenX_Simulator_Cube

# Initialize simulator and controller
sim = OpenX_Simulator_Cube(render=True)

# Set grid size
gridsize_x = 10
gridsize_y = 8

# Initiate list to store contact locations
contacts = []

# Directory to save images
output_dir = "sphero"

# Clear the captured images folder
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Initialize visit counts
dummy_visit_count = np.zeros((gridsize_x, gridsize_y))
real_visit_count = np.zeros((gridsize_x, gridsize_y))
merged_visit_count = np.zeros((gridsize_x, gridsize_y))
spheros = []

# Create the Sphero agent
sphero = Sphero(sim, 'white_sphero_0', 0, dummy_visit_count, real_visit_count, merged_visit_count, spheros)

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

# Generate the spanning tree path
x_range = (-1.0, 1.0)
y_range = (-0.75, 0.75)
spanning_tree_path = sphero.generate_spanning_tree_path(x_range, y_range, gridsize_x, gridsize_y)

# Function to move the spotlight along a raster path
def move_spotlight_raster(sim, step_count):
    spotlight_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'spotlight_free_joint')
    qpos_addr_spotlight = sim.model.jnt_qposadr[spotlight_joint_id]

    grid_size = 10
    x_values = np.linspace(-0.9, 0.9, grid_size)
    y_values = np.linspace(-0.65, 0.65, grid_size)

    path = [(x, y) for y in y_values for x in x_values]
    path_index = step_count % len(path)
    pos_spotlight = np.array(path[path_index] + (0.4,))

    # Ensure the spotlight remains within visible bounds
    pos_spotlight = np.clip(pos_spotlight, [-0.9, -0.65, 0.4], [0.9, 0.65, 0.4])
    sim.data.qpos[qpos_addr_spotlight:qpos_addr_spotlight + 3] = pos_spotlight

    mujoco.mj_forward(sim.model, sim.data)

# Function to run the simulation
def run_simulation():
    step_count = 0
    render_interval = 1  # Capture image at every step to ensure all positions are recorded

    while step_count < len(spanning_tree_path):  # Stop simulation after covering the grid
        if sim.viewer.is_alive:
            sphero.move_spanning_tree(step_count, spanning_tree_path)  # Move the sphero along the spanning tree path
            move_spotlight_raster(sim, step_count)  # Move the spotlight along the raster path
            ctrl = np.zeros(sim.model.nu)
            sim.step(ctrl)

            if step_count % render_interval == 0:
                sim.viewer.render()
                capture_image(sim, sim.viewer, step_count)
                contacts.extend(sphero.check_contacts())

            step_count += 1
        else:
            break

    sim.close_sim()

    return contacts

# Run the simulation
if __name__ == "__main__":
    contacts = run_simulation()
    print("Contacts:", contacts)
