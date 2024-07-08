import numpy as np
import mujoco
import mujoco_viewer
import glfw
import os
import matplotlib.pyplot as plt
from PIL import Image
import shutil

# Parent class for agents
class Agent:
    def __init__(self, sim, name):
        self.sim = sim
        self.name = name

    def move(self):
        pass

# Sphero class inheriting from Agent
class Sphero(Agent):
    def __init__(self, sim, name, id, dummy_visit_count, real_visit_count, grid_size=100, contact_influence_steps=100):
        super().__init__(sim, name)
        self.id = id
        self.dummy_visit_count = dummy_visit_count
        self.real_visit_count = real_visit_count
        self.grid_size = grid_size
        self.contact_influence_steps = contact_influence_steps
        self.contacts = []  # To store contacts

    def move(self, step_count, movement_type):
        if movement_type == "random":
            self.move_randomly(step_count)
        elif movement_type == "efficient":
            self.move_efficiently(step_count)
        elif movement_type == "ergodic":
            self.move_ergodically(step_count)

    def move_ergodically(self, step_count):
        sphere_joint_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, f'white_sphero_{self.id}_free_joint')
        qpos_addr_sphere = self.sim.model.jnt_qposadr[sphere_joint_id]
        pos_sphere = self.sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3].copy()

        # Update visit count
        x_idx = int((pos_sphere[0] + 1) / 2 * (self.grid_size - 1))
        y_idx = int((pos_sphere[1] + 0.75) / 1.5 * (self.grid_size - 1))
        self.dummy_visit_count[x_idx, y_idx] += 1
        self.real_visit_count[x_idx, y_idx] += 1

        # Calculate movement direction based on visit count
        move_directions = [(0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005)]
        direction_values = []
        for direction in move_directions:
            new_x_pos = pos_sphere[0] + direction[0]
            new_y_pos = pos_sphere[1] + direction[1]

            if -1 <= new_x_pos <= 1 and -0.75 <= new_y_pos <= 0.75:
                new_x_idx = int((new_x_pos + 1) / 2 * (self.grid_size - 1))
                new_y_idx = int((new_y_pos + 0.75) / 1.5 * (self.grid_size - 1))
                direction_values.append((self.dummy_visit_count[new_x_idx, new_y_idx], direction))
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

        self.sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3] = pos_sphere  # Set the position part of the free joint

        # Reset attraction to contact points after a certain number of steps
        if step_count % self.contact_influence_steps == 0:
            self.dummy_visit_count[:] = np.maximum(self.dummy_visit_count, 0)  # Ensure visit count does not go negative

    def check_contacts(self, step_count):
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1 = mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if (geom1 == self.name and geom2 != 'floor' and not geom2.startswith('wall') and not geom2.startswith('white_sphero')) or \
               (geom2 == self.name and geom1 != 'floor' and not geom1.startswith('wall') and not geom1.startswith('white_sphero')):
                print("Contact detected between", self.name, "and", geom1 if geom1 != self.name else geom2)
                pos_contact = contact.pos[:2].copy()  # Get only the x and y coordinates and copy the array
                normal_contact = contact.frame[:3].copy()  # Get the normal direction of the contact
                self.contacts.append((pos_contact, step_count, normal_contact))  # Append to the list
                print(f"Contact position: {pos_contact}, time step: {step_count}, normal: {normal_contact}")

                # Calculate grid indices for the contact position
                x_idx = int((pos_contact[0] + 1) / 2 * (self.grid_size - 1))
                y_idx = int((pos_contact[1] + 0.75) / 1.5 * (self.grid_size - 1))

                # Increase the value around the contact point in dummy visit count
                contact_radius = 5
                for dx in range(-contact_radius, contact_radius + 1):
                    for dy in range(-contact_radius, contact_radius + 1):
                        new_x_idx = x_idx + dx
                        new_y_idx = y_idx + dy
                        if 0 <= new_x_idx < self.grid_size and 0 <= new_y_idx < self.grid_size:
                            self.dummy_visit_count[new_x_idx, new_y_idx] -= 5  # Decrease dummy visit count to increase attractiveness

    def move_randomly(self, step_count):
        pos = self.sim.data.qpos[self.qpos_addr:self.qpos_addr + 3].copy()
        pos[:2] += np.random.uniform(-0.01, 0.01, size=2)
        pos[:2] = np.clip(pos[:2], -1, 1)
        pos[2] = np.clip(pos[2], 0.05, 0.5)
        self.sim.data.qpos[self.qpos_addr:self.qpos_addr + 3] = pos

    def move_efficiently(self, step_count):
        pos = self.sim.data.qpos[self.qpos_addr:self.qpos_addr + 3].copy()
        step_size = 0.01
        directions = [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size)]
        direction = directions[np.random.choice(len(directions))]
        pos[:2] += direction
        pos[0] = np.clip(pos[0], -1, 1)
        pos[1] = np.clip(pos[1], -0.75, 0.75)
        self.sim.data.qpos[self.qpos_addr:self.qpos_addr + 3] = pos

# Spotlight class inheriting from Agent
class Spotlight(Agent):
    def __init__(self, sim, name, id):
        super().__init__(sim, name)
        self.id = id
        self.grid_path = self.generate_4x4_grid_path((-0.3, 0.3), (-0.3, 0.3))
        self.fixed_z_height = 0.4

    # Function to generate the 4x4 grid path
    def generate_4x4_grid_path(self, x_range, y_range):
        x_values = np.linspace(x_range[0], x_range[1], 4)
        y_values = np.linspace(y_range[0], y_range[1], 4)
        path = []
        for y in y_values:
            for x in x_values:
                path.append((x, y))
        np.random.shuffle(path)  # Shuffle the path to ensure random traversal
        return path

    def move(self, step_count, grid_path):
        if self.id == 0:
            spotlight_joint_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'spotlight_free_joint')
        else:
            spotlight_joint_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, f'spotlight_free_joint_{self.id}')
        
        qpos_addr_spotlight = self.sim.model.jnt_qposadr[spotlight_joint_id]

        path_index = step_count % len(grid_path)
        pos_spotlight = np.array(grid_path[path_index] + (self.fixed_z_height,))

        # Ensure the spotlight remains within visible bounds
        pos_spotlight = np.clip(pos_spotlight, [-0.4, -0.4, self.fixed_z_height], [0.4, 0.4, self.fixed_z_height])
        self.sim.data.qpos[qpos_addr_spotlight:qpos_addr_spotlight + 3] = pos_spotlight

        mujoco.mj_forward(self.sim.model, self.sim.data)
        
# Function to handle key presses for the spotlight
def handle_key_presses(sim):
    if not hasattr(handle_key_presses, 'camera_switched'):
        handle_key_presses.camera_switched = False
        handle_key_presses.current_camera = 'fixed'

    spotlight_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'spotlight_free_joint')
    qpos_addr_spotlight = sim.model.jnt_qposadr[spotlight_joint_id]
    pos_spotlight = sim.data.qpos[qpos_addr_spotlight:qpos_addr_spotlight + 3].copy()
    if glfw.get_key(sim.viewer.window, glfw.KEY_I) == glfw.PRESS:
        pos_spotlight[1] += 0.01
    if glfw.get_key(sim.viewer.window, glfw.KEY_K) == glfw.PRESS:
        pos_spotlight[1] -= 0.01
    if glfw.get_key(sim.viewer.window, glfw.KEY_J) == glfw.PRESS:
        pos_spotlight[0] -= 0.01
    if glfw.get_key(sim.viewer.window, glfw.KEY_L) == glfw.PRESS:
        pos_spotlight[0] += 0.01

    pos_spotlight = np.clip(pos_spotlight, -0.4, 0.4)
    sim.data.qpos[qpos_addr_spotlight:qpos_addr_spotlight + 3] = pos_spotlight

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
def capture_image(sim, viewer, step_count, output_dir):
    width, height = viewer.viewport.width, viewer.viewport.height
    rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.zeros((height, width, 1), dtype=np.float32)

    spotlight_cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, 'spotlight_camera')

    original_camid = viewer.cam.fixedcamid
    original_camtype = viewer.cam.type

    viewer.cam.fixedcamid = spotlight_cam_id
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    mujoco.mjv_updateScene(sim.model, sim.data, viewer.vopt, None, viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, viewer.scn)
    mujoco.mjr_render(viewer.viewport, viewer.scn, viewer.ctx)
    mujoco.mjr_readPixels(rgb_buffer, depth_buffer, viewer.viewport, viewer.ctx)

    image = Image.fromarray(rgb_buffer)
    image.save(os.path.join(output_dir, f"step_{step_count}.png"))

    viewer.cam.fixedcamid = original_camid
    viewer.cam.type = original_camtype

# Function to generate the 4x4 grid path
def generate_4x4_grid_path(x_range, y_range):
    x_values = np.linspace(x_range[0], x_range[1], 4)
    y_values = np.linspace(y_range[0], y_range[1], 4)
    path = []
    for y in y_values:
        for x in x_values:
            path.append((x, y))
    return path

# Function to zero out the rotational velocity of the sphere
def zero_rotational_velocity(sim, joint_id):
    qvel_addr = sim.model.jnt_dofadr[joint_id]
    sim.data.qvel[qvel_addr + 3:qvel_addr + 6] = 0

# Function to plot contacts and edges
def plot_contacts_and_edges(contacts_array, all_object_edges_array, visit_count):
    if contacts_array.size > 0:
        plt.scatter(contacts_array['position'][:, 0], contacts_array['position'][:, 1], c='red', marker='o', label='Contact Points', s=10)
    else:
        print("No contact points to plot")

    if all_object_edges_array.size > 0:
        plt.scatter(all_object_edges_array[:, 0], all_object_edges_array[:, 1], c='blue', marker='x', label='Object Edges', s=10)
    else:
        print("No object edges to plot")

    plt.title('Contact Positions and Object Edges')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(visit_count.T, cmap='hot', origin='lower', extent=[-1, 1, -0.75, 0.75])
    plt.colorbar(label='Visit Count')
    plt.title('Heatmap of Visit Count')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()

# Function to calculate edges of all objects in the environment
def calculate_all_object_edges(sim):
    object_names = ['cube', 'cube2', 'cube3', 'red_cube1', 'Blue_cube', 'cylinder1', 'cylinder2', 'Red_cylinder', 'green_cylinder1', 'green_cylinder2']
    all_edges = []
    for object_name in object_names:
        edges = calculate_object_edges(sim, object_name)
        all_edges.extend(edges)
    return np.array(all_edges)

def calculate_object_edges(sim, object_name):
    object_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    object_pos = sim.data.xpos[object_body_id][:2]

    # Assuming object is a cube or similar box shape
    half_extent = 0.05  # This should match the size defined in the XML
    edges = [
        object_pos + np.array([half_extent, half_extent]),
        object_pos + np.array([-half_extent, half_extent]),
        object_pos + np.array([half_extent, -half_extent]),
        object_pos + np.array([-half_extent, -half_extent])
    ]
    return edges