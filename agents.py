import numpy as np
import mujoco
import mujoco_viewer
import glfw
import os
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2

class Agent(Node):
    def __init__(self, sim, name):
        super().__init__(name)
        self.sim = sim
        self.name = name
        self.publisher = self.create_publisher(String, f'/{name}_name', 10)
        self.subscribers = []

    def subscriber_callback(self, msg):
        # Get the name of the agent that is sending a message
        other_name = msg.data.split()[-1]

        # Find distance between self and agent sending message
        self_pos = self.sim.data.xpos[mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, self.name)]
        other_pos = self.sim.data.xpos[mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, other_name)]
        dist = np.linalg.norm(self_pos - other_pos)

        if self.name != other_name and dist < 0.8:
            if "sphero" in self.name and "sphero" in other_name:
                other_sphero = next((s for s in self.spheros if s.name == other_name), None)
                if other_sphero:
                    new_merged_visit_count = (self.merged_visit_count + other_sphero.merged_visit_count) / 2
                    self.merged_visit_count = new_merged_visit_count.copy()
                    self.dummy_visit_count - new_merged_visit_count.copy()

            print(f"{self.name} position: {self_pos}, {other_name} position: {other_pos}, distance between: {dist}")
            self.get_logger().info(f'Received message: {msg.data} on {self.name}')

    def move(self):
        pass

# Sphero class inheriting from Agent
class Sphero(Agent):
    def __init__(self, sim, name, id, dummy_visit_count, real_visit_count, merged_visit_count, spheros, grid_size=100, contact_influence_steps=100):
        super().__init__(sim, name)
        self.id = id
        self.dummy_visit_count = dummy_visit_count
        self.real_visit_count = real_visit_count
        self.merged_visit_count = merged_visit_count
        self.spheros = spheros  # Add this line to store the spheros list
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
        elif movement_type == "ergodic_old":
            self.move_ergodically_old(step_count)

        if step_count % 50 == 0:
            msg = String()
            msg.data = self.name
            self.publisher.publish(msg)

    def move_ergodically(self, step_count):
        sphere_joint_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, f'white_sphero_{self.id}_free_joint')
        qpos_addr_sphere = self.sim.model.jnt_qposadr[sphere_joint_id]
        pos_sphere = self.sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3].copy()

        # Update real visit count
        x_idx = int((pos_sphere[0] + 1) / 2 * (self.grid_size - 1))
        y_idx = int((pos_sphere[1] + 0.75) / 1.5 * (self.grid_size - 1))
        self.real_visit_count[x_idx, y_idx] += 1
        self.merged_visit_count[x_idx, y_idx] += 1
        self.dummy_visit_count[x_idx, y_idx] += 1

        # Calculate movement direction based on merged visit count
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

    def move_ergodically_old(self, step_count):
        sphere_joint_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, f'white_sphero_{self.id}_free_joint')
        qpos_addr_sphere = self.sim.model.jnt_qposadr[sphere_joint_id]
        pos_sphere = self.sim.data.qpos[qpos_addr_sphere:qpos_addr_sphere + 3].copy()

        # Update real visit count
        x_idx = int((pos_sphere[0] + 1) / 2 * (self.grid_size - 1))
        y_idx = int((pos_sphere[1] + 0.75) / 1.5 * (self.grid_size - 1))
        self.dummy_visit_count[x_idx, y_idx] += 1
        self.real_visit_count[x_idx, y_idx] += 1
        self.merged_visit_count[x_idx, y_idx] += 1

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
                pos_contact = contact.pos[:2].copy()  # Get only the x and y coordinates and copy the array
                normal_contact = contact.frame[:3].copy()  # Get the normal direction of the contact
                self.contacts.append((pos_contact, step_count, normal_contact))  # Append to the list

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

    # Function to zero out the rotational velocity of the sphere
    def zero_rotational_velocity(sim, joint_id):
        qvel_addr = sim.model.jnt_dofadr[joint_id]
        sim.data.qvel[qvel_addr + 3:qvel_addr + 6] = 0

# Spotlight class inheriting from Agent
class Spotlight(Agent):
    def __init__(self, sim, name, id, dummy_visit_count, real_visit_count, merged_visit_count):
        super().__init__(sim, name)
        self.id = id
        self.grid_path = self.generate_grid_path((-0.3, 0.3), (-0.3, 0.3), 8)
        self.fixed_z_height = 0.4
        self.dummy_visit_count = dummy_visit_count
        self.real_visit_count = real_visit_count
        self.merged_visit_count = merged_visit_count
        self.grid_size = 100  # Define grid size for the spotlight

    def move(self, step_count, movement_type):
        if movement_type == "raster":
            self.move_raster(step_count)
        elif movement_type == "ergodic":
            self.move_ergodically(step_count)

    def move_raster(self, step_count):
        if self.id == 0:
            spotlight_joint_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'spotlight_free_joint')
        else:
            spotlight_joint_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, f'spotlight_free_joint_{self.id}')

        qpos_addr_spotlight = self.sim.model.jnt_qposadr[spotlight_joint_id]

        path_index = step_count % len(self.grid_path)
        pos_spotlight = np.array(self.grid_path[path_index] + (self.fixed_z_height,))

        # Ensure the spotlight remains within visible bounds
        pos_spotlight = np.clip(pos_spotlight, [-0.4, -0.4, self.fixed_z_height], [0.4, 0.4, self.fixed_z_height])
        self.sim.data.qpos[qpos_addr_spotlight:qpos_addr_spotlight + 3] = pos_spotlight

        mujoco.mj_forward(self.sim.model, self.sim.data)

        if step_count % 50 == 0:
            msg = String()
            msg.data = self.name
            self.publisher.publish(msg)

    def move_ergodically(self, step_count):
        spotlight_joint_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, f'spotlight_free_joint_{self.id}')
        qpos_addr_spotlight = self.sim.model.jnt_qposadr[spotlight_joint_id]
        pos_spotlight = self.sim.data.qpos[qpos_addr_spotlight:qpos_addr_spotlight + 3].copy()

        # Update real visit count
        x_idx = int((pos_spotlight[0] + 0.4) / 0.8 * (self.grid_size - 1))
        y_idx = int((pos_spotlight[1] + 0.4) / 0.8 * (self.grid_size - 1))
        self.real_visit_count[x_idx, y_idx] += 1
        self.merged_visit_count[x_idx, y_idx] += 1
        self.dummy_visit_count[x_idx, y_idx] += 1

        # Calculate movement direction based on merged visit count
        move_directions = [(0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005)]
        direction_values = []
        for direction in move_directions:
            new_x_pos = pos_spotlight[0] + direction[0]
            new_y_pos = pos_spotlight[1] + direction[1]

            if -0.4 <= new_x_pos <= 0.4 and -0.4 <= new_y_pos <= 0.4:
                new_x_idx = int((new_x_pos + 0.4) / 0.8 * (self.grid_size - 1))
                new_y_idx = int((new_y_pos + 0.4) / 0.8 * (self.grid_size - 1))
                direction_values.append((self.dummy_visit_count[new_x_idx, new_y_idx], direction))
            else:
                direction_values.append((np.inf, direction))  # Assign a high value if out of bounds

        # Find the directions with the minimum visit count values
        min_value = min(direction_values, key=lambda x: x[0])[0]
        best_directions = [direction for value, direction in direction_values if value == min_value]

        # Pick a direction randomly from the best directions
        best_direction = best_directions[np.random.choice(len(best_directions))]

        # Move in the best direction
        pos_spotlight[:2] += best_direction

        # Ensure the spotlight remains within bounds
        pos_spotlight[0] = np.clip(pos_spotlight[0], -0.4, 0.4)
        pos_spotlight[1] = np.clip(pos_spotlight[1], -0.4, 0.4)

        self.sim.data.qpos[qpos_addr_spotlight:qpos_addr_spotlight + 3] = pos_spotlight  # Set the position part of the free joint

    # Function to capture images from the spotlight camera
    def capture_image(self, sim, viewer, step_count, output_dir, spotlight_name):
        width, height = viewer.viewport.width, viewer.viewport.height
        rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        depth_buffer = np.zeros((height, width, 1), dtype=np.float32)

        # Handle special case for the 0th spotlight
        if spotlight_name == 'spotlight':
            spotlight_cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, 'spotlight_camera')
        else:
            spotlight_cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, f'spotlight_camera_{spotlight_name.split("_")[-1]}')

        # Debug print to check camera names and IDs
        print(f"Spotlight name: {spotlight_name}")
        print(f"Camera ID for {spotlight_name}_camera: {spotlight_cam_id}")

        if spotlight_cam_id == -1:
            print(f"Error: Camera ID for {spotlight_name}_camera is not valid.")
            return None

        original_camid = viewer.cam.fixedcamid
        original_camtype = viewer.cam.type

        viewer.cam.fixedcamid = spotlight_cam_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        mujoco.mjv_updateScene(sim.model, sim.data, viewer.vopt, None, viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, viewer.scn)
        mujoco.mjr_render(viewer.viewport, viewer.scn, viewer.ctx)
        mujoco.mjr_readPixels(rgb_buffer, depth_buffer, viewer.viewport, viewer.ctx)

        image = Image.fromarray(rgb_buffer)
        spotlight_output_dir = os.path.join(output_dir, spotlight_name)
        os.makedirs(spotlight_output_dir, exist_ok=True)
        image_path = os.path.join(spotlight_output_dir, f"step_{step_count}.png")
        image.save(image_path)

        viewer.cam.fixedcamid = original_camid
        viewer.cam.type = original_camtype

        return image_path
    
    # Function to generate the grid path
    def generate_grid_path(self, x_range, y_range, grid_size):
        x_values = np.linspace(x_range[0], x_range[1], grid_size)
        y_values = np.linspace(y_range[0], y_range[1], grid_size)
        path = []
        for y in y_values:
            for x in x_values:
                path.append((x, y))
        return path
    
    # Function to process color from images and create matrix
    def evaluate_color_and_store_value(self, image_path, environment_matrix, detected_colors):
        # Load image
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges closer to true RGB colors
        color_ranges = {
            'red': ([0, 120, 70], [10, 255, 255]),
            'upper_red': ([170, 120, 70], [180, 255, 255]),
            'green': ([40, 100, 100], [70, 255, 255]),
            'blue': ([100, 150, 0], [140, 255, 255])
        }

        for color, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filter out small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    cx, cy = x + w // 2, y + h // 2  # Get the center of the bounding box

                    # Map the center of the bounding box to the environment matrix coordinates
                    matrix_x = int(cx / image.shape[1] * 100)
                    matrix_y = int(cy / image.shape[0] * 100)

                    detected_colors.append((color, (matrix_x, matrix_y)))

        return detected_colors


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

# Function to plot contacts and edges
def plot_contacts_and_edges(contacts_array, all_object_edges_array, spheros, environment_matrix, detected_colors):
    # Plot contact positions and object edges
    plt.figure(figsize=(12, 10))

    if contacts_array.size > 0:
        plt.subplot(2, 2, 1)
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

    # Plot heatmap of real visit count for each sphero
    n_spheros = len(spheros)
    fig, axs = plt.subplots(2, n_spheros, figsize=(20, 10))

    for idx, sphero in enumerate(spheros):
        ax = axs[0, idx]
        ax.imshow(sphero.real_visit_count.T, cmap='hot', origin='lower', extent=[-1, 1, -0.75, 0.75])
        ax.set_title(f'Real Visit Count for {sphero.name}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True)

    for idx, sphero in enumerate(spheros):
        ax = axs[1, idx]
        ax.imshow(sphero.merged_visit_count.T, cmap='hot', origin='lower', extent=[-1, 1, -0.75, 0.75])
        ax.set_title(f'Merged Visit Count for {sphero.name}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Separate plots for red, green, and blue detected colors with object edges and actual colors
    colors = ['red', 'green', 'blue']
    color_values = [1, 2, 3]
    for color, value in zip(colors, color_values):
        plt.figure(figsize=(8, 6))
        
        # Create a heatmap for the detected color
        heatmap = np.zeros((100, 100))
        for detected_color, (x, y) in detected_colors:
            if detected_color == color:
                heatmap[x, y] += 1
        
        heatmap = heatmap / heatmap.max()  # Normalize to create a probability map

        plt.imshow(heatmap.T, cmap='hot', origin='lower', extent=[0, 100, 0, 100], alpha=0.6)
        
        # Overlay object edges and actual colors
        for edge in all_object_edges_array:
            plt.plot(edge[:, 0] * 50 + 50, edge[:, 1] * 50 + 50, 'k-', linewidth=2)
        
        for x in range(100):
            for y in range(100):
                if environment_matrix[x, y] == value:
                    plt.gca().add_patch(plt.Rectangle((x, y), 1, 1, color=color, alpha=0.3))

        plt.title(f'{color.capitalize()} Detection Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
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

