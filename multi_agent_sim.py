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

# Argument parsing for shape, color, and number of spheros
parser = argparse.ArgumentParser(description="Choose the shape, color, and number of spheros.")
parser.add_argument("shape", choices=["cube", "cylinder"], help="Shape of the object: 'cube' or 'cylinder'")
parser.add_argument("color", choices=["red", "green", "blue"], help="Color of the object: 'red', 'green', or 'blue'")
parser.add_argument("num_spheros", type=int, help="Number of spheros in the environment")
args = parser.parse_args()

# Parent class for agents
class Agent:
    def __init__(self, sim, name):
        self.sim = sim
        self.name = name

    def move(self):
        pass

# Sphero class inheriting from Agent
class Sphero(Agent):
    def __init__(self, sim, name):
        super().__init__(sim, name)
        self.joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, f'{name}_free_joint')
        self.qpos_addr = sim.model.jnt_qposadr[self.joint_id]

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

    def move_ergodically(self, step_count, visit_count, grid_size, contact_influence_steps):
        pos = self.sim.data.qpos[self.qpos_addr:self.qpos_addr + 3].copy()
        x_idx = int((pos[0] + 1) / 2 * (grid_size - 1))
        y_idx = int((pos[1] + 0.75) / 1.5 * (grid_size - 1))
        visit_count[x_idx, y_idx] += 1

        move_directions = [
            (0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005)
        ]
        direction_values = []
        for direction in move_directions:
            new_x_pos = pos[0] + direction[0]
            new_y_pos = pos[1] + direction[1]

            if -1 <= new_x_pos <= 1 and -0.75 <= new_y_pos <= 0.75:
                new_x_idx = int((new_x_pos + 1) / 2 * (grid_size - 1))
                new_y_idx = int((new_y_pos + 0.75) / 1.5 * (grid_size - 1))
                direction_values.append((visit_count[new_x_idx, new_y_idx], direction))
            else:
                direction_values.append((np.inf, direction))

        min_value = min(direction_values, key=lambda x: x[0])[0]
        best_directions = [direction for value, direction in direction_values if value == min_value]
        best_direction = best_directions[np.random.choice(len(best_directions))]
        pos[:2] += best_direction

        pos[0] = np.clip(pos[0], -1, 1)
        pos[1] = np.clip(pos[1], -0.75, 0.75)
        self.sim.data.qpos[self.qpos_addr:self.qpos_addr + 3] = pos

        if step_count % contact_influence_steps == 0:
            visit_count[:] = np.maximum(visit_count, 0)

# Spotlight class inheriting from Agent
class Spotlight(Agent):
    def __init__(self, sim, name):
        super().__init__(sim, name)
        self.joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, f'{name}_free_joint')
        self.qpos_addr = sim.model.jnt_qposadr[self.joint_id]

    def move(self, step_count, grid_path, fixed_z_height=0.4):
        path_index = step_count % len(grid_path)
        pos = np.array(grid_path[path_index] + (fixed_z_height,))
        pos = np.clip(pos, [-0.4, -0.4, fixed_z_height], [0.4, 0.4, fixed_z_height])
        self.sim.data.qpos[self.qpos_addr:self.qpos_addr + 3] = pos

# Generate object-specific XML
def generate_object_xml(object_type, object_color):
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

# Generate spheros XML
def generate_spheros_xml(num_spheros):
    spheros_xml = ""
    for i in range(num_spheros):
        spheros_xml += f"""
        <body name="white_sphero_{i}" pos="{np.random.uniform(-0.3, 0.3)} {np.random.uniform(-0.3, 0.3)} 0.05">
            <geom name="white_sphero_{i}" type="sphere" size="0.05" rgba="1 1 1 1" contype="1" conaffinity="1"/>
            <joint name="white_sphero_{i}_free_joint" type="free" />
            <inertial pos="0 0 0" mass="10.0" diaginertia="1 1 1"/>
        </body>
        """
    return spheros_xml

# Generate the full XML
def generate_xml(object_type, object_color, num_spheros):
    object_xml = generate_object_xml(object_type, object_color)
    spheros_xml = generate_spheros_xml(num_spheros)
    xml = f"""
    <mujoco model="multicube_simulation">
        <compiler angle="degree" coordinate="local"/>
        <option timestep="0.001" gravity="0 0 0"/>

        <worldbody>
            <!-- Ground plane -->
            <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

            <!-- Walls to enclose the space -->
            <body name="wall_x_neg" pos="-0.4 0 0.025">
                <geom name="wall_x_neg_geom" type="box" size="0.01 0.4 0.05" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
            </body>
            <body name="wall_x_pos" pos="0.4 0 0.025">
                <geom name="wall_x_pos_geom" type="box" size="0.01 0.4 0.05" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
            </body>
            <body name="wall_y_neg" pos="0 -0.4 0.025">
                <geom name="wall_y_neg_geom" type="box" size="0.4 0.01 0.05" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
            </body>