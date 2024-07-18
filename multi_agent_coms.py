import numpy as np
import mujoco
import mujoco_viewer
import argparse
import os
import shutil
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from agents import Sphero, Spotlight, handle_key_presses, plot_contacts_and_edges, calculate_all_object_edges
from OpenX_Cube_simulator_mujoco2 import OpenX_Simulator_Cube  # Correct module name

parser = argparse.ArgumentParser(description="Run a simulation with Spheros and Spotlights.")
parser.add_argument("--num_spheros", type=int, required=True, help="Number of Spheros in the simulation.")
parser.add_argument("--num_spotlights", type=int, required=True, help="Number of Spotlights in the simulation.")
parser.add_argument("--sphero_movement", choices=["random", "efficient", "ergodic"], required=True, help="Movement function for the Spheros: 'random', 'efficient', or 'ergodic'.")
parser.add_argument("--spotlight_movement", choices=["raster", "ergodic"], required=True, help="Movement function for the Spotlights: 'raster' or 'ergodic'.")
args = parser.parse_args()

output_dir = "captured_images"

# Load the provided XML file
with open('multi_agent_run.xml', 'r') as file:
    xml_content = file.read()

# Generate spheros XML
def generate_spheros_xml(num_spheros):
    spheros_xml = ""
    grid_size = int(np.ceil(np.sqrt(num_spheros)))
    x_values = np.linspace(-0.3, 0.3, grid_size)
    y_values = np.linspace(-0.3, 0.3, grid_size)
    positions = [(x, y) for x in x_values for y in y_values]

    for i in range(num_spheros):
        x, y = positions[i]
        spheros_xml += f"""
        <body name="white_sphero_{i}" pos="{x} {y} 0.05">
            <geom name="white_sphero_{i}" type="sphere" size="0.05" rgba="1 1 1 1" contype="1" conaffinity="1"/>
            <joint name="white_sphero_{i}_free_joint" type="free" />
            <inertial pos="0 0 0" mass="10.0" diaginertia="1 1 1"/>
        </body>
        """
    return spheros_xml

# Generate spotlights XML
def generate_spotlights_xml(num_spotlights):
    spotlights_xml = ""
    grid_size = int(np.ceil(np.sqrt(num_spotlights)))
    x_values = np.linspace(-0.4, 0.4, grid_size)
    y_values = np.linspace(-0.4, 0.4, grid_size)
    positions = [(x, y) for x in x_values for y in y_values]

    for i in range(num_spotlights):
        x, y = positions[i]
        if i == 0:
            # First spotlight with name "spotlight"
            spotlights_xml += f"""
            <body name="spotlight_body" pos="{x} {y} 0.5">
                <joint name="spotlight_free_joint" type="free" />
                <light name="spotlight" pos="0 0 0" dir="0 0 -1" diffuse="1 1 1" specular="0.5 0.5 0.5" cutoff="45" exponent="10" directional="false"/>
                <camera name="spotlight_camera" pos="0 0 0.1" mode="trackcom"/>
                <inertial pos="0 0 0" mass="0.01" diaginertia="0.001 0.001 0.001"/>
            </body>
            """
        else:
            # Subsequent spotlights with indexed names
            spotlights_xml += f"""
            <body name="spotlight_body_{i}" pos="{x} {y} 0.5">
                <joint name="spotlight_free_joint_{i}" type="free" />
                <light name="spotlight_{i}" pos="0 0 0" dir="0 0 -1" diffuse="1 1 1" specular="0.5 0.5 0.5" cutoff="45" exponent="10" directional="false"/>
                <camera name="spotlight_camera_{i}" pos="0 0 0.1" mode="trackcom"/>
                <inertial pos="0 0 0" mass="0.01" diaginertia="0.001 0.001 0.001"/>
            </body>
            """
    return spotlights_xml

# Insert Spotlights and Spheros
spheros_xml = generate_spheros_xml(args.num_spheros)
spotlights_xml = generate_spotlights_xml(args.num_spotlights)

# Insert Spotlights and Spheros before the fixed camera
insert_position = xml_content.find('<!-- Insert agents here -->')
xml_content = xml_content[:insert_position] + spheros_xml + spotlights_xml + xml_content[insert_position:]

# Write the modified XML to a file for debugging
with open('generated_env.xml', 'w') as f:
    f.write(xml_content)

# Save the XML file for inspection
with open('saved_generated_env.xml', 'w') as f:
    f.write(xml_content)

class SimulationPublisher(Node):
    def __init__(self):
        super().__init__('simulation_publisher')
        self.publishers_dict = {}

def create_publishers(node, spheros, spotlights):
    all_agents = spheros + spotlights
    
    for agent in all_agents:
        topic_name = f"/{agent.name}_name"
        agent.publisher = node.create_publisher(String, topic_name, 10)
        
        # Subscribe to all other agents' topics
        for other_agent in all_agents:
            if other_agent.name != agent.name:
                other_topic_name = f"/{other_agent.name}_name"
                subscriber = node.create_subscription(String, other_topic_name, agent.subscriber_callback, 10)
                agent.subscribers.append(subscriber)

def run_simulation(sim, spheros, spotlights, sphero_movement, spotlight_movement, node):
    sim_time = 5.0
    step_count = 0
    render_interval = 10  # Render every 10 steps to reduce performance impact
    environment_matrix = np.zeros((100, 100))  # Initialize the environment matrix

    detected_colors = []  # Initialize the detected colors list

    while step_count < 1000:  # Ensure the simulation runs for 1000 steps
        if sim.viewer.is_alive:
            for sphero in spheros:
                sphero.move(step_count, sphero_movement)
                sphero.check_contacts(step_count)
            for spotlight in spotlights:
                spotlight.move(step_count, spotlight_movement)
                if step_count % render_interval == 0:
                    image_path = spotlight.capture_image(sim, sim.viewer, step_count, output_dir, spotlight.name)
                    if image_path:
                        detected_colors = spotlight.evaluate_color_and_store_value(image_path, environment_matrix, detected_colors)

            ctrl = np.zeros(sim.model.nu)
            sim.step(ctrl)

            rclpy.spin_once(node, timeout_sec=0.1)  # Process incoming messages

            print(f"Step Count: {step_count}")

            if step_count % 50 == 0:
                for sphero in spheros:
                    msg = String()
                    msg.data = sphero.name
                    sphero.publisher.publish(msg)

                for spotlight in spotlights:
                    msg = String()
                    msg.data = spotlight.name
                    spotlight.publisher.publish(msg)

            step_count += 1
        else:
            break

    sim.close_sim()

    # Collect contacts from all spheros
    all_contacts = []
    for sphero in spheros:
        all_contacts.extend(sphero.contacts)

    # Convert the contacts list to a structured NumPy array
    if all_contacts:
        contacts_array = np.array(all_contacts, dtype=[('position', float, (2,)), ('time_step', int), ('normal', float, (3,))])
    else:
        contacts_array = np.array([], dtype=[('position', float, (2,)), ('time_step', int), ('normal', float, (3,))])

    # Calculate all object edges
    all_object_edges_array = calculate_all_object_edges(sim)

    # Plot the results including merged and real visit counts for each sphero
    plot_contacts_and_edges(contacts_array, all_object_edges_array, spheros, environment_matrix, detected_colors)

    return contacts_array, all_object_edges_array, spheros  # Ensure it returns the correct values

if __name__ == "__main__":
    rclpy.init()
    node = SimulationPublisher()

    # Initialize the simulator with the generated XML file
    sim = OpenX_Simulator_Cube(render=True, model_xml='generated_env.xml')

    # Debug print to ensure correct initialization
    print("Running simulation with the following parameters:")
    print(f"Number of Spheros: {args.num_spheros}")
    print(f"Number of Spotlights: {args.num_spotlights}")
    print(f"Sphero Movement type: {args.sphero_movement}")
    print(f"Spotlight Movement type: {args.spotlight_movement}")

    # Create spheros
    spheros = [Sphero(sim, f'white_sphero_{i}', i, np.zeros((100, 100)), np.zeros((100, 100)), np.zeros((100, 100)), None) for i in range(args.num_spheros)]
    # After creating all spheros, assign the list to each instance
    for sphero in spheros:
        sphero.spheros = spheros

    # Create spotlights
    spotlights = [Spotlight(sim, 'spotlight', 0, np.zeros((100, 100)), np.zeros((100, 100)), np.zeros((100, 100)))] + \
                 [Spotlight(sim, f'spotlight_body_{i}', i, np.zeros((100, 100)), np.zeros((100, 100)), np.zeros((100, 100))) for i in range(1, args.num_spotlights)]

    # Create publishers and subscribers for agents
    create_publishers(node, spheros, spotlights)

    # Run the simulation and get the results
    contacts_array, all_object_edges_array, spheros = run_simulation(sim, spheros, spotlights, args.sphero_movement, args.spotlight_movement, node)

    # Plot the results
    plot_contacts_and_edges(contacts_array, all_object_edges_array, spheros, environment_matrix)

    rclpy.shutdown()

