# Heterogeneous Swarm Simulation

This project simulates a heterogeneous swarm of agents, including Spheros and Spotlights (drones), in a MuJoCo environment. The agents can perform various movements, including random, efficient, and ergodic, and interact with each other within the simulation environment. The Sphero agents log contacts with objects and trajectory histories throughout the simulation. The Spotlight agents use a CNN to predict the location of target objects, which have been set as green cubes.

## Features
- **Spheros**: Move using different algorithms like random, efficient, and ergodic movement.
- **Spotlights**: Capture images and classify objects, with movements including raster, ergodic, and random.
- **Simulation**: Inter-agent communication/trajectory merging and real-time visualization.

## Installation

### Prerequisites
- Python 3.9 or higher
- MuJoCo
- TensorFlow/Keras
- NumPy, Matplotlib, and other standard Python dependencies.
- Must be run in a ROS2 environment

### Steps
1. Clone this repository:
    ```sh
    git clone https://github.com/yourusername/hetero_swarm.git
    ```
2. Navigate to the project directory:
    ```sh
    cd hetero_swarm
    ```
3. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up MuJoCo following the [official guide](https://mujoco.readthedocs.io/en/latest/installation.html).

## Usage

To run a simulation, use the following command:

```sh
python multi_agent_coms.py --num_spheros 3 --num_spotlights 2 --sphero_movement ergodic --spotlight_movement random
