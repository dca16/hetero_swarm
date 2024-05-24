import numpy as np
import matplotlib.pyplot as plt
from OpenX_Cube_simulator_mujoco2 import OpenX_Simulator_Cube
from IK_planner import calc_IK_step
from IK_torque_control import IKTorqueControl

# Initialize simulator and torque controller
sim = OpenX_Simulator_Cube(render=True) 
torque_controller = IKTorqueControl(sim)

# Set up variables for running simulation
e = 1  # Set to high value for the first iteration of each phase of motion
eps = 0.01
sim_time = 3.

# Set up fixed torques for gripper
t_open = 1
t_close = -0.2

# Set up desired positions for each phase of motion
p_box_init = sim.get_site_pose("box")[1]
x_ofst = 0.05
z_ofst = -0.03
# Note: I made the change in x 0.0479735 instead of 0.05 because my initial
# simulation was moving the box slightly too much, and this gets me closer
# to an actual change in 0.05m from initial to final position
p_des = [
    [p_box_init[0] - x_ofst, p_box_init[1], 0.12 - z_ofst],
    [p_box_init[0] - x_ofst, p_box_init[1], -z_ofst],
    [p_box_init[0] - x_ofst, p_box_init[1], 0.12 - z_ofst],
    [p_box_init[0] - x_ofst + 0.0479735, p_box_init[1], 0.12 - z_ofst],
    [p_box_init[0] - x_ofst + 0.0479735, p_box_init[1], -z_ofst],
    [p_box_init[0] - x_ofst + 0.0479735, p_box_init[1], -z_ofst]
]
count=0

# Arrays to store plotting data
time_values = []
theta_values = []
theta_dot_values = []
theta_dot_dot_values = []
end_effector_positions = []
box_positions = []

# Run the simulation
while sim.t < sim_time:
    # Set up current and desired end-effector pose, current jacobian
    pose_cur = sim.get_site_pose("end_effector_link")
    jac_cur = sim.get_jacSite("end_effector_link")

    # Evaluate current theta and next theta using calc_IK_step
    theta_cur = sim.get_robot_joint_state()
    theta_next = theta_cur + calc_IK_step(np.eye(3), p_des[count], np.eye(3), pose_cur[1], jac_cur)

    # Evaluate and execute torques
    torque = torque_controller.get_torques(theta_next)
    if 2 <= count <= 4:
        torque[-1] = t_close
    else:
        torque[-1] = t_open
    sim.step(torque)

    # Collect data for plotting
    time_values.append(sim.t)
    theta_values.append(theta_cur)
    theta_dot_values.append(sim.get_robot_jointvel_state())
    theta_dot_dot_values.append(sim.get_robot_jointacc_state())
    end_effector_positions.append(sim.get_robot_ee_state()[1])
    box_positions.append(sim.get_box_state())

    # Update count if at desired location (within allowed error)
    e = np.linalg.norm(sim.get_site_pose("end_effector_link")[1] - p_des[count])
    if e < eps:
        count += 1
        if count > 5:
            count = 5
        e = 1

# Make plots
# Robot joint pose vs time
plt.figure(figsize=(10, 6))
for i in range(len(theta_values[0])):
    plt.plot(time_values, [theta[i] for theta in theta_values], label=f'Joint {i+1}')
plt.xlabel('Time')
plt.ylabel('Joint Pose (theta)')
plt.title('Robot Joint Pose (theta) vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Robot joint velocity vs time
plt.figure(figsize=(10, 6))
for i in range(len(theta_dot_values[0])):
    plt.plot(time_values, [theta_dot[i] for theta_dot in theta_dot_values], label=f'Joint {i+1}')
plt.xlabel('Time')
plt.ylabel('Joint Velocity (theta_dot)')
plt.title('Robot Joint Velocity (theta_dot) vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Robot joint acceleration vs time
plt.figure(figsize=(10, 6))
for i in range(len(theta_dot_dot_values[0])):
    plt.plot(time_values, [theta_dot_dot[i] for theta_dot_dot in theta_dot_dot_values], label=f'Joint {i+1}')
plt.xlabel('Time')
plt.ylabel('Joint Accelerations (theta_dot_dot)')
plt.title('Robot Joint Accelerations (theta_dot_dot) vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Robot end-effector positions vs time
end_effector_positions = np.array(end_effector_positions)
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(time_values, end_effector_positions[:, i], label=f'End-effector Position {["x", "y", "z"][i]}')
plt.xlabel('Time')
plt.ylabel('End-effector Position (X)')
plt.title('Robot End-effector positions X = (x, y, z) vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Box positions vs time
box_positions = np.array(box_positions)
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(time_values, box_positions[:, i], label=f'Box Position {["x", "y", "z"][i]}')
plt.xlabel('Time')
plt.ylabel('Box Position (X)')
plt.title('Box positions X = (x, y, z) vs Time')
plt.legend()
plt.grid(True)
plt.show()