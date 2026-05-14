# simple_controller_pkg

## Introduction
This package provides a learning-based cooperative controller for a group of autonomous underwater vehicles (AUVs) operating together in simulation. The controller is designed for cooperative mission execution under uncertain dynamics and includes a neural-network-based learning component to improve coordination, tracking, and adaptability.

The package also includes supporting launch files, configuration files, CUDA-based RBF utilities, and a yaw extraction node used in the control pipeline.

## Tested Environment
- ROS 2: Jazzy
- Ubuntu: 24.04

## Directory Information
- `config`
  - `swarm_control.yaml`: controller and mission-related parameters.
- `include`
  - `rbf_cuda.hpp`: header for CUDA-based RBF neural network utilities.
- `launch`
  - `swarm_control.launch.py`: launch file for the controller setup.
  <!-- - `swarm_control.launch_twoauvs.py`: launch file for the two-AUV cooperative setup. -->
- `src`
  - `rbf_cuda.cu`: CUDA implementation of the RBF neural-network-related computations.
  - `simple_controller.cpp`: main controller implementation.
  <!-- - `simple_controller_V1.cpp`, `simple_controller_V2.cpp`, `simple_controller_V3.cpp`, `simple_controller_V4.cpp`: development versions of the controller. -->
  <!-- - `simple_controller_firstv.cpp`, `simple_controller_org.cpp`: earlier/reference controller implementations. -->
  <!-- - `simple_controller_twoauvs.cpp`: controller implementation for multi-AUV cooperative experiments. -->
  - `yaw_extractor_node.cpp`: node for extracting desired yaw information from odometry/waypoint topics.
- `test`
  - `sendwaypoints_srv_call.txt`: example service call for sending waypoints.

## Main Features
- Learning-based cooperative control for multiple agents
- Neural-network-based adaptation for uncertain system dynamics
- Support for single-vehicle and multi-vehicle launch setups
- CUDA-based RBF utility implementation
- Yaw extraction utility for waypoint-following/control integration

## Dependencies
This package is intended to be used inside a ROS 2 workspace together with the simulation and MVP-related packages used in the project.

Typical dependencies may include:
- `rclcpp`
- `std_msgs`
- `geometry_msgs`
- `nav_msgs`
- `tf2`
- `tf2_ros`
- `tf2_geometry_msgs`
- `Eigen3`
- CUDA (for `rbf_cuda.cu`)

Please make sure the required ROS 2 packages and system libraries are installed before building.

## Building the Workspace
From the root of your ROS 2 workspace:

```bash
cd ~/ros2_ws
colcon build --packages-select simple_controller_pkg
source install/setup.bash
```

## Learning Phase Control
Each controller node exposes a runtime parameter called `learning_phase`.
The supported values are:

- `learning`
- `steady_recording`
- `frozen`

The commands below switch all three agents together. They assume the
controller nodes are running as:

- `/mauv_1/simple_controller_node`
- `/mauv_2/simple_controller_node`
- `/mauv_3/simple_controller_node`

### Start Steady Recording For All Agents
This keeps learning active and starts collecting steady-phase weight samples
inside each controller.

```bash
for agent in mauv_1 mauv_2 mauv_3; do
  ros2 param set /${agent}/simple_controller_node learning_phase steady_recording
done
```

### Freeze Learning For All Agents
This computes the averaged steady-phase weight `w_bar` for each agent,
uploads it to the GPU, and disables further weight updates.

```bash
for agent in mauv_1 mauv_2 mauv_3; do
  ros2 param set /${agent}/simple_controller_node learning_phase frozen
done
```

### Switch All Agents To Shared Swarm Knowledge
After all three agents have frozen their local `w_bar`, the
`swarm_weight_manager_node` computes:

```text
Wbar_shared = (Wbar_1 + Wbar_2 + Wbar_3) / 3
```

Use the command below to tell all agents to replace their local frozen
knowledge with the shared swarm-average weight:

```bash
for agent in mauv_1 mauv_2 mauv_3; do
  ros2 param set /${agent}/simple_controller_node knowledge_source swarm_average
done
```

### Reuse The Saved Shared Weight On A Fresh Run
To start directly with the saved shared swarm weight and no new learning:

```bash
ros2 launch simple_controller_pkg swarm_control.launch.py use_saved_shared_weights:=true
```

To start directly with the saved shared weight in the rotated formation:

```bash
ros2 launch simple_controller_pkg swarm_control.launch.py use_saved_shared_weights:=true formation_profile:=rotated
```

The saved files are:

- `~/.ros/simple_controller/shared_wbar.bin`
- `~/.ros/simple_controller/shared_wbar.meta`

### Return All Agents To Local Frozen Knowledge
If you want to switch back from the shared swarm average to each agent's own
local frozen `w_bar`:

```bash
for agent in mauv_1 mauv_2 mauv_3; do
  ros2 param set /${agent}/simple_controller_node knowledge_source local_average
done
```

### Rotate The Formation For Transfer Testing
This keeps the mission running, but changes the role/slot assignment so the
agents test the learned knowledge in a new configuration:

- `mauv_1` takes the old `mauv_2` slot
- `mauv_2` takes the old `mauv_3` slot
- `mauv_3` takes the old `mauv_1` slot

```bash
for agent in mauv_1 mauv_2 mauv_3; do
  ros2 param set /${agent}/simple_controller_node formation_profile rotated
done
```

### Return To The Training Formation
To restore the original learned/training configuration:

```bash
for agent in mauv_1 mauv_2 mauv_3; do
  ros2 param set /${agent}/simple_controller_node formation_profile training
done
```

## Mission Selection
Each controller node also exposes a runtime parameter called
`mission_profile`.
The supported values are:

- `figure_eight`
- `quarter_square_hold`

The default mission remains `figure_eight`.
The new `quarter_square_hold` mission moves along a smooth square-corner
path and then holds the final point.

### Start The Quarter-Square Mission At Launch

```bash
ros2 launch simple_controller_pkg swarm_control.launch.py mission_profile:=quarter_square_hold
```

### Switch All Agents To The Quarter-Square Mission At Runtime

```bash
for agent in mauv_1 mauv_2 mauv_3; do
  ros2 param set /${agent}/simple_controller_node mission_profile quarter_square_hold
done
```

### Return All Agents To The Figure-Eight Mission

```bash
for agent in mauv_1 mauv_2 mauv_3; do
  ros2 param set /${agent}/simple_controller_node mission_profile figure_eight
done
```

The quarter-square geometry and timing are configured in
`config/swarm_control.yaml` using:

- `quarter_square_origin_x`
- `quarter_square_origin_y`
- `quarter_square_side_length`
- `quarter_square_turn_radius`
- `quarter_square_leg_time`
- `quarter_square_turn_time`

### Return All Agents To Learning Mode
This re-enables online weight updates for all agents.

```bash
for agent in mauv_1 mauv_2 mauv_3; do
  ros2 param set /${agent}/simple_controller_node learning_phase learning
done
```

### Optional Check
To confirm the current learning phase for all three agents:

```bash
for agent in mauv_1 mauv_2 mauv_3; do
  ros2 param get /${agent}/simple_controller_node learning_phase
done
```

To confirm the active knowledge source for all three agents:

```bash
for agent in mauv_1 mauv_2 mauv_3; do
  ros2 param get /${agent}/simple_controller_node knowledge_source
done
```

To fully confirm the runtime mode, check both:

```bash
for agent in mauv_1 mauv_2 mauv_3; do
  echo -n "${agent} phase: "
  ros2 param get /${agent}/simple_controller_node learning_phase
  echo -n "${agent} source: "
  ros2 param get /${agent}/simple_controller_node knowledge_source
done
```

To confirm the active formation profile for all three agents:

```bash
for agent in mauv_1 mauv_2 mauv_3; do
  ros2 param get /${agent}/simple_controller_node formation_profile
done
```

### Important Note
Do not switch to `frozen` before running `steady_recording` long enough to
collect samples. If no steady-phase samples were recorded, the node will
reject the freeze request and stay in its current mode.
