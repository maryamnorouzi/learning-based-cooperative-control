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