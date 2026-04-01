# import os

# from ament_index_python.packages import get_package_share_directory
# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.substitutions import LaunchConfiguration
# from launch.actions import TimerAction

# def generate_launch_description():

#     vehicle_name = 'mauv_1'

#     # param path
#     robot_param_path = os.path.join(
#         get_package_share_directory('simple_controller_pkg'),
#         'config'
#     )
#     control_param_file = os.path.join(robot_param_path, 'swarm_control.yaml') 
    
#     return LaunchDescription([

#         TimerAction(period=0.0,
#             actions=[
#                     Node(
#                         package="simple_controller_pkg",
#                         executable="simple_controller_node",
#                         namespace=vehicle_name,
#                         name="simple_controller_node",
#                         prefix=['stdbuf -o L'],
#                         output="screen",
#                         parameters=[control_param_file],
#                         remappings=[
#                             ('simple_controller_node/odom', 'odometry/filtered')  # (from, to)
#                         ]
#                     )
#             ])
        
# ])


import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def make_controller_node(vehicle_name, offset_x, offset_y, offset_z, offset_pitch):
    robot_param_path = os.path.join(
        get_package_share_directory('simple_controller_pkg'),
        'config'
    )
    control_param_file = os.path.join(robot_param_path, 'swarm_control.yaml')

    return Node(
        package="simple_controller_pkg",
        executable="simple_controller_node",
        namespace=vehicle_name,
        name="simple_controller_node",
        prefix=['stdbuf -o L'],
        output="screen",
        parameters=[
            control_param_file,
            {
                "offset_x": offset_x,
                "offset_y": offset_y,
                "offset_z": offset_z,
                "offset_pitch": offset_pitch,
            }
        ],
    )


def generate_launch_description():
    return LaunchDescription([
        TimerAction(
            period=0.0,
            actions=[
                make_controller_node("mauv_1", 0.0,  0.0, 0.0,  0.0),
                make_controller_node("mauv_2", 7.0, -7.0, 0.0,  0.0),
            ]
        )
    ])