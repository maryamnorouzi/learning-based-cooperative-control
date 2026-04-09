from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Path to the shared swarm-controller parameter file.
CONTROL_PARAM_FILE = (
    Path(get_package_share_directory("simple_controller_pkg"))
    / "config"
    / "swarm_control.yaml"
)

SHARED_WEIGHT_DIR = Path.home() / ".ros" / "simple_controller"
SHARED_WEIGHT_FILE = SHARED_WEIGHT_DIR / "shared_wbar.bin"
SHARED_WEIGHT_META_FILE = SHARED_WEIGHT_DIR / "shared_wbar.meta"


def make_controller_node(
    vehicle_name,
    training_offset,
    rotated_offset,
    neighbor_names,
    learning_phase,
    knowledge_source,
    formation_profile,
):
    """Create one namespaced controller node for a swarm vehicle."""

    return Node(
        package="simple_controller_pkg",
        executable="simple_controller_node",
        namespace=vehicle_name,
        name="simple_controller_node",
        prefix=["stdbuf -o L"],
        output="screen",
        parameters=[
            str(CONTROL_PARAM_FILE),
            {
                "neighbor_names": neighbor_names,
                "offset_x": training_offset[0],
                "offset_y": training_offset[1],
                "offset_z": training_offset[2],
                "offset_pitch": training_offset[3],
                "rotated_offset_x": rotated_offset[0],
                "rotated_offset_y": rotated_offset[1],
                "rotated_offset_z": rotated_offset[2],
                "rotated_offset_pitch": rotated_offset[3],
                "learning_phase": learning_phase,
                "knowledge_source": knowledge_source,
                "formation_profile": formation_profile,
            }
        ],
    )


def launch_setup(context, *args, **kwargs):
    """Create launch actions for training mode or saved-weight reuse."""
    use_saved_shared_weights = (
        LaunchConfiguration("use_saved_shared_weights").perform(context).lower()
        == "true"
    )
    formation_profile = LaunchConfiguration("formation_profile").perform(context)
    shared_wbar_save_path = LaunchConfiguration("shared_wbar_save_path").perform(
        context
    )
    shared_wbar_meta_path = LaunchConfiguration("shared_wbar_meta_path").perform(
        context
    )

    learning_phase = "learning"
    knowledge_source = "local_average"
    if use_saved_shared_weights:
        learning_phase = "frozen"
        knowledge_source = "swarm_average"

    swarm_weight_manager = Node(
        package="simple_controller_pkg",
        executable="swarm_weight_manager_node",
        name="swarm_weight_manager_node",
        output="screen",
        parameters=[
            str(CONTROL_PARAM_FILE),
            {
                "shared_wbar_save_path": shared_wbar_save_path,
                "shared_wbar_meta_path": shared_wbar_meta_path,
                "auto_save_shared_wbar": True,
                "auto_load_shared_wbar": True,
            },
        ],
    )

    controller_actions = [
        make_controller_node(
            "mauv_1",
            (0.0, 0.0, 0.0, 0.0),
            (3.0, -3.0, 0.0, 0.0),
            ["mauv_2"],
            learning_phase,
            knowledge_source,
            formation_profile,
        ),
        make_controller_node(
            "mauv_2",
            (3.0, -3.0, 0.0, 0.0),
            (-3.0, 3.0, 0.0, 0.0),
            ["mauv_1", "mauv_3"],
            learning_phase,
            knowledge_source,
            formation_profile,
        ),
        make_controller_node(
            "mauv_3",
            (-3.0, 3.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            ["mauv_2"],
            learning_phase,
            knowledge_source,
            formation_profile,
        ),
    ]

    return [
        swarm_weight_manager,
        TimerAction(
            period=0.0,
            actions=controller_actions,
        )
    ]


def generate_launch_description():
    """Launch the three-vehicle cooperative controller configuration."""
    return LaunchDescription([
        DeclareLaunchArgument(
            "use_saved_shared_weights",
            default_value="false",
            description=(
                "Start controllers in frozen mode using the saved shared swarm "
                "weight."
            ),
        ),
        DeclareLaunchArgument(
            "formation_profile",
            default_value="training",
            description="Initial formation profile: training or rotated.",
        ),
        DeclareLaunchArgument(
            "shared_wbar_save_path",
            default_value=str(SHARED_WEIGHT_FILE),
            description="Path to the saved shared swarm weight binary file.",
        ),
        DeclareLaunchArgument(
            "shared_wbar_meta_path",
            default_value=str(SHARED_WEIGHT_META_FILE),
            description="Path to the saved shared swarm weight metadata file.",
        ),
        OpaqueFunction(function=launch_setup),
    ])
