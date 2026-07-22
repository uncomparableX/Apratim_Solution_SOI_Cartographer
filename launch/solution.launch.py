from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("Apratim_Solution")

    solution_node = Node(
        package="Apratim_Solution",
        executable="solution_node",
        name="solution_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            os.path.join(pkg_share, "config", "parameters.yaml"),
            os.path.join(pkg_share, "config", "navigation.yaml"),
            os.path.join(pkg_share, "config", "exploration.yaml"),
            os.path.join(pkg_share, "config", "communication.yaml"),
            os.path.join(pkg_share, "config", "mapping.yaml"),
            # use_sim_time must be True: this node always runs against
            # challenge_bridge's Gazebo simulation, never a standalone real
            # clock. Without it, every ROS2 timer here (control loop,
            # replanning, mapping, exploration) fires on wall-clock time
            # while all sensor data and the underlying environment progress
            # on Gazebo's simulation clock. Those two clocks agree only if
            # Gazebo happens to run at ~100% real-time factor; if Gazebo is
            # running slower (headless GPU-LiDAR simulation commonly runs
            # far below real-time, especially under WSL2/virtualized GPU
            # setups), the control loop keeps firing every wall-clock 0.1s
            # while the robot's actual pose and map have barely changed,
            # so it repeatedly re-evaluates and re-fails the same stale
            # plan, and every seconds-scale timeout in the code effectively
            # expires almost immediately in simulation-time terms.
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription([
        solution_node,
    ])