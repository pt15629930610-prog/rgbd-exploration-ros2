from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory('camera_processing'))
    world = share / 'worlds' / 'two_rooms.sdf'
    simulator = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', str(world)],
        additional_env={'LIBGL_DRI3_DISABLE': '1'},
        output='screen',
    )
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/rgbd_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/rgbd_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            '/model/mobile_robot/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
            '/model/mobile_robot/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
        ],
        output='screen',
    )
    perception = Node(
        package='camera_processing',
        executable='rgbd_subscriber',
        parameters=[{'use_sim_time': True, 'show_images': False, 'camera_height_m': 0.78}],
        output='screen',
    )
    map_nodes = [
        Node(
            package='camera_processing',
            executable=executable,
            parameters=[{'use_sim_time': True}],
            output='screen',
        )
        for executable in ('occ_mapping', 'map_accumulator', 'open3d_viewer')
    ]
    explorer = Node(
        package='camera_processing',
        executable='frontier_explorer',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('explore')),
        output='screen',
    )
    return LaunchDescription([
        DeclareLaunchArgument('explore', default_value='false',
                              description='Enable autonomous frontier exploration'),
        simulator, bridge, perception, *map_nodes, explorer,
    ])
