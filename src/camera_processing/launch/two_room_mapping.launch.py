import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, OpaqueFunction
from launch.conditions import IfCondition, LaunchConfigurationEquals, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def launch_map_accumulator(context, pose_feedback):
    greenhouse = LaunchConfiguration('world').perform(context) == 'four_doors.sdf'
    return [Node(
        package='camera_processing', executable='map_accumulator', output='screen',
        parameters=[{'use_sim_time': True, 'map_size_m': [16., 24.] if greenhouse else [12., 12.]}],
        remappings=[('/model/mobile_robot/odometry', pose_feedback)],
    )]


def generate_launch_description():
    share = Path(get_package_share_directory('camera_processing'))
    world = PathJoinSubstitution([str(share), 'worlds', LaunchConfiguration('world')])
    pose_feedback = '/model/mobile_robot/world_odometry'
    simulator = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', world],
        additional_env={
            'LIBGL_DRI3_DISABLE': '1',
            'IGN_GAZEBO_RESOURCE_PATH': os.pathsep.join(filter(None, (
                str(share / 'worlds'), os.environ.get('IGN_GAZEBO_RESOURCE_PATH')))),
        },
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
            pose_feedback + '@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
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
    rgb_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['/rgbd_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image'],
        condition=LaunchConfigurationEquals('world', 'four_doors.sdf'),
        output='screen',
    )
    map_nodes = [
        Node(
            package='camera_processing',
            executable=executable,
            parameters=[{'use_sim_time': True}],
            output='screen',
        )
        for executable in ('occ_mapping', 'open3d_viewer')
    ]
    map_nodes.append(OpaqueFunction(function=launch_map_accumulator, args=[pose_feedback]))
    explorer = Node(
        package='camera_processing',
        executable='frontier_explorer',
        parameters=[{'use_sim_time': True, 'pose_topic': pose_feedback}],
        condition=IfCondition(LaunchConfiguration('explore')),
        output='screen',
    )
    return LaunchDescription([
        DeclareLaunchArgument('vlm_server', default_value='http://127.0.0.1:18080',
                              description='Qwen server reachable from Ubuntu'),
        DeclareLaunchArgument('world', default_value='two_rooms.sdf',
                              choices=['two_rooms.sdf', 'four_doors.sdf']),
        DeclareLaunchArgument('explore', default_value='false',
                              description='Enable autonomous frontier exploration'),
        simulator, bridge, rgb_bridge, perception, *map_nodes, explorer,
        GroupAction(condition=LaunchConfigurationEquals('world', 'four_doors.sdf'),
                    actions=[Node(package='camera_processing', executable='door_navigator',
                                  name='door_navigator', output='screen',
                                  condition=UnlessCondition(LaunchConfiguration('explore')),
                                  parameters=[{'use_sim_time': True, 'pose_topic': pose_feedback,
                                               'vlm_server': LaunchConfiguration('vlm_server')}])]),
    ])
