import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, OpaqueFunction
from launch.conditions import IfCondition, LaunchConfigurationEquals, LaunchConfigurationNotEquals, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def launch_occupancy_maps(context, pose_feedback):
    world = LaunchConfiguration('world').perform(context)
    sizes = {'two_rooms.sdf': [12., 12.], 'four_doors.sdf': [16., 24.],
             'greenhouse_routes.sdf': [16., 28.]}
    local = {'use_sim_time': True}
    if world == 'greenhouse_routes.sdf':
        local['max_obstacle_z_above_camera_m'] = .27  # 1.05 m from the floor; camera is at .78 m.
    return [Node(
        package='camera_processing', executable='occ_mapping', output='screen', parameters=[local],
    ), Node(
        package='camera_processing', executable='map_accumulator', output='screen',
        parameters=[{'use_sim_time': True, 'map_size_m': sizes[world],
                     'show_trail': world == 'greenhouse_routes.sdf'}],
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
        condition=LaunchConfigurationNotEquals('world', 'two_rooms.sdf'),
        output='screen',
    )
    cloud_view = Node(package='camera_processing', executable='open3d_viewer', output='screen',
                      parameters=[{'use_sim_time': True}])
    occupancy = OpaqueFunction(function=launch_occupancy_maps, args=[pose_feedback])
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
        DeclareLaunchArgument('patrol_lora', default_value='false', choices=['true', 'false'],
                              description='Apply the loaded patrol adapter on instruction parsing requests'),
        DeclareLaunchArgument('patrol_lora_scale', default_value='0.75',
                              description='Patrol language adapter strength between zero and one'),
        DeclareLaunchArgument('world', default_value='two_rooms.sdf',
                              choices=['two_rooms.sdf', 'four_doors.sdf', 'greenhouse_routes.sdf']),
        DeclareLaunchArgument('explore', default_value='false',
                              description='Enable autonomous frontier exploration'),
        simulator, bridge, rgb_bridge, perception, cloud_view, occupancy, explorer,
        GroupAction(condition=LaunchConfigurationEquals('world', 'greenhouse_routes.sdf'),
                    actions=[Node(package='camera_processing', executable='online_navigator',
                                  name='greenhouse_online_navigator', output='screen',
                                  additional_env={'GREENHOUSE_PATROL_LORA': LaunchConfiguration('patrol_lora'),
                                                  'GREENHOUSE_PATROL_LORA_SCALE': LaunchConfiguration('patrol_lora_scale')},
                                  condition=UnlessCondition(LaunchConfiguration('explore')),
                                  parameters=[{'use_sim_time': True, 'pose_topic': pose_feedback,
                                               'vlm_server': LaunchConfiguration('vlm_server')}])]),
        GroupAction(condition=LaunchConfigurationEquals('world', 'four_doors.sdf'),
                    actions=[Node(package='camera_processing', executable='door_navigator',
                                  name='door_navigator', output='screen',
                                  condition=UnlessCondition(LaunchConfiguration('explore')),
                                  parameters=[{'use_sim_time': True, 'pose_topic': pose_feedback,
                                               'vlm_server': LaunchConfiguration('vlm_server')}])]),
    ])
