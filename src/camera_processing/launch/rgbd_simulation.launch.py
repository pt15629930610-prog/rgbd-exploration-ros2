
import os

from launch import LaunchDescription

from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction
)

from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    # =====================================
    # 1. Gazebo 世界文件
    # =====================================

    default_world = os.path.expanduser(
        '~/gazebo_worlds/rgbd_minimal.sdf'
    )

    world = LaunchConfiguration('world')

    # =====================================
    # 2. 启动 Gazebo
    # =====================================

    gazebo = ExecuteProcess(
        cmd=[
            'ign',
            'gazebo',
            '-r',
            world
        ],
        additional_env={
            'LIBGL_DRI3_DISABLE': '1'
        },
        output='screen'
    )

    # =====================================
    # 3. Gazebo 与 ROS 2 Bridge
    # =====================================

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',

        arguments=[

            # Gazebo -> ROS 2：仿真时钟
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',

            # RGB 图像
            '/rgbd_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',

            # Depth 深度图
            '/rgbd_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',

            # CameraInfo
            '/rgbd_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',

            # ROS 2 -> Gazebo：小车速度控制
            '/model/mobile_robot/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',

            # Gazebo -> ROS 2：小车里程计
            '/model/mobile_robot/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'

        ],

        output='screen'
    )

    # =====================================
    # 4. RGB-D 数据处理节点
    # =====================================

    rgbd_node = Node(
        package='camera_processing',
        executable='rgbd_subscriber',

        parameters=[
            {'use_sim_time': True}
        ],

        output='screen'
    )

    # =====================================
    # 5. DBSCAN 障碍物聚类节点
    # =====================================

    clustering_node = Node(
        package='camera_processing',
        executable='obstacle_clustering',

        parameters=[
            {'use_sim_time': True}
        ],

        output='screen'
    )

    # =====================================
    # 6. Open3D 三维可视化节点
    # =====================================

    viewer_node = Node(
        package='camera_processing',
        executable='open3d_viewer',

        parameters=[
            {'use_sim_time': True}
        ],

        output='screen'
    )

    # =====================================
    # 7. OCC 占据栅格节点
    # =====================================

    occ_node = Node(
        package='camera_processing',
        executable='occ_mapping',

        parameters=[
            {'use_sim_time': True}
        ],

        output='screen'
    )

    # =====================================
    # 8. 统一启动
    # =====================================

    return LaunchDescription([

        # Gazebo 世界文件参数
        DeclareLaunchArgument(
            'world',
            default_value=default_world,
            description='Gazebo world file'
        ),

        # 启动 Gazebo
        gazebo,

        # 启动 ROS-Gazebo Bridge
        TimerAction(
            period=1.0,
            actions=[bridge]
        ),

        # 启动 RGB-D 感知
        TimerAction(
            period=2.0,
            actions=[rgbd_node]
        ),

        # 启动 DBSCAN
        TimerAction(
            period=3.0,
            actions=[clustering_node]
        ),

        # 启动 Open3D
        TimerAction(
            period=4.0,
            actions=[viewer_node]
        ),

        # 启动 OCC
        TimerAction(
            period=5.0,
            actions=[occ_node]
        ),

    ])
