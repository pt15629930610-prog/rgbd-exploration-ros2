# RGB-D Mobile Robot Exploration

## 1. Development Environment

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Fortress
- Python 3
- OpenCV
- NumPy
- Open3D

## 2. Project Structure

- src/camera_processing: ROS 2 perception package
- worlds: Gazebo simulation worlds

## 3. Main Modules

- rgbd_subscriber.py: RGB-D processing and point cloud generation
- obstacle_clustering.py: DBSCAN obstacle clustering
- open3d_viewer.py: 3D point cloud visualization
- occ_mapping.py: Local occupancy grid mapping
- map_accumulator.py: Global occupancy grid mapping

## 4. Build

source /opt/ros/humble/setup.bash

cd ~/robot_ws

colcon build --packages-select camera_processing --symlink-install

source install/setup.bash

### 双房间建图

在 Ubuntu 图形桌面完成上述编译后运行：

```bash
ros2 launch camera_processing two_room_mapping.launch.py
```

场景随包安装。两间房共 8 m × 6 m，小车从西间原点朝 +X 出发。
正前方约 1.43 m 是隔墙，左前方是净宽 1.4 m 的门，东间有一个箱体。
Gazebo 在后台运行，桌面只显示以下三个窗口：

- **Global OCC Map**：基于里程计累积的二维地图（`/exploration/map`）。
- **RGB-D Local Occupancy Grid**：车头朝上的当前局部地图（`/rgbd_camera/occ_grid`）。
- **RGB-D 3D Obstacle Perception**：当前相机视野内的地面（灰色）和障碍物（红色）点云。

双房间使用 5 Hz 深度观测；场景为平地，相机水平安装，距地 0.78 m。
三维窗口按帧刷新，全局 OCC 在车辆移动时逐步积累。可从另一个已加载工作区环境的终端
手动给 `/model/mobile_robot/cmd_vel` 发送速度指令。先观察正前方隔墙，再慢速左转看门口，
转回时检查同一段墙在全局图里是否重合。用 Ctrl+C 结束 launch。

## 5. Main Topics

- /rgbd_camera/depth_image
- /rgbd_camera/ground
- /rgbd_camera/obstacles
- /rgbd_camera/occ_grid
- /exploration/map
- /model/mobile_robot/cmd_vel
- /model/mobile_robot/odometry
