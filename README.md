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
启动后可以在 Gazebo 中观察小车的实际位置，并与三个感知窗口对照。

- **Gazebo** 显示双房间、门口、箱体和小车的运动。
- **Global OCC Map**：基于里程计累积的二维地图（`/exploration/map`）。
- **RGB-D Local Occupancy Grid**：车头朝上的当前局部地图（`/rgbd_camera/occ_grid`）。
- **RGB-D 3D Obstacle Perception**：当前相机视野内的地面（灰色）和障碍物（红色）点云。

双房间使用 5 Hz 深度观测；场景为平地，相机水平安装，距地 0.78 m。
三维窗口按帧刷新，全局 OCC 在车辆移动时逐步积累。可从另一个已加载工作区环境的终端
手动给 `/model/mobile_robot/cmd_vel` 发送速度指令。先观察正前方隔墙，再慢速左转看门口，
转回时检查同一段墙在全局图里是否重合。用 Ctrl+C 结束 launch。

### 自主探索双房间

在 Ubuntu 上编译并加载工作区后，先用 Ctrl+C 结束上一轮仿真，再启动自主模式。
同一轮仿真只启动一个 launch，自主运行期间由探索节点独占速度指令。

```bash
ros2 launch camera_processing two_room_mapping.launch.py explore:=true
```

小车等待全局地图、局部 OCC 和里程计，随后原地转一圈。按默认角速度，
首轮扫描约需 17 秒仿真时间。算法从观测到的自由格中寻找与未知区相邻的
连通边界组，在已知自由空间内用 A* 选路；障碍物的规划膨胀半径为 0.55 m，
对角移动需要两侧均可通行。规划只使用在线地图。

在 **Global OCC Map** 中，蓝色箭头表示小车，绿色折线表示最近一次规划的路径，
橙色圆点表示当前目标。小车以最高 0.16 m/s 跟踪路径，到达观察位置后再扫描。
路径失效时重新规划，局部图检测到前方受阻或跟踪停滞时尝试其他目标。
地图或里程计超过 1.2 秒仿真时间未更新时，控制器发零速度并等待恢复。

终端显示 `Complete` 表示补扫后仍未找到符合条件的可达边界；
`Blocked` 表示净空不足、原地扫描卡住或剩余目标重试失败。两种状态都会停车。
Ctrl+C 结束整轮实验。省略 `explore:=true` 可进入手动建图模式。

## 5. Main Topics

- /rgbd_camera/depth_image
- /rgbd_camera/ground
- /rgbd_camera/obstacles
- /rgbd_camera/occ_grid
- /exploration/map
- /exploration/path
- /model/mobile_robot/cmd_vel
- /model/mobile_robot/odometry
