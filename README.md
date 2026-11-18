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

## 5. Main Topics

- /rgbd_camera/depth_image
- /rgbd_camera/ground
- /rgbd_camera/obstacles
- /rgbd_camera/occ_grid
- /model/mobile_robot/cmd_vel
- /model/mobile_robot/odometry
