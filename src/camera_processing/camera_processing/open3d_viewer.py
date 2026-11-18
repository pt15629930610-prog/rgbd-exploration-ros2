
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import MarkerArray, Marker

import numpy as np
import open3d as o3d
import cv2

import threading
import time


# =========================================
# 1. 相机坐标转换为小车方向坐标
# =========================================

def transform_points(points):

    points = np.asarray(
        points,
        dtype=np.float64
    ).reshape(-1, 3)

    if len(points) == 0:
        return points

    aligned = np.empty_like(points)

    aligned[:, 0] = points[:, 2]
    aligned[:, 1] = -points[:, 0]
    aligned[:, 2] = -points[:, 1]

    return aligned


# =========================================
# 2. 生成三维障碍物编号
# =========================================

def create_label_cloud(text, center, size):

    canvas = np.zeros(
        (64, 160),
        dtype=np.uint8
    )

    cv2.putText(
        canvas,
        str(text),
        (5, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        255,
        2,
        cv2.LINE_AA
    )

    ys, xs = np.where(canvas > 0)

    if len(xs) == 0:
        return None

    scale = 0.003

    # 编号放在包围盒顶部
    x = center[0] + (
        xs - (xs.min() + xs.max()) / 2
    ) * scale

    y = center[1] + (
        (ys.min() + ys.max()) / 2 - ys
    ) * scale

    z = np.full(
        len(xs),
        center[2] + size[2] / 2 + 0.04
    )

    points = np.column_stack((x, y, z))

    label = o3d.geometry.PointCloud()

    label.points = o3d.utility.Vector3dVector(
        points
    )

    label.paint_uniform_color([1.0, 1.0, 1.0])

    return label


# =========================================
# 3. 将 MarkerArray 转换为 Open3D 几何对象
# =========================================

def build_cluster_geometries(marker_array):

    geometries = []

    boxes = {}
    labels = {}

    has_cluster_points = False

    for marker in marker_array.markers:

        if marker.action == Marker.DELETEALL:
            continue

        color = [
            marker.color.r,
            marker.color.g,
            marker.color.b
        ]

        # ---------------------------------
        # 3.1 聚类点云
        # ---------------------------------

        if marker.type == Marker.POINTS:

            if not marker.points:
                continue

            raw_points = np.array([
                [p.x, p.y, p.z]
                for p in marker.points
            ], dtype=np.float64)

            points = transform_points(raw_points)

            cloud = o3d.geometry.PointCloud()

            cloud.points = (
                o3d.utility.Vector3dVector(points)
            )

            cloud.paint_uniform_color(color)

            geometries.append(cloud)

            has_cluster_points = True

        # ---------------------------------
        # 3.2 三维包围盒
        # ---------------------------------

        elif marker.type == Marker.CUBE:

            # 原相机坐标中的中心
            raw_center = np.array([[
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z
            ]])

            center = transform_points(raw_center)[0]

            # 坐标轴发生置换，因此尺寸也要置换
            size = np.array([
                marker.scale.z,
                marker.scale.x,
                marker.scale.y
            ])

            if np.any(size <= 0):
                continue

            minimum = center - size / 2
            maximum = center + size / 2

            aabb = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=minimum,
                max_bound=maximum
            )

            box = (
                o3d.geometry.LineSet
                .create_from_axis_aligned_bounding_box(aabb)
            )

            box.colors = (
                o3d.utility.Vector3dVector(
                    [color] * len(box.lines)
                )
            )

            geometries.append(box)

            boxes[marker.id] = (
                center,
                size,
                color
            )

        # ---------------------------------
        # 3.3 障碍物编号
        # ---------------------------------

        elif marker.type == Marker.TEXT_VIEW_FACING:

            labels[marker.id] = marker.text

    # ---------------------------------
    # 3.4 在包围盒顶部生成编号
    # ---------------------------------

    for marker_id, text in labels.items():

        if marker_id not in boxes:
            continue

        center, size, color = boxes[marker_id]

        label = create_label_cloud(
            text,
            center,
            size
        )

        if label is not None:
            geometries.append(label)

    return geometries, has_cluster_points


# =========================================
# 4. ROS 2 Open3D 节点
# =========================================

class Open3DViewer(Node):

    def __init__(self):

        super().__init__('open3d_viewer')

        self.lock = threading.Lock()

        self.latest_ground = None
        self.latest_obstacles = None
        self.latest_markers = None

        # 地面点云
        self.ground_sub = self.create_subscription(
            PointCloud2,
            '/rgbd_camera/ground',
            self.ground_callback,
            10
        )

        # 障碍物点云
        self.obstacle_sub = self.create_subscription(
            PointCloud2,
            '/rgbd_camera/obstacles',
            self.obstacle_callback,
            10
        )

        # DBSCAN 聚类结果
        self.marker_sub = self.create_subscription(
            MarkerArray,
            '/rgbd_camera/cluster_markers',
            self.marker_callback,
            10
        )

        self.get_logger().info(
            'Open3D Viewer Started!'
        )

    # -------------------------------------
    # PointCloud2 转 NumPy
    # -------------------------------------

    def convert_pointcloud(self, msg):

        cloud = point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        )

        if isinstance(cloud, np.ndarray):

            if cloud.dtype.names:

                points = np.column_stack((
                    cloud['x'],
                    cloud['y'],
                    cloud['z']
                ))

            else:

                points = np.asarray(
                    cloud,
                    dtype=np.float64
                ).reshape(-1, 3)

        else:

            points = np.asarray(
                list(cloud),
                dtype=np.float64
            ).reshape(-1, 3)

        valid = np.isfinite(points).all(axis=1)

        return transform_points(points[valid])

    # -------------------------------------
    # 三个回调函数
    # -------------------------------------

    def ground_callback(self, msg):

        points = self.convert_pointcloud(msg)

        with self.lock:
            self.latest_ground = points

    def obstacle_callback(self, msg):

        points = self.convert_pointcloud(msg)

        with self.lock:
            self.latest_obstacles = points

    def marker_callback(self, msg):

        with self.lock:
            self.latest_markers = msg


# =========================================
# 5. Open3D 主程序
# =========================================

def main(args=None):

    rclpy.init(args=args)

    node = Open3DViewer()

    # ROS 2 后台线程
    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )

    ros_thread.start()

    # 创建 Open3D 窗口
    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name='RGB-D 3D Obstacle Perception',
        width=1280,
        height=800
    )

    ground_pcd = o3d.geometry.PointCloud()
    obstacle_pcd = o3d.geometry.PointCloud()

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3,
        origin=[0, 0, 0]
    )

    vis.add_geometry(ground_pcd)
    vis.add_geometry(obstacle_pcd)
    vis.add_geometry(axis)

    render_option = vis.get_render_option()

    render_option.point_size = 3.0

    render_option.background_color = np.array([
        0.05, 0.05, 0.05
    ])

    view_initialized = False

    dynamic_geometries = []

    cluster_active = False

    try:

        while rclpy.ok():

            if not vis.poll_events():
                break

            with node.lock:

                ground = node.latest_ground
                obstacles = node.latest_obstacles
                markers = node.latest_markers

                node.latest_ground = None
                node.latest_obstacles = None
                node.latest_markers = None

            # ---------------------------------
            # 更新地面
            # ---------------------------------

            if ground is not None:

                ground_pcd.points = (
                    o3d.utility.Vector3dVector(ground)
                )

                ground_pcd.paint_uniform_color([
                    0.65, 0.65, 0.65
                ])

                vis.update_geometry(ground_pcd)

            # ---------------------------------
            # 更新原始障碍物点云
            # 没有聚类结果时作为备用显示
            # ---------------------------------

            if obstacles is not None and not cluster_active:

                obstacle_pcd.points = (
                    o3d.utility.Vector3dVector(obstacles)
                )

                obstacle_pcd.paint_uniform_color([
                    1.0, 0.1, 0.1
                ])

                vis.update_geometry(obstacle_pcd)

            # ---------------------------------
            # 更新 DBSCAN 聚类结果
            # ---------------------------------

            if markers is not None:

                # 删除上一帧的包围盒、编号和聚类点云
                for geometry in dynamic_geometries:

                    vis.remove_geometry(
                        geometry,
                        reset_bounding_box=False
                    )

                dynamic_geometries = []

                # 生成新几何对象
                dynamic_geometries, cluster_active = (
                    build_cluster_geometries(markers)
                )

                # 添加新几何对象
                for geometry in dynamic_geometries:

                    vis.add_geometry(
                        geometry,
                        reset_bounding_box=False
                    )

                # 有聚类点时，隐藏原始红色障碍物点云
                if cluster_active:

                    obstacle_pcd.points = (
                        o3d.utility.Vector3dVector(
                            np.empty((0, 3))
                        )
                    )

                    vis.update_geometry(obstacle_pcd)

            # ---------------------------------
            # 第一次收到数据时自动调整视角
            # ---------------------------------

            if not view_initialized:

                if (
                    ground is not None or
                    obstacles is not None or
                    markers is not None
                ):

                    vis.reset_view_point(True)

                    view_initialized = True

            vis.update_renderer()

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:

        vis.destroy_window()

        node.destroy_node()

        rclpy.shutdown()

        ros_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
