
import rclpy
from rclpy.node import Node

import numpy as np
from sklearn.cluster import DBSCAN

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class ObstacleClustering(Node):

    def __init__(self):
        super().__init__('obstacle_clustering')

        # =====================================
        # 1. DBSCAN 参数
        # =====================================

        self.declare_parameter('eps', 0.12)
        self.declare_parameter('min_samples', 5)
        self.declare_parameter('min_cluster_points', 20)

        self.eps = float(self.get_parameter('eps').value)
        self.min_samples = int(
            self.get_parameter('min_samples').value
        )
        self.min_cluster_points = int(
            self.get_parameter('min_cluster_points').value
        )

        # 聚类颜色
        self.colors = [
            (1.0, 0.1, 0.1),
            (0.1, 1.0, 0.1),
            (0.1, 0.3, 1.0),
            (1.0, 0.8, 0.1),
            (1.0, 0.1, 1.0),
            (0.1, 1.0, 1.0),
            (1.0, 0.5, 0.1),
        ]

        self.frame_count = 0

        # =====================================
        # 2. 订阅障碍物点云
        # =====================================

        self.obstacle_sub = self.create_subscription(
            PointCloud2,
            '/rgbd_camera/obstacles',
            self.obstacle_callback,
            10
        )

        # =====================================
        # 3. 发布聚类可视化结果
        # =====================================

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/rgbd_camera/cluster_markers',
            10
        )

        self.get_logger().info(
            'DBSCAN Obstacle Clustering Started!'
        )

        self.get_logger().info(
            f'eps={self.eps}, '
            f'min_samples={self.min_samples}, '
            f'min_cluster_points={self.min_cluster_points}'
        )

    # =====================================
    # 4. 点云回调函数
    # =====================================

    def obstacle_callback(self, msg):

        # 4.1 读取 PointCloud2 中的 XYZ 数据
        cloud = point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        )

        if isinstance(cloud, np.ndarray) and cloud.dtype.names:
            points = np.column_stack((
                cloud['x'],
                cloud['y'],
                cloud['z']
            ))
        else:
            points = np.asarray(
                list(cloud),
                dtype=np.float32
            )

        if points.size == 0:
            self.clear_markers(msg)
            return

        points = points.reshape(-1, 3)

        # 4.2 删除无效点
        points = points[
            np.isfinite(points).all(axis=1)
        ]

        if len(points) < self.min_samples:
            self.clear_markers(msg)
            return

        # =====================================
        # 5. DBSCAN 聚类
        # =====================================

        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        )

        labels = clustering.fit_predict(points)

        # -1 表示噪声点，不属于任何有效聚类
        unique_labels = sorted(
            set(labels.tolist()) - {-1}
        )

        noise_count = int(np.sum(labels == -1))

        # =====================================
        # 6. 创建 MarkerArray
        # =====================================

        marker_array = MarkerArray()

        # 清除上一帧的显示结果
        clear_marker = Marker()
        clear_marker.header = msg.header
        clear_marker.action = Marker.DELETEALL

        marker_array.markers.append(clear_marker)

        cluster_results = []

        for cluster_id, label in enumerate(unique_labels):

            # 提取当前 Cluster 的点
            cluster_points = points[labels == label]

            # 过滤点数过少的聚类
            if len(cluster_points) < self.min_cluster_points:
                continue

            # =================================
            # 7. 计算障碍物中心和尺寸
            # =================================

            min_xyz = np.min(cluster_points, axis=0)
            max_xyz = np.max(cluster_points, axis=0)

            center = (min_xyz + max_xyz) / 2.0
            size = max_xyz - min_xyz

            cluster_results.append(
                (label, center, size, len(cluster_points))
            )

            color = self.colors[
                cluster_id % len(self.colors)
            ]

            # =================================
            # 8. 发布不同颜色的障碍物点云
            # =================================

            point_marker = Marker()

            point_marker.header = msg.header
            point_marker.ns = 'dbscan_points'
            point_marker.id = cluster_id
            point_marker.type = Marker.POINTS
            point_marker.action = Marker.ADD

            point_marker.pose.orientation.w = 1.0

            point_marker.scale.x = 0.025
            point_marker.scale.y = 0.025

            point_marker.color.r = color[0]
            point_marker.color.g = color[1]
            point_marker.color.b = color[2]
            point_marker.color.a = 1.0

            point_marker.points = [
                Point(
                    x=float(p[0]),
                    y=float(p[1]),
                    z=float(p[2])
                )
                for p in cluster_points
            ]

            marker_array.markers.append(point_marker)

            # =================================
            # 9. 发布障碍物三维包围盒
            # =================================

            box_marker = Marker()

            box_marker.header = msg.header
            box_marker.ns = 'dbscan_boxes'
            box_marker.id = cluster_id
            box_marker.type = Marker.CUBE
            box_marker.action = Marker.ADD

            box_marker.pose.position.x = float(center[0])
            box_marker.pose.position.y = float(center[1])
            box_marker.pose.position.z = float(center[2])

            box_marker.pose.orientation.w = 1.0

            box_marker.scale.x = max(float(size[0]), 0.02)
            box_marker.scale.y = max(float(size[1]), 0.02)
            box_marker.scale.z = max(float(size[2]), 0.02)

            box_marker.color.r = color[0]
            box_marker.color.g = color[1]
            box_marker.color.b = color[2]
            box_marker.color.a = 0.25

            marker_array.markers.append(box_marker)

            # =================================
            # 10. 发布障碍物编号
            # =================================

            text_marker = Marker()

            text_marker.header = msg.header
            text_marker.ns = 'dbscan_labels'
            text_marker.id = cluster_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = float(center[0])
            text_marker.pose.position.y = float(center[1])
            text_marker.pose.position.z = float(max_xyz[2] + 0.10)

            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.10

            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            text_marker.text = f'C{label}'

            marker_array.markers.append(text_marker)

        # =====================================
        # 11. 发布聚类结果
        # =====================================

        self.marker_pub.publish(marker_array)

        self.frame_count += 1

        # 每15帧打印一次，避免终端信息过多
        if self.frame_count % 15 == 0:

            self.get_logger().info(
                f'Clusters={len(cluster_results)}, '
                f'Noise={noise_count}, '
                f'Input Points={len(points)}'
            )

            for label, center, size, count in cluster_results:

                self.get_logger().info(
                    f'Cluster {label} | '
                    f'Center=({center[0]:.2f}, '
                    f'{center[1]:.2f}, '
                    f'{center[2]:.2f}) m | '
                    f'Size=({size[0]:.2f}, '
                    f'{size[1]:.2f}, '
                    f'{size[2]:.2f}) m | '
                    f'Points={count}'
                )

    # =====================================
    # 12. 清除旧的可视化结果
    # =====================================

    def clear_markers(self, msg):

        marker_array = MarkerArray()

        marker = Marker()
        marker.header = msg.header
        marker.action = Marker.DELETEALL

        marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)


def main(args=None):

    rclpy.init(args=args)

    node = ObstacleClustering()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
