import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np


class RGBDSubscriber(Node):

    def __init__(self):
        super().__init__('rgbd_subscriber')

        # =====================================
        # 1. 初始化变量
        # =====================================

        # ROS 图像转换工具
        self.bridge = CvBridge()
        self.show_images = self.declare_parameter('show_images', False).value
        # Positive only for level-floor simulations with a known camera height.
        self.camera_height = self.declare_parameter('camera_height_m', 0.0).value

        # 相机内参
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # 相机内参是否获取成功
        self.camera_info_ready = False

        # 深度图计数器
        self.depth_count = 0
        self.cloud_every_n_frames = max(
            1, self.declare_parameter('cloud_every_n_frames', 1).value
        )

        # =====================================
        # 2. 创建 Subscriber
        # =====================================

        # RGB 图像订阅
        if self.show_images:
            self.rgb_sub = self.create_subscription(
                Image,
                '/rgbd_camera/image',
                self.rgb_callback,
                10
            )

        # Depth 深度图订阅
        self.depth_sub = self.create_subscription(
            Image,
            '/rgbd_camera/depth_image',
            self.depth_callback,
            10
        )

        # CameraInfo 相机内参订阅
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/rgbd_camera/camera_info',
            self.camera_info_callback,
            10
        )

        self.get_logger().info(
            'RGB-D Subscriber Started!'
        )
        # 创建 PointCloud2 发布者
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            '/rgbd_camera/points',
            10
        )

        # 点云降采样步长
        self.pointcloud_stride = 4
        # 降采样后的点云
        self.downsampled_pub = self.create_publisher(
            PointCloud2,
            '/rgbd_camera/points_downsampled',
            10
        )

        # 地面点云
        self.ground_pub = self.create_publisher(
            PointCloud2,
            '/rgbd_camera/ground',
            10
        )

        # 障碍物点云
        self.obstacle_pub = self.create_publisher(
            PointCloud2,
            '/rgbd_camera/obstacles',
            10
        )

        # Voxel Grid 参数：体素边长 4 cm
        self.voxel_size = 0.04

        # RANSAC 参数
        self.ransac_iterations = 50
        self.ransac_distance_threshold = 0.03

        # 随机数生成器
        self.rng = np.random.default_rng()

    # =====================================
    # 3. RGB 图像回调函数
    # =====================================

    def rgb_callback(self, msg):

        try:
            # ROS Image 转换为 OpenCV 图像
            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            # 显示 RGB 图像
            cv2.imshow('RGB Image', image)
            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(
                f'RGB conversion error: {e}'
            )
    
    # =====================================
    # 4. CameraInfo 相机内参回调函数
    # =====================================

    def camera_info_callback(self, msg):

        # 从 K 矩阵中提取相机内参
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        # 检查焦距是否有效
        if not np.isfinite(self.fx) or not np.isfinite(self.fy):
            return

        if self.fx <= 0 or self.fy <= 0:
            return

        # 只在第一次获取成功时打印
        if not self.camera_info_ready:

            self.get_logger().info(
                f'Camera Intrinsics:\n'
                f'fx = {self.fx:.2f}\n'
                f'fy = {self.fy:.2f}\n'
                f'cx = {self.cx:.2f}\n'
                f'cy = {self.cy:.2f}'
            )

        self.camera_info_ready = True

    # =====================================
    # 5. Depth 深度图回调函数
    # =====================================

    def depth_callback(self, msg):

        try:
            if not self.camera_info_ready:
                return
            self.depth_count += 1
            publish_cloud = (self.depth_count - 1) % self.cloud_every_n_frames == 0
            if not (publish_cloud or self.show_images):
                return

            # 5.1 ROS Depth Image 转换为 NumPy 数组
            depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )

            depth = np.asarray(
                depth,
                dtype=np.float32
            )

            if publish_cloud:
                self.publish_pointcloud(depth, msg)
            if not self.show_images:
                return

            height, width = depth.shape
            # 6. Depth 图像可视化
            # =====================================

            # 提取有效深度值
            valid = (
                np.isfinite(depth) &
                (depth > 0)
            )

            # 创建显示用灰度图
            depth_display = np.zeros(
                depth.shape,
                dtype=np.uint8
            )

            # 将有效深度限制在 0.1~5.0 m
            clipped = np.clip(
                depth[valid],
                0.1,
                5.0
            )

            # 深度值映射到 0~255
            depth_display[valid] = (
                255 * (5.0 - clipped) / 4.9
            ).astype(np.uint8)

            # 转换为伪彩色图
            depth_color = cv2.applyColorMap(
                depth_display,
                cv2.COLORMAP_JET
            )

            # 无效深度显示为黑色
            depth_color[~valid] = 0

            # 在图像上标记采样位置
            if self.camera_info_ready:

                u = int(round(self.cx))
                v = int(round(self.cy))

                if 0 <= u < width and 0 <= v < height:

                    cv2.drawMarker(
                        depth_color,
                        (u, v),
                        (255, 255, 255),
                        cv2.MARKER_CROSS,
                        12,
                        2
                    )

            # 显示深度图
            cv2.imshow(
                'Depth Image',
                depth_color
            )

            cv2.waitKey(1)

        except CvBridgeError as e:

            self.get_logger().error(
                f'Depth conversion error: {e}'
            )
            
    def publish_pointcloud(self, depth, msg):

        if not self.camera_info_ready:
            return

        # 1. 获取深度图尺寸
        height, width = depth.shape

        # 2. 按照 stride 进行降采样
        stride = self.pointcloud_stride

        # 生成像素坐标矩阵
        v, u = np.mgrid[
            0:height:stride,
            0:width:stride
        ]

        # 获取对应位置的深度值
        z = depth[::stride, ::stride]

        # 3. 筛选有效深度
        valid = (
            np.isfinite(z) &
            (z > 0.1) &
            (z < 5.0)
        )

        if not np.any(valid):
            return

        # 4. 利用针孔相机模型计算三维坐标
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        # 5. 提取有效点
        points = np.column_stack((
            x[valid],
            y[valid],
            z[valid]
        )).astype(np.float32)

        header = msg.header
        if self.pointcloud_pub.get_subscription_count():
            self.pointcloud_pub.publish(point_cloud2.create_cloud_xyz32(header, points))

        downsampled_points = self.voxel_downsample(points)
        if self.downsampled_pub.get_subscription_count():
            self.downsampled_pub.publish(
                point_cloud2.create_cloud_xyz32(header, downsampled_points)
            )

        if self.camera_height > 0.0:
            # Optical +Y points down. Walls can fill the entire image at the start.
            ground_mask = np.abs(downsampled_points[:, 1] - self.camera_height) < 0.05
            ground_points = downsampled_points[ground_mask]
            obstacle_points = downsampled_points[~ground_mask]
        else:
            ground_points, obstacle_points = self.ransac_ground_segmentation(downsampled_points)
            if ground_points is None:
                return

        # 5. 发布地面点云
        ground_cloud = point_cloud2.create_cloud_xyz32(
            header,
            ground_points
        )

        self.ground_pub.publish(ground_cloud)


        # 6. 发布障碍物点云
        obstacle_cloud = point_cloud2.create_cloud_xyz32(
            header,
            obstacle_points
        )

        self.obstacle_pub.publish(obstacle_cloud)

    def voxel_downsample(self, points):

        if len(points) == 0:
            return points

        # 1. 计算每个点所属的体素编号
        voxel_indices = np.floor(
            points / self.voxel_size
        ).astype(np.int32)

        # 2. 找出不同体素及每个点对应的体素编号
        _, inverse = np.unique(
            voxel_indices,
            axis=0,
            return_inverse=True
        )

        # 3. 统计每个体素内的点数
        counts = np.bincount(inverse)

        # 4. 累加每个体素内的 XYZ 坐标
        voxel_sum = np.zeros(
            (len(counts), 3),
            dtype=np.float32
        )

        np.add.at(
            voxel_sum,
            inverse,
            points
        )

        # 5. 计算每个体素的中心点
        downsampled = voxel_sum / counts[:, None]

        return downsampled.astype(np.float32)

    def ransac_ground_segmentation(self, points):

        if len(points) < 30:
            return None, None

        best_mask = None
        best_count = 0

    # 1. 随机拟合平面
        for _ in range(self.ransac_iterations):

            # 随机选取三个点
            indices = self.rng.choice(
                len(points),
                3,
                replace=False
            )

            p1, p2, p3 = points[indices]

            # 计算平面法向量
            normal = np.cross(
                p2 - p1,
                p3 - p1
            )

            norm = np.linalg.norm(normal)

            if norm < 1e-6:
                continue

            normal = normal / norm

            d = -np.dot(normal, p1)

            # 2. 计算所有点到平面的距离
            distances = np.abs(
                points @ normal + d
            )

            # 3. 判断哪些点属于平面
            mask = (
                distances <
                self.ransac_distance_threshold
            )

            count = np.count_nonzero(mask)

            # 4. 保留内点最多的平面
            if count > best_count:

                best_count = count
                best_mask = mask

        # 5. 判断是否找到有效平面
        min_inliers = max(
            30,
            int(len(points) * 0.1)
        )

        if best_mask is None or best_count < min_inliers:
            return None, None

        # 6. 使用 SVD 对平面进行优化
        ground_candidate = points[best_mask]

        centroid = np.mean(
            ground_candidate,
            axis=0
        )

        _, _, vh = np.linalg.svd(
            ground_candidate - centroid,
            full_matrices=False
        )

        normal = vh[-1]

        d = -np.dot(normal, centroid)

        # 7. 重新计算地面点
        distances = np.abs(
            points @ normal + d
        )

        ground_mask = (
            distances <
            self.ransac_distance_threshold
        )

        # 8. 分离地面和障碍物
        ground_points = points[ground_mask]

        obstacle_points = points[~ground_mask]

        return ground_points, obstacle_points


    # =====================================
    # 7. 节点关闭
    # =====================================


def main(args=None):

    # 初始化 ROS 2
    rclpy.init(args=args)

    # 创建 RGB-D 订阅节点
    node = RGBDSubscriber()

    try:
        # 保持节点运行，持续接收数据
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        # 销毁节点
        node.destroy_node()

        # 关闭 ROS 2
        rclpy.shutdown()

        # 关闭 OpenCV 窗口
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
