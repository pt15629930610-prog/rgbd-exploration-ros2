
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import OccupancyGrid

import numpy as np
import cv2


# =================================================
# 1. OCC 地图参数
# =================================================

MAP_X_MIN = 0.0
MAP_X_MAX = 5.0

MAP_Y_MIN = -3.0
MAP_Y_MAX = 3.0

RESOLUTION = 0.05

GRID_ROWS = int(
    round((MAP_X_MAX - MAP_X_MIN) / RESOLUTION)
)

GRID_COLS = int(
    round((MAP_Y_MAX - MAP_Y_MIN) / RESOLUTION)
)

# 窗口刷新频率：10 Hz
UPDATE_PERIOD = 0.1

# 自定义局部坐标系
FRAME_ID = 'camera_aligned'


# =================================================
# 2. 相机坐标转换为小车方向坐标
# =================================================

def transform_points(points):

    points = np.asarray(
        points,
        dtype=np.float64
    ).reshape(-1, 3)

    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    aligned = np.empty_like(points)

    # X：小车前方
    aligned[:, 0] = points[:, 2]

    # Y：小车左侧
    aligned[:, 1] = -points[:, 0]

    # Z：小车上方
    aligned[:, 2] = -points[:, 1]

    return aligned


# =================================================
# 3. PointCloud2 转 NumPy
# =================================================

def convert_pointcloud(msg):

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

    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    valid = np.isfinite(points).all(axis=1)

    return transform_points(points[valid])


# =================================================
# 4. 三维点转换为二维栅格索引
# =================================================

def points_to_grid(points):

    if len(points) == 0:

        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32)
        )

    x = points[:, 0]
    y = points[:, 1]

    valid = (
        (x >= MAP_X_MIN) &
        (x < MAP_X_MAX) &
        (y >= MAP_Y_MIN) &
        (y < MAP_Y_MAX)
    )

    x = x[valid]
    y = y[valid]

    # 前后方向索引
    ix = np.floor(
        (x - MAP_X_MIN) / RESOLUTION
    ).astype(np.int32)

    # 左右方向索引
    iy = np.floor(
        (y - MAP_Y_MIN) / RESOLUTION
    ).astype(np.int32)

    # OpenCV 图像坐标：前方在上
    rows = GRID_ROWS - 1 - ix

    # 小车左侧在图像左边
    cols = GRID_COLS - 1 - iy

    return rows, cols


# =================================================
# 5. 射线追踪：使用 OpenCV 绘制自由空间
# =================================================

def mark_free_rays(grid, rows, cols):

    if len(rows) == 0:
        return

    # 相机原点在栅格中的位置
    origin_rows, origin_cols = points_to_grid(
        np.array([[0.0, 0.0, 0.0]])
    )

    origin_row = int(origin_rows[0])
    origin_col = int(origin_cols[0])

    # 使用独立的掩膜记录自由空间
    free_mask = np.zeros(
        (GRID_ROWS, GRID_COLS),
        dtype=np.uint8
    )

    # 相同终点只绘制一次
    endpoints = np.unique(
        np.column_stack((rows, cols)),
        axis=0
    )

    for row, col in endpoints:

        cv2.line(
            free_mask,
            (origin_col, origin_row),
            (int(col), int(row)),
            255,
            1,
            cv2.LINE_8
        )

    grid[free_mask > 0] = 0


# =================================================
# 6. OCC ROS 2 节点
# =================================================

class OCCMapping(Node):

    def __init__(self):

        super().__init__('occ_mapping')

        # -1：Unknown
        #  0：Free
        # 100：Occupied

        self.grid = np.full(
            (GRID_ROWS, GRID_COLS),
            -1,
            dtype=np.int8
        )

        # 缓存最近一次有效点云
        self.latest_ground = None
        self.latest_obstacles = None

        # 是否需要重新计算地图
        self.map_dirty = False

        # 是否已经生成过地图
        self.map_ready = False

        self.frame_count = 0

        # -----------------------------------------
        # 订阅地面点云
        # -----------------------------------------

        self.ground_sub = self.create_subscription(
            PointCloud2,
            '/rgbd_camera/ground',
            self.ground_callback,
            10
        )

        # -----------------------------------------
        # 订阅障碍物点云
        # -----------------------------------------

        self.obstacle_sub = self.create_subscription(
            PointCloud2,
            '/rgbd_camera/obstacles',
            self.obstacle_callback,
            10
        )

        # -----------------------------------------
        # 发布 OCC 地图
        # -----------------------------------------

        self.occ_pub = self.create_publisher(
            OccupancyGrid,
            '/rgbd_camera/occ_grid',
            10
        )

        # -----------------------------------------
        # 定时器：持续刷新显示
        # -----------------------------------------

        self.timer = self.create_timer(
            UPDATE_PERIOD,
            self.update_map
        )

        self.get_logger().info(
            f'OCC Mapping Started! '
            f'Grid={GRID_ROWS}x{GRID_COLS}, '
            f'Resolution={RESOLUTION}m, '
            f'Update={1.0 / UPDATE_PERIOD:.1f}Hz'
        )

    # =================================================
    # 7. 地面点云回调
    # =================================================

    def ground_callback(self, msg):

        self.latest_ground = convert_pointcloud(msg)

        self.map_dirty = True

    # =================================================
    # 8. 障碍物点云回调
    # =================================================

    def obstacle_callback(self, msg):

        self.latest_obstacles = convert_pointcloud(msg)

        self.map_dirty = True

    # =================================================
    # 9. 重建当前局部 OCC
    # =================================================

    def rebuild_map(self):

        # 重新初始化当前局部地图
        self.grid.fill(-1)

        # -----------------------------------------
        # 9.1 地面射线：自由空间
        # -----------------------------------------

        if self.latest_ground is not None:

            ground_rows, ground_cols = points_to_grid(
                self.latest_ground
            )

            mark_free_rays(
                self.grid,
                ground_rows,
                ground_cols
            )

        # -----------------------------------------
        # 9.2 障碍物射线：自由空间
        # -----------------------------------------

        if self.latest_obstacles is not None:

            obstacle_rows, obstacle_cols = points_to_grid(
                self.latest_obstacles
            )

            mark_free_rays(
                self.grid,
                obstacle_rows,
                obstacle_cols
            )

            # -------------------------------------
            # 9.3 障碍物终点：占据空间
            # -------------------------------------

            if len(obstacle_rows) > 0:

                endpoints = np.unique(
                    np.column_stack((
                        obstacle_rows,
                        obstacle_cols
                    )),
                    axis=0
                )

                self.grid[
                    endpoints[:, 0],
                    endpoints[:, 1]
                ] = 100

        self.map_ready = True

        self.map_dirty = False

    # =================================================
    # 10. 发布 ROS 2 OccupancyGrid
    # =================================================

    def publish_occupancy_grid(self):

        msg = OccupancyGrid()

        now = self.get_clock().now().to_msg()

        msg.header.stamp = now
        msg.header.frame_id = FRAME_ID

        msg.info.map_load_time = now

        msg.info.resolution = RESOLUTION

        # ROS 地图：X 为 width，Y 为 height
        msg.info.width = GRID_ROWS
        msg.info.height = GRID_COLS

        msg.info.origin.position.x = MAP_X_MIN
        msg.info.origin.position.y = MAP_Y_MIN
        msg.info.origin.position.z = 0.0

        msg.info.origin.orientation.w = 1.0

        # 转换 OpenCV 行列顺序为 ROS 地图存储顺序
        ros_grid = np.flip(
            self.grid,
            axis=(0, 1)
        ).T

        msg.data = (
            ros_grid.astype(np.int8)
            .flatten()
            .tolist()
        )

        self.occ_pub.publish(msg)

    # =================================================
    # 11. OpenCV 可视化
    # =================================================

    def show_occupancy_grid(self):

        image = np.zeros(
            (GRID_ROWS, GRID_COLS, 3),
            dtype=np.uint8
        )

        # Unknown：深灰色
        image[self.grid == -1] = (
            55, 55, 55
        )

        # Free：浅灰色
        image[self.grid == 0] = (
            235, 235, 235
        )

        # Occupied：红色
        image[self.grid == 100] = (
            0, 0, 255
        )

        # -----------------------------------------
        # 11.1 绘制辅助网格
        # 每10格为0.5m
        # -----------------------------------------

        for row in range(0, GRID_ROWS, 10):

            cv2.line(
                image,
                (0, row),
                (GRID_COLS - 1, row),
                (100, 100, 100),
                1
            )

        for col in range(0, GRID_COLS, 10):

            cv2.line(
                image,
                (col, 0),
                (col, GRID_ROWS - 1),
                (100, 100, 100),
                1
            )

        # -----------------------------------------
        # 11.2 绘制相机原点
        # -----------------------------------------

        origin_rows, origin_cols = points_to_grid(
            np.array([[0.0, 0.0, 0.0]])
        )

        origin_row = int(origin_rows[0])
        origin_col = int(origin_cols[0])

        cv2.circle(
            image,
            (origin_col, origin_row),
            3,
            (255, 255, 0),
            -1
        )

        # X：前方
        cv2.arrowedLine(
            image,
            (origin_col, origin_row),
            (origin_col, origin_row - 20),
            (255, 0, 0),
            1,
            tipLength=0.3
        )

        # Y：左侧
        cv2.arrowedLine(
            image,
            (origin_col, origin_row),
            (origin_col - 20, origin_row),
            (0, 255, 0),
            1,
            tipLength=0.3
        )

        # -----------------------------------------
        # 11.3 放大显示
        # -----------------------------------------

        scale = 6

        resized = cv2.resize(
            image,
            (
                GRID_COLS * scale,
                GRID_ROWS * scale
            ),
            interpolation=cv2.INTER_NEAREST
        )

        # 标题栏
        header = np.zeros(
            (45, GRID_COLS * scale, 3),
            dtype=np.uint8
        )

        cv2.putText(
            header,
            'Local OCC | X Forward | Y Left | Resolution: 0.05m',
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        canvas = np.vstack((
            header,
            resized
        ))

        cv2.imshow(
            'RGB-D Local Occupancy Grid',
            canvas
        )

        cv2.waitKey(1)

    # =================================================
    # 12. 定时更新
    # =================================================

    def update_map(self):

        # 只有新点云到达时才重新计算地图
        if self.map_dirty:

            self.rebuild_map()

            self.publish_occupancy_grid()

            self.frame_count += 1

            if self.frame_count % 20 == 0:

                free_count = np.count_nonzero(
                    self.grid == 0
                )

                occupied_count = np.count_nonzero(
                    self.grid == 100
                )

                unknown_count = np.count_nonzero(
                    self.grid == -1
                )

                self.get_logger().info(
                    f'OCC | '
                    f'Free={free_count} | '
                    f'Occupied={occupied_count} | '
                    f'Unknown={unknown_count}'
                )

        # 即使没有新点云，也持续刷新窗口
        if self.map_ready:

            self.show_occupancy_grid()


# =================================================
# 13. 主程序
# =================================================

def main(args=None):

    rclpy.init(args=args)

    node = OCCMapping()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        cv2.destroyAllWindows()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
