
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import deque

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry


# =====================================================
# 1. 全局地图参数
# =====================================================

# 全局地图范围：12m × 12m
MAP_X_MIN = -6.0
MAP_X_MAX = 6.0

MAP_Y_MIN = -6.0
MAP_Y_MAX = 6.0

# 全局地图分辨率
MAP_RESOLUTION = 0.05

MAP_WIDTH = int(
    (MAP_X_MAX - MAP_X_MIN) / MAP_RESOLUTION
)

MAP_HEIGHT = int(
    (MAP_Y_MAX - MAP_Y_MIN) / MAP_RESOLUTION
)

# 地图显示窗口刷新频率
DISPLAY_PERIOD = 0.1

# 局部地图的实际坐标系名称
LOCAL_FRAME = "camera_aligned"

# 里程计时间匹配最大误差
MAX_ODOM_DT = 0.25


# =====================================================
# 2. 四元数转换为航向角
# =====================================================

def quaternion_to_yaw(q):

    return np.arctan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def stamp_to_seconds(stamp):

    return stamp.sec + stamp.nanosec * 1e-9


# =====================================================
# 3. 全局地图累计节点
# =====================================================

class MapAccumulator(Node):

    def __init__(self):

        super().__init__("map_accumulator")

        # -1: Unknown
        #  0: Free
        # 100: Occupied

        # 全局地图内部采用 [y_index, x_index] 排列
        self.global_grid = np.full(
            (MAP_HEIGHT, MAP_WIDTH),
            -1,
            dtype=np.int8
        )

        # 里程计缓存
        self.odom_buffer = deque(maxlen=300)

        self.latest_odom = None

        self.map_count = 0

        self.warned_frame = False

        # ---------------------------------------------
        # 订阅原有局部 OCC 地图
        # ---------------------------------------------

        self.local_map_sub = self.create_subscription(
            OccupancyGrid,
            "/rgbd_camera/occ_grid",
            self.local_map_callback,
            10
        )

        # ---------------------------------------------
        # 订阅小车里程计
        # ---------------------------------------------

        self.odom_sub = self.create_subscription(
            Odometry,
            "/model/mobile_robot/odometry",
            self.odom_callback,
            50
        )

        # ---------------------------------------------
        # 发布全局 OCC 地图
        # ---------------------------------------------

        self.global_map_pub = self.create_publisher(
            OccupancyGrid,
            "/exploration/map",
            10
        )

        # ---------------------------------------------
        # OpenCV 定时显示
        # ---------------------------------------------

        self.timer = self.create_timer(
            DISPLAY_PERIOD,
            self.display_map
        )

        cv2.namedWindow(
            "Global OCC Map",
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            "Global OCC Map",
            850,
            850
        )

        self.get_logger().info(
            "Global Map Accumulator Started!"
        )

        self.get_logger().info(
            f"Map Size: {MAP_WIDTH}x{MAP_HEIGHT} | "
            f"Resolution: {MAP_RESOLUTION}m"
        )

    # =================================================
    # 4. 里程计回调
    # =================================================

    def odom_callback(self, msg):

        if msg.header.frame_id != "odom":
            return

        if msg.child_frame_id != "base_link":
            return

        self.odom_buffer.append(msg)

        self.latest_odom = msg

    # =================================================
    # 5. 根据地图时间寻找对应里程计
    # =================================================

    def find_nearest_odom(self, stamp):

        if len(self.odom_buffer) == 0:
            return None, None

        map_time = stamp_to_seconds(stamp)

        best = min(
            self.odom_buffer,
            key=lambda m: abs(
                stamp_to_seconds(m.header.stamp) - map_time
            )
        )

        dt = abs(
            stamp_to_seconds(best.header.stamp) - map_time
        )

        if dt > MAX_ODOM_DT:
            return None, dt

        return best, dt

    # =================================================
    # 6. 局部地图回调
    # =================================================

    def local_map_callback(self, msg):

        # 检查局部地图坐标系
        if msg.header.frame_id != LOCAL_FRAME:

            if not self.warned_frame:

                self.get_logger().warning(
                    f"Unexpected local frame: "
                    f"{msg.header.frame_id}"
                )

                self.warned_frame = True

            return

        # 找到地图时间对应的里程计
        odom, dt = self.find_nearest_odom(
            msg.header.stamp
        )

        if odom is None:

            self.get_logger().warning(
                f"No synchronized odometry. "
                f"Time difference={dt}"
            )

            return

        self.integrate_local_map(
            msg,
            odom
        )

        self.map_count += 1

        # 发布全局地图
        self.publish_global_map(
            msg.header.stamp
        )

        if self.map_count % 10 == 0:

            local = np.asarray(
                msg.data,
                dtype=np.int16
            )

            self.get_logger().info(
                f"Map Frames={self.map_count} | "
                f"Local Occupied={np.count_nonzero(local >= 50)} | "
                f"Local Free={np.count_nonzero(local == 0)} | "
                f"Global Occupied={np.count_nonzero(self.global_grid == 100)} | "
                f"Global Free={np.count_nonzero(self.global_grid == 0)}"
            )

    # =================================================
    # 7. 局部地图转换到 odom 坐标系
    # =================================================

    def integrate_local_map(self, msg, odom):

        info = msg.info

        width = int(info.width)
        height = int(info.height)

        resolution = float(info.resolution)

        if width <= 0 or height <= 0 or resolution <= 0:
            return

        data = np.asarray(
            msg.data,
            dtype=np.int16
        )

        if len(data) != width * height:
            self.get_logger().error(
                "OccupancyGrid data size mismatch!"
            )
            return

        # ROS OccupancyGrid:
        # data 按照 [y, x] 顺序排列
        local_grid = data.reshape(
            (height, width)
        )

        # ---------------------------------------------
        # 7.1 找到已经观测过的栅格
        # ---------------------------------------------

        iy, ix = np.indices(
            (height, width)
        )

        known = local_grid >= 0

        if not np.any(known):
            return

        ix = ix[known]
        iy = iy[known]

        values = local_grid[known]

        # ---------------------------------------------
        # 7.2 局部栅格中心坐标
        # ---------------------------------------------

        dx = (
            ix + 0.5
        ) * resolution

        dy = (
            iy + 0.5
        ) * resolution

        # 地图原点姿态
        map_yaw = quaternion_to_yaw(
            info.origin.orientation
        )

        cm = np.cos(map_yaw)
        sm = np.sin(map_yaw)

        local_x = (
            info.origin.position.x +
            cm * dx - sm * dy
        )

        local_y = (
            info.origin.position.y +
            sm * dx + cm * dy
        )

        # ---------------------------------------------
        # 7.3 获取小车在 odom 下的位置
        # ---------------------------------------------

        pose = odom.pose.pose

        robot_x = pose.position.x
        robot_y = pose.position.y

        robot_yaw = quaternion_to_yaw(
            pose.orientation
        )

        cr = np.cos(robot_yaw)
        sr = np.sin(robot_yaw)

        # ---------------------------------------------
        # 7.4 局部坐标转换为全局坐标
        # ---------------------------------------------

        global_x = (
            robot_x +
            cr * local_x -
            sr * local_y
        )

        global_y = (
            robot_y +
            sr * local_x +
            cr * local_y
        )

        # ---------------------------------------------
        # 7.5 转换为全局栅格索引
        # ---------------------------------------------

        gx = np.floor(
            (global_x - MAP_X_MIN) /
            MAP_RESOLUTION
        ).astype(np.int32)

        gy = np.floor(
            (global_y - MAP_Y_MIN) /
            MAP_RESOLUTION
        ).astype(np.int32)

        valid = (
            (gx >= 0) &
            (gx < MAP_WIDTH) &
            (gy >= 0) &
            (gy < MAP_HEIGHT)
        )

        gx = gx[valid]
        gy = gy[valid]

        values = values[valid]

        if len(values) == 0:
            return

        # ---------------------------------------------
        # 7.6 融合地图
        # ---------------------------------------------

        # 先更新空闲区域
        free_mask = values == 0

        fx = gx[free_mask]
        fy = gy[free_mask]

        # 已经记录的障碍物不被空闲观测覆盖
        old_values = self.global_grid[fy, fx]

        self.global_grid[fy, fx] = np.where(
            old_values == 100,
            100,
            0
        )

        # 再更新障碍物区域
        occupied_mask = values >= 50

        ox = gx[occupied_mask]
        oy = gy[occupied_mask]

        self.global_grid[oy, ox] = 100

    # =================================================
    # 8. 发布全局 OccupancyGrid
    # =================================================

    def publish_global_map(self, stamp):

        msg = OccupancyGrid()

        msg.header.stamp = stamp
        msg.header.frame_id = "odom"

        msg.info.map_load_time = stamp

        msg.info.resolution = MAP_RESOLUTION

        msg.info.width = MAP_WIDTH
        msg.info.height = MAP_HEIGHT

        msg.info.origin.position.x = MAP_X_MIN
        msg.info.origin.position.y = MAP_Y_MIN
        msg.info.origin.position.z = 0.0

        msg.info.origin.orientation.w = 1.0

        msg.data = (
            self.global_grid
            .astype(np.int8)
            .flatten()
            .tolist()
        )

        self.global_map_pub.publish(msg)

    # =================================================
    # 9. OpenCV 全局地图显示
    # =================================================

    def display_map(self):

        # OpenCV 的第0行在图像顶部
        # ROS 地图的第0行对应最小Y
        display_grid = np.flipud(
            self.global_grid
        )

        image = np.zeros(
            (MAP_HEIGHT, MAP_WIDTH, 3),
            dtype=np.uint8
        )

        # Unknown：灰色
        image[display_grid == -1] = (
            75, 75, 75
        )

        # Free：白色
        image[display_grid == 0] = (
            240, 240, 240
        )

        # Occupied：黑色
        image[display_grid >= 50] = (
            0, 0, 0
        )

        # ---------------------------------------------
        # 9.1 绘制每隔1m的辅助网格
        # ---------------------------------------------

        interval = int(
            1.0 / MAP_RESOLUTION
        )

        for col in range(0, MAP_WIDTH, interval):

            cv2.line(
                image,
                (col, 0),
                (col, MAP_HEIGHT - 1),
                (180, 180, 180),
                1
            )

        for row in range(0, MAP_HEIGHT, interval):

            cv2.line(
                image,
                (0, row),
                (MAP_WIDTH - 1, row),
                (180, 180, 180),
                1
            )

        # ---------------------------------------------
        # 9.2 绘制小车位置和朝向
        # ---------------------------------------------

        if self.latest_odom is not None:

            pose = self.latest_odom.pose.pose

            rx = pose.position.x
            ry = pose.position.y

            yaw = quaternion_to_yaw(
                pose.orientation
            )

            col = int(
                (rx - MAP_X_MIN) / MAP_RESOLUTION
            )

            row = MAP_HEIGHT - 1 - int(
                (ry - MAP_Y_MIN) / MAP_RESOLUTION
            )

            if (
                0 <= col < MAP_WIDTH and
                0 <= row < MAP_HEIGHT
            ):

                cv2.circle(
                    image,
                    (col, row),
                    5,
                    (255, 0, 0),
                    -1
                )

                # 朝向箭头
                arrow_length = 20

                end_col = int(
                    col + arrow_length * np.cos(yaw)
                )

                end_row = int(
                    row - arrow_length * np.sin(yaw)
                )

                cv2.arrowedLine(
                    image,
                    (col, row),
                    (end_col, end_row),
                    (255, 0, 0),
                    2,
                    tipLength=0.3
                )

        # ---------------------------------------------
        # 9.3 地图状态统计
        # ---------------------------------------------

        unknown_count = np.count_nonzero(
            self.global_grid == -1
        )

        free_count = np.count_nonzero(
            self.global_grid == 0
        )

        occupied_count = np.count_nonzero(
            self.global_grid == 100
        )

        known_count = (
            free_count + occupied_count
        )

        # 标题区域
        header = np.zeros(
            (65, MAP_WIDTH, 3),
            dtype=np.uint8
        )

        cv2.putText(
            header,
            "GLOBAL OCC MAP",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        info_text = (
            f"Free:{free_count} "
            f"Occupied:{occupied_count} "
            f"Unknown:{unknown_count}"
        )

        cv2.putText(
            header,
            info_text,
            (8, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1
        )

        canvas = np.vstack((
            header,
            image
        ))

        scale = 3

        canvas = cv2.resize(
            canvas,
            (
                MAP_WIDTH * scale,
                (MAP_HEIGHT + 65) * scale
            ),
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow(
            "Global OCC Map",
            canvas
        )

        key = cv2.waitKey(1) & 0xFF

        # 按 S 保存地图
        if key == ord('s'):

            save_path = os.path.expanduser(
                "~/robot_ws/global_occ_map.png"
            )

            cv2.imwrite(
                save_path,
                canvas
            )

            self.get_logger().info(
                f"Map saved: {save_path}"
            )

        # 按 Q 退出
        elif key == ord('q'):

            self.get_logger().info(
                "Closing Global Map Viewer."
            )

            rclpy.shutdown()


# =====================================================
# 10. 主函数
# =====================================================

def main(args=None):

    rclpy.init(args=args)

    node = MapAccumulator()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        cv2.destroyAllWindows()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":

    main()
