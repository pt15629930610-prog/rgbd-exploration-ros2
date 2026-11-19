"""Observe selected headings, search reachable frontiers, and park at a verified door."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from .door_run_log import DoorRunLog
from .door_snapshot import MODEL, query_image
from .door_task import door_box, locate_panel, parse_command, search_path
from .frontier_explorer import FrontierExplorer, seconds, wrap
from .frontier_planner import astar


def observe(server, image, depth, fx, cx, pose, direction):
    raw = query_image(server, image, 'jpeg',
                          '只根据这张画面，能辨认出房门本体吗？看见回答“看见”，'
                          '没有或不确定回答“未见”。只输出一个选项。')
    details = [f'Qwen 可见性原答={raw.strip()!r}']
    visible = raw.strip(' \t\n。.!！“”')
    if visible == '未见':
        return None, '；'.join(details)
    if visible != '看见':
        raise ValueError('无法判读模型回答：' + visible)
    reply = query_image(server, image, 'jpeg',
                        '查找有门框或把手的房门。图中没有可辨认的门就只输出 []。'
                        '若可见，只输出门板的二维边界框 [xmin,ymin,xmax,ymax]，坐标归一化到0到1000。')
    details.append(f'门框原答={reply.strip()!r}')
    try:
        box = door_box(reply)
    except ValueError as exc:
        raise ValueError('；'.join(details) + '；门框解析失败') from exc
    point = None if box is None else locate_panel(box, depth, fx, cx, pose, direction,
                                                  details=details)
    return point, '；'.join(details)


class DoorNavigator(FrontierExplorer):
    def __init__(self):
        super().__init__()
        self.phase = 'idle'
        self.entrance = self.task = self.landmark = None
        self.server = self.declare_parameter('vlm_server', 'http://127.0.0.1:18080').value
        self.bridge = CvBridge()
        self.intrinsics = self.pair = self.pending = None
        self.vision_worker = ThreadPoolExecutor(max_workers=1)
        self.generation = 0
        self.poses = deque(maxlen=100)
        self.create_subscription(String, '/door/task', self.new_task, 5)
        self.frame_pub = self.create_publisher(Image, '/door/observation_image', 1)
        self.create_subscription(CameraInfo, '/rgbd_camera/camera_info', self.info, qos_profile_sensor_data)
        subs = [message_filters.Subscriber(self, Image, '/rgbd_camera/' + topic,
                                           qos_profile=qos_profile_sensor_data)
                for topic in ('image', 'depth_image')]
        self.sync = message_filters.ApproximateTimeSynchronizer(subs, 5, .035)
        self.sync.registerCallback(self.images)
        self.run_log = None
        try:
            self.run_log = DoorRunLog(Path.home() / 'door_nav_logs')
            self.get_logger().info(f'本次导航日志：{self.run_log.path}')
            self.record(f'四门导航启动，模型 {MODEL}，服务 {self.server}')
        except OSError as exc:
            self.get_logger().warning(f'无法创建导航日志：{exc}')
        self.report('Waiting for /door/task')

    def record(self, message, jpeg=None):
        if self.run_log is not None:
            try:
                self.run_log.event(message, jpeg)
            except OSError as exc:
                self.get_logger().warning(f'导航日志写入失败：{exc}')
                try:
                    self.run_log.close()
                except OSError:
                    pass
                self.run_log = None

    def report(self, status):
        if self.status != status:
            self.record(status)
        super().report(status)

    def images(self, rgb, depth):
        self.pair = (seconds(rgb.header), rgb, depth)

    def info(self, msg):
        if msg.k[0] > 0:
            self.intrinsics = (msg.k[0], msg.k[2])

    def odometry(self, msg):
        super().odometry(msg)
        if self.pose is not None and self.stamps['odom'] == seconds(msg.header):
            self.poses.append((self.stamps['odom'], self.pose))
            if self.entrance is None:
                self.entrance = self.pose

    def new_task(self, msg):
        cancel = msg.data.strip() in ('停止', '停车', '取消')
        if not cancel:
            try:
                task, speed = parse_command(msg.data)
            except ValueError as exc:
                self.report(str(exc))
                return
            if task is None:
                if self.phase == 'idle':
                    self.report('No active door task. Include speed in the next door command.')
                else:
                    self.speed_limit = speed
                    self.report(f'Speed limit updated: {speed:.2f} m/s.')
                return
        self.command()
        self.set_path([])
        self.generation += 1
        self.phase = 'idle'
        self.task = self.landmark = None
        if cancel:
            self.report('Task cancelled.')
            return
        self.task = task
        self.speed_limit = .16 if speed is None else speed
        self.failed.clear()
        self.scanned.clear()
        self.failed_parking_cell = None
        self.route_failures = 0
        self.view_number = 0
        self.last_view_pose = None
        self.phase = 'start'
        self.report(f'Received: {msg.data} | angle {math.degrees(task[0]):+.0f} deg, '
                    f'stop {task[1]:.2f} m, speed <= {self.speed_limit:.2f} m/s.')

    def begin_scan(self, confirming=False):
        self.set_path([])
        self.en_route = False
        self.verifying = (self.landmark is not None and
                          self.near_parking_point(self.landmark))
        offset = self.task[0]
        toward_center = -1 if offset > 0 else 1
        offsets = (0,) if self.verifying else (0, toward_center*.85, -toward_center*.85)
        self.headings = deque(self.direction + angle for angle in offsets)
        self.next_view()

    def next_view(self):
        if self.en_route:
            self.en_route = False
            self.phase = 'drive'
            self.progress_time = self.get_clock().now().nanoseconds * 1e-9
            self.log_view('本次行进复看未锁定目标，恢复跟踪并重新检查路径')
        elif self.headings:
            self.heading = self.headings.popleft()
            self.phase = 'turn'
            self.turn_started = self.get_clock().now().nanoseconds * 1e-9
        else:
            if self.verifying:
                self.halt('Door could not be confirmed at the parking point.')
                return
            self.scanned.append(self.cell())
            self.phase = 'plan'

    def halt(self, reason):
        self.command()
        self.set_path([])
        self.phase = 'idle'
        self.report(reason)

    def tick(self):
        if self.phase == 'idle':
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.entrance is None or any(not -.1 <= now-t <= 1.2 for t in self.stamps.values()):
            self.command()
            self.report('Waiting for fresh pose and occupancy maps.')
            return
        if self.turn_blocked or self.mapped_turn_obstacle():
            self.halt('Stopped: obstacle inside chassis clearance.')
            return
        if self.phase == 'start':
            self.direction = self.entrance[2] + self.task[0]
            self.begin_scan()
        elif self.phase == 'drive':
            if not self.parking_route and self.due_for_view():
                self.command()
                self.en_route = True
                self.verifying = False
                self.heading = self.pose[2]
                self.capture_after = now + .25
                self.phase = 'capture'
            else:
                self.drive(now, check_frontier=False)
        elif self.phase == 'plan':
            self.plan_task(now)
        elif self.phase == 'turn':
            error = wrap(self.heading-self.pose[2])
            if abs(error) <= .055:
                self.command()
                self.capture_after = now + .25
                self.phase = 'capture'
            elif now-self.turn_started > 25:
                self.halt('Stopped: observation turn timed out.')
            else:
                self.command(angular=max(-.48, min(.48, 1.5*error)))
        elif self.phase == 'capture':
            self.capture(now)
        elif self.phase == 'vision':
            self.command()
            if self.pending.done():
                self.finish_view()

    def due_for_view(self):
        if self.last_view_pose is None:
            return True
        distance = math.dist(self.pose[:2], self.last_view_pose[:2])
        angle = abs(wrap(self.pose[2] - self.last_view_pose[2]))
        return distance >= .9 or (distance >= .25 and angle >= math.radians(35))

    def log_view(self, message, jpeg=None):
        line = f'[门观测 {self.view_number}] {message}'
        self.get_logger().info(line)
        self.record(line, jpeg)

    def capture(self, now):
        self.command()
        # An old HTTP request may finish after cancellation. Never apply it
        # to a replacement command, or enqueue a backlog of pictures.
        if self.pending is not None and not self.pending.done():
            return
        if not self.pair or not self.intrinsics or not self.poses or not (
                self.capture_after <= self.pair[0] and -.1 <= now-self.pair[0] <= .7):
            self.report('Waiting for synchronized RGB, depth and camera info.')
            return
        stamp, rgb, depth = self.pair
        t, pose = min(self.poses, key=lambda p: abs(p[0]-stamp))
        if abs(t-stamp) > .12:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')
            metres = self.bridge.imgmsg_to_cv2(depth, 'passthrough')
            if depth.encoding == '16UC1':
                metres = metres.astype(np.float32)*.001
            elif depth.encoding != '32FC1':
                raise ValueError('Unexpected depth encoding: ' + depth.encoding)
            if metres.shape != image.shape[:2]:
                raise ValueError('RGB and depth dimensions differ')
            ok, jpeg = cv2.imencode('.jpg', image)
            if not ok:
                raise ValueError('JPEG encoding failed')
            self.pending = self.vision_worker.submit(observe, self.server, jpeg, metres,
                                                     *self.intrinsics, pose, self.direction)
            self.camera_pose = pose
            self.last_view_pose = pose
            self.view_number += 1
            self.frame_pub.publish(rgb)
            source = '行进复看' if self.en_route else '门前复核' if self.verifying else '定向观察'
            self.log_view(f'{source}，RGB 时间戳 {stamp:.3f}，车位 ({pose[0]:.2f}, {pose[1]:.2f}) m，'
                          f'拍摄航向 {math.degrees(pose[2]):.0f}°，任务方向 {math.degrees(self.direction):.0f}°',
                          jpeg)
        except Exception as exc:
            self.halt('Camera error: ' + str(exc))
            return
        self.request_id = self.generation
        self.phase = 'vision'
        self.report('Checking the observed door with Qwen and depth.')

    def finish_view(self):
        if self.request_id != self.generation:
            return
        if (math.dist(self.pose[:2], self.camera_pose[:2]) > .10 or
                abs(wrap(self.pose[2]-self.camera_pose[2])) > .09):
            self.phase = 'turn'
            self.log_view('等待推理期间车身位姿变化，丢弃旧帧，重新取图')
            self.turn_started = self.get_clock().now().nanoseconds * 1e-9
            return
        try:
            point, details = self.pending.result()
        except Exception as exc:
            self.halt('Vision request failed: ' + str(exc))
            return
        normal = np.array([math.cos(self.direction), math.sin(self.direction)])
        self.log_view(details)
        if point is None:
            self.log_view('没有通过视觉与深度检查的门位，继续观察')
            self.next_view()
            return
        projection = np.dot(point-self.entrance[:2], normal)
        self.log_view(f'候选门 ({point[0]:.2f}, {point[1]:.2f}) m，相对入口的方向投影 {projection:.2f} m')
        if projection < .6:
            self.log_view('方向投影小于 0.60 m，排除该候选门')
            self.next_view()
            return
        previous = self.landmark
        if previous is not None and np.linalg.norm(point-previous) > .65:
            self.log_view('与已记录门位相差超过 0.65 m，继续复查')
            self.next_view()
            return
        self.landmark = point
        self.en_route = False
        if previous is not None and self.verifying and self.near_parking_point(point):
            self.log_view('门前复核通过，距离与横向误差均满足停车条件')
            self.halt(f'Arrived: {np.dot(point-self.pose[:2], normal):.2f} m from door.')
            return
        self.log_view('接受门位，按指令距离计算停车点')
        self.set_path([])
        self.phase = 'plan'

    def near_parking_point(self, point):
        normal = np.array([math.cos(self.direction), math.sin(self.direction)])
        delta = point-self.pose[:2]
        lateral = abs(-normal[1]*delta[0] + normal[0]*delta[1])
        return abs(np.dot(delta, normal)-self.task[1]) <= .14 and lateral <= .22

    def route_arrived(self):
        self.route_failures = 0
        self.failed_parking_cell = None
        super().route_arrived()

    def retry_goal(self, reason):
        self.command()
        self.failed.append(self.path[-1])
        if self.parking_route:
            self.failed_parking_cell = self.path[-1]
        self.route_failures += 1
        if self.route_failures >= 4:
            self.halt(f'Stopped after 4 consecutive failed approaches: {reason}.')
        else:
            self.report(f'Approach {self.route_failures}/4: {reason}; checking alternate views.')
            self.begin_scan()

    def plan_task(self, now):
        self.command()
        start = self.cell()
        hint = None
        if self.landmark is not None:
            if self.near_parking_point(self.landmark):
                self.begin_scan()
                return
            x, y = self.landmark - self.task[1]*np.array(
                [math.cos(self.direction), math.sin(self.direction)])
            hint = (math.floor((y-self.origin.y)/self.resolution),
                    math.floor((x-self.origin.x)/self.resolution))
            recent_failure = (self.failed_parking_cell is not None and
                              math.dist(hint, self.failed_parking_cell)*self.resolution < .65)
            path = [] if recent_failure else astar(self.walk, start, hint)
            if path:
                self.parking_route = True
                self.report(f'Following parking route to ({x:.2f}, {y:.2f}) m.')
                self.start_drive(path, now)
                return
            why = '该处最近接近失败' if recent_failure else '当前可通行地图中 A* 未找到路径'
            self.log_view(f'已定位门，停车点 ({x:.2f}, {y:.2f}) m；{why}，改选靠近停车点的 frontier')
        path = search_path(self.walk, self.frontier, start, self.resolution, self.direction,
                           self.scanned+self.failed, hint)
        if path:
            self.parking_route = False
            self.start_drive(path, now)
            self.report('Driving to selected frontier at (%.2f, %.2f) m.' % self.xy(path[-1]))
        else:
            self.halt('Stopped: no reachable unvisited viewpoint for this task.')


def main():
    rclpy.init()
    node = DoorNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.record(f'导航异常：{exc!r}')
        raise
    finally:
        if rclpy.ok():
            node.command()
        node.record(f'导航节点退出，最后状态：{node.status}')
        if node.run_log is not None:
            try:
                node.run_log.close()
            except OSError as exc:
                node.get_logger().warning(f'关闭导航日志失败：{exc}')
        node.vision_worker.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
