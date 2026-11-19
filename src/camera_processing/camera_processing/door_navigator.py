"""Observe selected headings, search reachable frontiers, and park at a verified door."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math

import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from .door_snapshot import query_image
from .door_task import door_box, locate_panel, parse_command, search_path
from .frontier_explorer import FrontierExplorer, seconds, wrap
from .frontier_planner import astar


def observe(server, rgb, depth, fx, cx, pose, direction):
    ok, image = cv2.imencode('.jpg', rgb)
    if not ok:
        raise ValueError('JPEG encoding failed')
    visible = query_image(server, image, 'jpeg',
                          '只根据这张画面，能辨认出房门本体吗？看见回答“看见”，'
                          '没有或不确定回答“未见”。只输出一个选项。').strip(' \t\n。.!！“”')
    if visible == '未见':
        return None
    if visible != '看见':
        raise ValueError('无法判读模型回答：' + visible)
    reply = query_image(server, image, 'jpeg',
                        '查找有门框或把手的房门。图中没有可辨认的门就只输出 []。'
                        '若可见，只输出门板的二维边界框 [xmin,ymin,xmax,ymax]，坐标归一化到0到1000。')
    box = door_box(reply)
    return None if box is None else locate_panel(box, depth, fx, cx, pose, direction)


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
        self.create_subscription(CameraInfo, '/rgbd_camera/camera_info', self.info, qos_profile_sensor_data)
        subs = [message_filters.Subscriber(self, Image, '/rgbd_camera/' + topic,
                                           qos_profile=qos_profile_sensor_data)
                for topic in ('image', 'depth_image')]
        self.sync = message_filters.ApproximateTimeSynchronizer(subs, 5, .035)
        self.sync.registerCallback(self.images)
        self.report('Waiting for /door/task')

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
        self.phase = 'start'
        self.report(f'Received: {msg.data} | angle {math.degrees(task[0]):+.0f} deg, '
                    f'stop {task[1]:.2f} m, speed <= {self.speed_limit:.2f} m/s.')

    def begin_scan(self, confirming=False):
        self.set_path([])
        self.verifying = (self.landmark is not None and
                          self.near_parking_point(self.landmark))
        offset = self.task[0]
        toward_center = -1 if offset > 0 else 1
        offsets = (0,) if self.verifying else (0, toward_center*.85, -toward_center*.85)
        self.headings = deque(self.direction + angle for angle in offsets)
        self.next_view()

    def next_view(self):
        if self.headings:
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
            self.pending = self.vision_worker.submit(observe, self.server, image, metres,
                                                     *self.intrinsics, pose, self.direction)
            self.camera_pose = pose
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
            self.turn_started = self.get_clock().now().nanoseconds * 1e-9
            return
        try:
            point = self.pending.result()
        except Exception as exc:
            self.halt('Vision request failed: ' + str(exc))
            return
        normal = np.array([math.cos(self.direction), math.sin(self.direction)])
        if point is None or np.dot(point-self.entrance[:2], normal) < .6:
            self.next_view()
            return
        previous = self.landmark
        if previous is not None and np.linalg.norm(point-previous) > .65:
            self.next_view()
            return
        self.landmark = point
        if previous is not None and self.verifying and self.near_parking_point(point):
            self.halt(f'Arrived: {np.dot(point-self.pose[:2], normal):.2f} m from door.')
            return
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
                self.start_drive(path, now)
                return
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
    finally:
        if rclpy.ok():
            node.command()
        node.vision_worker.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
