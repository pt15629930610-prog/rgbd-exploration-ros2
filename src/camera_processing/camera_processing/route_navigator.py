"""Execute a language-grounded sequence of greenhouse road segments with live RGB-D clearance."""

from concurrent.futures import ThreadPoolExecutor
import json
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String
from nav_msgs.msg import Path

from .aisle_tracking import ARRIVAL_M, crossed_right_door, observed_path
from .door_log_sync import windows_log_server
from .door_run_log import DoorRunLog, RUN_LOG_DIRECTORY
from .frontier_explorer import FrontierExplorer, seconds, wrap
from .frontier_planner import clear_segment
from .greenhouse_roads import ENTRANCE_POSE, RIGHT_SILL, compile_route, from_world, to_world
from .route_language import interpret


class RouteNavigator(FrontierExplorer):
    def __init__(self):
        super().__init__()
        self.phase = 'idle'
        self.entrance = ENTRANCE_POSE
        self.server = self.declare_parameter('vlm_server', 'http://127.0.0.1:18080').value
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.journal = None
        self.mission_pub = self.create_publisher(Path, '/greenhouse/itinerary', 1)
        self.feedback = self.create_publisher(String, '/greenhouse/feedback', 10)
        self.create_subscription(String, '/greenhouse/instruction', self.instruction, 5)
        try:
            self.journal = DoorRunLog(RUN_LOG_DIRECTORY, windows_log_server(self.server))
            self.get_logger().info(f'大棚路线日志 {self.journal.path}')
        except OSError as exc:
            self.get_logger().warning(f'日志创建失败 {exc}')
        self.say('Waiting for a greenhouse route.', '请在 route_console 输入路线')

    def odometry(self, msg):
        if msg.header.frame_id != 'odom' or msg.child_frame_id != 'base_link':
            return
        if seconds(msg.header) < self.stamps['odom']-.5:
            self.hold('Simulation restarted.', '仿真时钟回退，已取消旧路线')
            self.publish_points(self.mission_pub, [])
        super().odometry(msg)

    def say(self, status, message):
        if self.status == status:
            return
        self.report(status)
        self.get_logger().info(message)
        self.feedback.publish(String(data=message))
        if self.journal is not None:
            try:
                self.journal.event(message)
            except OSError as exc:
                self.get_logger().warning(f'路线日志写入失败 {exc}')

    def hold(self, status, message):
        self.command()
        self.set_path([])
        self.phase = 'idle'
        if self.pending is not None:
            self.pending.cancel()
            self.pending = None
        self.say(status, message)

    def instruction(self, msg):
        text = msg.data.strip()
        if not text:
            return
        self.hold('Stopped.', '已停车并取消上一条指令')
        self.publish_points(self.mission_pub, [])
        if text in ('停车', '取消', '停止', '结束'):
            return
        self.translation = []
        self.pending = self.worker.submit(interpret, self.server, text, self.translation)
        self.phase = 'interpret'
        self.say('Reading instruction with Qwen.', '收到指令：' + text)

    def begin_leg(self, now):
        self.set_path([])
        self.wait_started = None
        self.next_plan = now
        self.look = 0
        self.settled = None
        self.best_distance = math.inf
        self.last_progress = now
        self.last_pose = from_world(self.pose[:2], self.entrance)
        self.phase = 'follow'
        step = self.mission[self.index]
        if step.point == RIGHT_SILL:
            self.crossed = False
        self.describe_leg()

    def describe_leg(self):
        step = self.mission[self.index]
        self.say(f'Route {self.index+1}/{len(self.mission)} at ({step.point[0]:.1f}, {step.point[1]:.1f})',
                 f'执行 {self.index+1}/{len(self.mission)}：{step.label}，'
                 f'入口坐标（前 {step.point[0]:.2f}，左 {step.point[1]:.2f}）米')

    def activate(self, now):
        position = from_world(self.pose[:2], self.entrance)
        try:
            self.mission = compile_route(self.steps, position)
        except ValueError as exc:
            self.hold('Route needs clarification.', str(exc))
            return
        if not self.mission:
            self.hold('Already at requested location.', '当前位置已满足任务')
            return
        self.index = 0
        self.edge_start = self.pose[:2]
        self.crossed = False
        self.publish_points(self.mission_pub, [self.edge_start] +
                            [to_world(s.point, self.entrance) for s in self.mission])
        self.begin_leg(now)

    def tick(self):
        if self.phase == 'idle':
            return
        if self.phase == 'interpret':
            self.command()
            if not self.pending.done():
                return
            try:
                self.steps = self.pending.result()
            except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
                self.hold('Instruction not executable.', '\n'.join([*self.translation, f'未执行：{exc}']))
                return
            self.pending = None
            self.phase = 'ready'
            self.say('Instruction parsed.', '\n'.join(self.translation) +
                     '\n解析动作：' + json.dumps(self.steps, ensure_ascii=False))
        now = self.get_clock().now().nanoseconds*1e-9
        if self.pose is None or any(not -.10 <= now-t <= 1.2 for t in self.stamps.values()):
            self.command()
            if self.phase == 'follow':
                if self.path:
                    self.set_path([])
                self.next_plan, self.last_progress, self.wait_started = now, now, None
                self.last_pose = None
            self.say('Waiting for live occupancy and pose.', '停车等待新鲜的位姿和占据图')
            return
        if self.turn_blocked or self.mapped_turn_obstacle():
            self.hold('Holding: obstacle in chassis clearance.', '底盘净空内出现障碍，已停车')
            return
        if self.phase == 'ready':
            self.activate(now)
            return
        if self.status == 'Waiting for live occupancy and pose.':
            self.describe_leg()
        self.follow(now)

    def follow(self, now):
        step = self.mission[self.index]
        here = from_world(self.pose[:2], self.entrance)
        if self.last_pose is not None and math.dist(self.last_pose, here) > .65:
            self.hold('Holding: discontinuous odometry.', '位姿出现跳变，已停止路线')
            return
        if step.point == RIGHT_SILL or step.outside:
            if crossed_right_door(self.last_pose, here, RIGHT_SILL):
                self.crossed = True
        self.last_pose = here
        dist = math.dist(here, step.point)
        if dist <= ARRIVAL_M:
            self.command()
            if step.wait:
                if self.wait_started is None:
                    self.wait_started = now
                if now-self.wait_started < step.wait:
                    return
            if step.outside and not self.crossed:
                self.hold('Holding: door crossing unverified.', '缺少连续位姿穿越右门的证据，未判定出棚')
                return
            self.say(f'Reached checkpoint {self.index+1}.',
                     f'完成 {self.index+1}：{step.label}，实测入口坐标 ({here[0]:.2f}, {here[1]:.2f}) 米')
            self.edge_start = to_world(step.point, self.entrance)
            self.index += 1
            if self.index == len(self.mission):
                self.hold('Route complete.', '路线完成，小车已停车')
            else:
                self.begin_leg(now)
            return
        self.wait_started = None
        if dist < self.best_distance-.07:
            self.best_distance, self.last_progress = dist, now
        if now-self.last_progress > 25.:
            self.hold('Holding: no progress on required road.', '当前必经路段持续无进展，已停车')
            return
        target = to_world(step.point, self.entrance)
        if now >= self.next_plan:
            path = observed_path(self.walk, (self.origin.x, self.origin.y), self.resolution,
                                 self.pose[:2], self.edge_start, target)
            if path != self.path:
                self.set_path(path)
            self.next_plan = now+.45
        cell = self.cell()
        ahead = next((self.xy(c) for c in reversed(self.path)
                      if .07 < math.dist(self.xy(c), self.pose[:2]) <= .39
                      and clear_segment(self.walk, cell, c)), None)
        if ahead is None:
            self.look_for_road(now, target)
            return
        self.look, self.settled = 0, None
        angle = wrap(math.atan2(ahead[1]-self.pose[1], ahead[0]-self.pose[0])-self.pose[2])
        angular = max(-.42, min(.42, 1.45*angle))
        if abs(angle) > .14:
            self.command(angular=angular)
        elif self.forward_blocked:
            self.command()
        else:
            self.command(min(self.speed_limit, .55*dist), angular)

    def look_for_road(self, now, target):
        offsets = (0., .45, -.45, 0.)
        bearing = math.atan2(target[1]-self.pose[1], target[0]-self.pose[0])
        error = wrap(bearing+offsets[self.look]-self.pose[2])
        if abs(error) > .075:
            self.settled = None
            self.command(angular=max(-.42, min(.42, 1.4*error)))
            return
        self.command()
        if self.settled is None:
            self.settled = now
        if now-self.settled < 1. or self.stamps['global'] <= self.settled:
            return
        self.look += 1
        self.settled = None
        if self.look == len(offsets):
            self.hold('Holding: required aisle not traversable.',
                      '补看后仍无已观测的安全通路，已停在当前必经段')


def main():
    rclpy.init()
    node = RouteNavigator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.command()
        if node.journal:
            try:
                node.journal.event('路线节点退出，最后状态：' + node.status)
            finally:
                node.journal.close()
                print(f'大棚路线日志 {node.journal.path}', flush=True)
        node.worker.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
