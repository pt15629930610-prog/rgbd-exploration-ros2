"""ROS adapter for language-directed navigation on an initially unknown occupancy map."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from .door_log_sync import windows_log_server
from .door_run_log import DoorRunLog, RUN_LOG_DIRECTORY
from .frontier_explorer import FrontierExplorer, seconds
from .frontier_planner import obstacle_distance
from .online_language import interpret_online
from .online_roads import RoadView
from .online_skills import MotionSkills, Unmet, crossed_threshold
from .online_vision import LiveVision


class OnlineNavigator(MotionSkills, FrontierExplorer):
    def __init__(self):
        super().__init__()
        self.server = self.declare_parameter('vlm_server', 'http://127.0.0.1:18080').value
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.vision = LiveVision(self, self.server)
        self.feedback = self.create_publisher(String, '/greenhouse/feedback', 10)
        self.create_subscription(String, '/greenhouse/instruction', self.instruction, 5)
        self.phase = 'idle'
        self.program = self.pending = self.entrance = None
        self.feature = self.entered = self.loop_trace = self.portal = None
        self.last_row = None
        self.row_laps = 0
        self.vision_key = None
        self.responses = 0
        self.clearance_hold = None
        self.clearance_recovery = None
        self.motion_revision = 0
        self.recovery_sites = []
        self.last_image_log = -math.inf
        self.journal = None
        try:
            self.journal = DoorRunLog(RUN_LOG_DIRECTORY, windows_log_server(self.server))
            self.get_logger().info(f'在线探索日志 {self.journal.path}')
        except OSError as exc:
            self.get_logger().warning(f'日志创建失败 {exc}')
        self.note('在线探索已启动，等待入口位姿和自然语言指令', 'Waiting for instruction.')

    def note(self, message, status=None, jpeg=None):
        if status is not None:
            self.report(status)
        self.get_logger().info(message)
        self.feedback.publish(String(data=message))
        if self.journal:
            try:
                self.journal.event(message, jpeg)
                return True
            except OSError as exc:
                self.get_logger().warning(f'日志写入失败 {exc}')
        return False

    def global_map(self, msg):
        if msg.header.frame_id != 'odom':
            return
        super().global_map(msg)
        self.view = RoadView(self.grid, self.walk, (self.origin.x, self.origin.y), self.resolution)

    def local_map(self, msg):
        super().local_map(msg)
        if msg.header.frame_id != 'camera_aligned':
            return
        info = msg.info
        rows, cols = np.where(np.asarray(msg.data).reshape(info.height, info.width) >= 50)
        self.local_hits = (info.origin.position.x+(cols+.5)*info.resolution,
                           info.origin.position.y+(rows+.5)*info.resolution, info.resolution)

    def odometry(self, msg):
        if msg.header.frame_id != 'odom' or msg.child_frame_id != 'base_link':
            return
        stamp = seconds(msg.header)
        previous = self.pose
        if stamp < self.stamps['odom']-.5:
            self.stop()
            self.entrance = self.feature = self.last_row = None
            self.entered = self.loop_trace = None
            if hasattr(self, 'approach_track'):
                self.approach_track.clear()
            self.stamps['local'] = self.stamps['global'] = -math.inf
            previous = None
        elif stamp < self.stamps['odom']:
            return
        super().odometry(msg)
        if self.entrance is None:
            self.entrance = self.pose
            self.course = self.pose[2]
            self.note(f'入口参考取自首帧实测位姿 {self.entrance}')
        self.vision.pose(stamp, self.pose)
        if previous is not None and math.dist(previous[:2], self.here) > .65:
            self.stop()
            self.feature = self.last_row = None
            self.approach_track.clear()
            self.note('位姿出现跳变，任务已停止，先核对定位', 'Stopped: odometry jumped.')
            return
        self.remember_position()
        if self.portal and previous is not None and crossed_threshold(self.portal, previous, self.here):
            self.crossed = True
        if self.loop_trace is not None:
            self.loop_trace.observe(self.here)

    def stop(self, *, keep_row=False):
        self.command()
        self.clearance_hold = None
        self.clearance_recovery = None
        self.clear_path()
        if self.pending is not None:
            self.pending.cancel()
        self.pending = None
        self.program = None
        self.phase = 'idle'
        self.vision.reset()
        self.vision_key = self.portal = None
        if not keep_row:
            self.loop_trace = self.entered = None

    def instruction(self, msg):
        text = msg.data.strip()
        if not text:
            return
        cancel = text in ('停车', '取消', '停止', '结束', '急停')
        self.stop(keep_row=self.phase == 'idle' and not cancel)
        if cancel:
            self.feature = None
            self.note('已停车并取消当前指令', 'Stopped.')
            return
        if self.pose:
            # A new command starts at the current robot; cardinal language still refers to entrance.
            self.course = self.pose[2]
        self.speed_limit = .16
        self.skipped = []
        self.trace = []
        self.pending = self.worker.submit(interpret_online, self.server, text, self.trace)
        self.phase = 'reading'
        self.note(f'收到指令 {text}', 'Interpreting with Qwen.')

    def execute(self, steps):
        for i, step in enumerate(steps):
            self.step_index = i
            self.recovery_sites = []
            self.started = self.now
            self.deadline = self.now + max(720.*step.get('count', 1), step.get('seconds', 0.)+30.)
            self.vision_key = None
            self.vision.reset()
            self.clear_path()
            if step['do'] == 'unsupported' and not step.get('dependent', True):
                self.command()
                self.feature = self.loop_trace = self.entered = None
                self.skipped.append(step['reason'])
                self.note(f'第 {i+1} 步已跳过：{step["reason"]}')
                continue
            if 'speed' in step:
                self.speed_limit = step['speed']
            if step['do'] not in ('enter', 'loop', 'patrol', 'traverse_row', 'wait', 'photo', 'inspect', 'speed'):
                self.entered = self.loop_trace = None
            self.note(f'开始第 {i+1}/{len(steps)} 步 {step}，速度上限 {self.speed_limit:.2f} m/s',
                      f'Online {i+1}/{len(steps)}: {step["do"]}')
            following = steps[i+1]['do'] if i+1 < len(steps) else None
            yield from self.run_skill(step, following)
            self.command()
            self.clear_path()
            self.vision_key = None
            self.note(f'完成第 {i+1} 步 {step["do"]}，实测位置 {self.here}')
            yield

    def vision_tick(self):
        if not self.vision_key:
            return
        kind, label = self.vision_key
        evidence = self.vision.poll(kind, self.now, label)
        if evidence is not None:
            self.responses += 1
            if evidence.error:
                self.vision_errors += 1
                if self.vision_errors >= 2:
                    raise Unmet(evidence.detail)
            else:
                self.vision_errors = 0
            if kind in ('photo', 'inspect') and not evidence.error:
                self.snapshot = evidence
                return
            if evidence.point is not None:
                self.images.append(evidence)
            elif evidence.panel is not None:
                point, normal = evidence.panel
                self.panels.append(replace(evidence, point=point, heading=normal, jpeg=b''))
            if evidence.point is not None or self.now-self.last_image_log > 3.:
                self.note(f'帧 {evidence.stamp:.2f} 拍摄位姿 {evidence.pose} {evidence.detail}', jpeg=evidence.jpeg)
                self.last_image_log = self.now

    def clearance_image(self):
        row, col = self.cell()
        radius = math.ceil(2./self.resolution)
        top, left = max(0, row-radius), max(0, col-radius)
        cells = self.grid[top:row+radius+1, left:col+radius+1]
        if not cells.size:
            return None
        # Row indices increase along +Y; flip for the same orientation as the global window.
        grey = np.where(cells < 0, 105, np.where(cells >= 50, 15, 235)).astype(np.uint8)
        scale = 4
        image = cv2.resize(cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), None,
                           fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        def pixel(cell):
            return ((cell[1]-left)*scale+scale//2, (cell[0]-top)*scale+scale//2)
        if len(self.path) >= 2:
            cv2.polylines(image, [np.array([pixel(c) for c in self.path], np.int32)], False, (0,170,0), 2)
        center = pixel((row,col))
        cv2.circle(image, center, round(.42*scale/self.resolution), (0,0,235), 2)
        cv2.circle(image, center, 3, (255,0,0), -1)
        ok, encoded = cv2.imencode('.jpg', cv2.flip(image, 0))
        return encoded.tobytes() if ok else None

    def clearance_ready(self):
        if self.clearance_recovery is not None:
            try:
                next(self.clearance_recovery)
            except StopIteration as done:
                self.clearance_recovery = None
                self.command()
                if not done.value:
                    raise Unmet('净空不足，已检查近期车迹和前方空间，未能安全移至宽处')
                self.motion_revision += 1
                self.clearance_stamp = self.stamps['local'], self.stamps['global']
                self.note('已移至宽处，等待地图更新，再继续原步骤并重算路径')
            return False
        mapped = self.mapped_turn_obstacle()
        if self.turn_blocked or mapped:
            self.command()
            detail = (f'位姿 {self.pose}；局部触发={self.turn_blocked}，'
                      f'最近 {self.local_obstacle_distance:.2f} m；全局触发={mapped}，'
                      f'最近格距约 {obstacle_distance(self.grid, self.cell(), self.resolution, 3.):.2f} m')
            if self.clearance_hold is None:
                self.clearance_hold = self.now
                self.note('底盘净空保护制动，原地复测。'+detail+
                          '。附全局 OCC 局部截图：蓝点车位，红圈净空，绿线当前路径，黑色障碍。',
                          jpeg=self.clearance_image())
                self.clear_path()
            elif self.now-self.clearance_hold >= 3.:
                if sum(math.dist(self.here, p) < .35 for p in self.recovery_sites) >= 2:
                    raise Unmet('同一位置反复触发净空保护，已尝试换向和重规划。'+detail)
                self.recovery_sites.append(self.here)
                self.clearance_recovery = self.retreat_for_turn()
                self.note('净空持续不足，保留当前任务并检查前后脱困通路。'+detail)
            self.clearance_stamp = (self.stamps['local'], self.stamps['global'])
            return False
        if self.clearance_hold is not None:
            if any(self.stamps[k] <= stamp for k, stamp in zip(('local','global'), self.clearance_stamp)):
                self.command()
                return False
            self.clearance_hold = None
            self.note('局部与全局地图已刷新，净空恢复，继续当前步骤')
        return True

    def tick(self):
        if self.phase == 'idle':
            return
        self.now = self.get_clock().now().nanoseconds*1e-9
        if self.phase == 'reading':
            self.command()
            if not self.pending.done():
                return
            try:
                steps = self.pending.result()
            except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
                self.stop()
                self.note(f'模型解析未完成 {exc}', 'Instruction unavailable.')
                return
            self.pending = None
            self.note('\n'.join([*self.trace, json.dumps(steps, ensure_ascii=False)]))
            self.program = self.execute(steps)
            self.step_index, self.started = 0, self.now
            self.deadline = self.now+720.
            self.phase = 'active'
        if self.pose is None or any(not -.10 <= self.now-t <= 1.2 for t in self.stamps.values()):
            self.command()
            self.clear_path()
            self.report('Waiting for fresh RGB-D map and odometry.')
            return
        try:
            if self.now > self.deadline:
                raise Unmet('该步骤在允许的仿真时间内仍未完成')
            if not self.clearance_ready():
                return
            if self.status == 'Waiting for fresh RGB-D map and odometry.':
                self.report(f'Online step {self.step_index+1}: observations resumed.')
            self.vision_tick()
            next(self.program)
        except StopIteration:
            self.stop(keep_row=True)
            if self.skipped:
                self.note('可执行部分结束，小车停车。已跳过：' + '、'.join(self.skipped),
                          'Instruction partially completed.')
            else:
                self.note('所有已解析步骤均已完成，小车停车', 'Instruction complete.')
        except Unmet as exc:
            self.stop()
            self.note(f'第 {self.step_index+1} 步停止：{exc}',
                      'Stopped: current step unverified.')


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = OnlineNavigator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.vision.close()
        node.worker.shutdown(wait=False, cancel_futures=True)
        if node.journal:
            try:
                node.journal.event('在线探索退出，最后状态 ' + node.status)
            finally:
                node.journal.close()
            print(f'在线探索日志 {node.journal.path}', flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
