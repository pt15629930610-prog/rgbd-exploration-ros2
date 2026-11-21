"""ROS adapter for language-directed navigation on an initially unknown occupancy map."""

from concurrent.futures import ThreadPoolExecutor
import json
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from .door_log_sync import windows_log_server
from .door_run_log import DoorRunLog, RUN_LOG_DIRECTORY
from .frontier_explorer import FrontierExplorer, seconds
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
        self.vision_key = None
        self.responses = 0
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

    def odometry(self, msg):
        if msg.header.frame_id != 'odom' or msg.child_frame_id != 'base_link':
            return
        stamp = seconds(msg.header)
        previous = self.pose
        if stamp < self.stamps['odom']-.5:
            self.stop()
            self.entrance = self.feature = None
            self.entered = self.loop_trace = None
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
            self.feature = None
            self.note('位姿出现跳变，任务已停止，先核对定位', 'Stopped: odometry jumped.')
            return
        if self.portal and previous is not None and crossed_threshold(self.portal, previous, self.here):
            self.crossed = True
        if self.loop_trace is not None:
            self.loop_trace.observe(self.here)

    def stop(self, *, keep_row=False):
        self.command()
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
            self.started = self.now
            self.deadline = self.now + max(720., step.get('seconds', 0.)+30.)
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
            if step['do'] not in ('enter', 'loop', 'wait', 'photo', 'inspect'):
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
            if evidence.point is not None or self.now-self.last_image_log > 3.:
                self.note(f'帧 {evidence.stamp:.2f} {evidence.detail}', jpeg=evidence.jpeg)
                self.last_image_log = self.now

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
            if self.turn_blocked or self.mapped_turn_obstacle():
                raise Unmet('底盘净空内有障碍')
            if self.now > self.deadline:
                raise Unmet('该步骤在允许的仿真时间内仍未完成')
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
