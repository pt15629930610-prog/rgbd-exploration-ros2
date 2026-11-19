"""Closed-loop autonomous exploration for the two-room RGB-D simulation."""

import hashlib
import math
from pathlib import Path as FilePath

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from .frontier_planner import (
    choose_path, clear_segment, frontiers, inside, obstacle_distance,
    observation_path, reachable_cells, traversable,
)


def seconds(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        source = FilePath(__file__).resolve()
        fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        self.get_logger().info(f'Explorer source SHA256 {fingerprint} ({source})')
        self.velocity = self.create_publisher(Twist, '/model/mobile_robot/cmd_vel', 1)
        self.route = self.create_publisher(Path, '/exploration/path', 1)
        self.state_pub = self.create_publisher(String, '/exploration/status', 1)
        self.create_subscription(OccupancyGrid, '/exploration/map', self.global_map, 1)
        self.create_subscription(OccupancyGrid, '/rgbd_camera/occ_grid', self.local_map, 1)
        wheel_topic = '/model/mobile_robot/odometry'
        pose_topic = self.declare_parameter('pose_topic', wheel_topic).value
        self.create_subscription(Odometry, pose_topic, self.odometry, 1)
        self.wheel_sample = None
        if pose_topic != wheel_topic:
            self.create_subscription(Odometry, wheel_topic, self.wheel_feedback, 1)
        self.get_logger().info(f'Exploration pose feedback: {pose_topic}')
        self.stamps = dict.fromkeys(('global', 'local', 'odom'), -math.inf)
        self.walk = self.frontier = None
        self.pose = None
        self.path = []
        self.speed_limit = .16
        self.scanned = []
        self.failed = []
        self.retried = False
        self.phase = 'scan'
        self.last_heading = None
        self.sweep = 0.0
        self.scan_start_xy = None
        self.confirming = False
        self.status = ''
        self.timer = self.create_timer(0.1, self.tick)

    def global_map(self, msg):
        if msg.header.frame_id != 'odom':
            return
        info = msg.info
        grid = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
        self.resolution = info.resolution
        self.origin = info.origin.position
        start = self.cell() if self.pose is not None else None
        self.walk = traversable(grid, self.resolution, start)
        self.grid = grid
        self.frontier = frontiers(grid)
        self.stamps['global'] = seconds(msg.header)

    def local_map(self, msg):
        if msg.header.frame_id != 'camera_aligned':
            return
        info = msg.info
        grid = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
        rows, cols = np.where(grid >= 50)
        x = info.origin.position.x + (cols + 0.5) * info.resolution
        y = info.origin.position.y + (rows + 0.5) * info.resolution
        self.turn_blocked = bool(np.any(x*x + y*y < 0.42**2))
        self.forward_blocked = bool(np.any((x < 0.46) & (np.abs(y) < 0.30)))
        self.stamps['local'] = seconds(msg.header)

    def odometry(self, msg):
        if msg.header.frame_id != 'odom' or msg.child_frame_id != 'base_link':
            return
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.pose = (p.position.x, p.position.y, yaw)
        self.stamps['odom'] = seconds(msg.header)
        self.yaw_rate = msg.twist.twist.angular.z

    def wheel_feedback(self, msg):
        self.wheel_sample = seconds(msg.header), msg.twist.twist.angular.z

    def cell(self):
        return (math.floor((self.pose[1] - self.origin.y) / self.resolution),
                math.floor((self.pose[0] - self.origin.x) / self.resolution))

    def xy(self, cell):
        return (self.origin.x + (cell[1] + 0.5) * self.resolution,
                self.origin.y + (cell[0] + 0.5) * self.resolution)

    def set_path(self, path):
        self.path = path
        route = Path()
        route.header.frame_id = 'odom'
        route.header.stamp = self.get_clock().now().to_msg()
        for cell in path:
            point = PoseStamped()
            point.header = route.header
            point.pose.position.x, point.pose.position.y = self.xy(cell)
            point.pose.orientation.w = 1.0
            route.poses.append(point)
        self.route.publish(route)

    def report(self, status):
        if self.status != status:
            self.get_logger().info(status)
            self.state_pub.publish(String(data=status))
            self.status = status

    def command(self, linear=0.0, angular=0.0):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.velocity.publish(msg)

    def begin_scan(self, confirming=False):
        self.set_path([])
        self.phase = 'scan'
        self.confirming = confirming
        self.sweep = 0.0
        self.last_heading = None
        self.scan_start_xy = None

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.phase in ('complete', 'blocked'):
            self.command()
            return
        # Clock and sensor messages arrive on separate bridge topics.
        stale = [name for name, t in self.stamps.items() if not -0.10 <= now - t <= 1.2]
        if stale:
            self.command()
            if self.phase == 'drive':
                self.phase = 'plan'
                self.set_path([])
            self.last_heading = None
            self.report('Waiting for fresh data: ' + ', '.join(stale))
            return

        if self.turn_blocked:
            self.command()
            self.phase = 'blocked'
            self.set_path([])
            self.report('Blocked: obstacle inside rotation clearance.')
        elif self.phase == 'scan':
            self.scan(now)
        elif self.phase == 'settle':
            self.command()
            # Plan only after a new observation taken once the turn has stopped.
            if self.stamps['global'] > self.stopped_at + 0.25:
                self.scanned.append(self.cell())
                self.phase = 'plan'
        elif self.phase == 'plan':
            self.plan(now)
        else:
            self.drive(now)

    def scan(self, now):
        yaw = self.pose[2]
        if self.scan_start_xy is None:
            self.scan_start_xy = self.pose[:2]
            self.max_drift = 0.0
            self.next_scan_log = now + 3.0
        if self.last_heading is None:
            self.progress_time = now
            self.progress_sweep = self.sweep
        else:
            self.sweep += wrap(yaw - self.last_heading)
        self.last_heading = yaw
        self.max_drift = max(self.max_drift, math.dist(self.pose[:2], self.scan_start_xy))
        # The mapper already uses the moving chassis pose. Check clearance at
        # its current position throughout the turn, including the rear side.
        if self.mapped_turn_obstacle():
            self.command()
            self.phase = 'blocked'
            self.report('Blocked: insufficient mapped clearance for rotation.')
            self.log_scan(now)
            return
        if self.sweep >= 2 * math.pi:
            self.command()
            self.phase = 'settle'
            self.stopped_at = now
            self.log_scan(now)
            self.report('Full turn complete. Waiting for updated map to plan.')
            return
        if self.sweep > self.progress_sweep + 0.10:
            self.progress_sweep = self.sweep
            self.progress_time = now
        if now - self.progress_time > 4.0:
            self.command()
            self.phase = 'blocked'
            self.report('Blocked: scan commanded but no angular progress.')
            self.log_scan(now)
            return
        if now >= self.next_scan_log:
            self.log_scan(now)
            self.next_scan_log = now + 3.0
        self.report('Scanning surroundings.')
        self.command(angular=0.38)

    def mapped_turn_obstacle(self):
        # The depth camera only sees ahead. Check mapped obstacles behind and
        # beside the drifting chassis as well.
        row, col = self.cell()
        if not inside(self.grid, (row, col)):
            return True
        radius = math.ceil(0.42 / self.resolution) + 1
        top, left = max(0, row - radius), max(0, col - radius)
        window = self.grid[top:row + radius + 1, left:col + radius + 1]
        rows, cols = np.where(window >= 50)
        dx = self.origin.x + (left + cols + 0.5) * self.resolution - self.pose[0]
        dy = self.origin.y + (top + rows + 0.5) * self.resolution - self.pose[1]
        return bool(np.any(dx*dx + dy*dy < 0.42**2))

    def log_scan(self, now):
        wheel = 'unavailable'
        if self.wheel_sample and abs(now - self.wheel_sample[0]) <= 1.2:
            wheel = f'{self.wheel_sample[1]:+.2f}'
        self.get_logger().info(
            f'Scan: yaw change {math.degrees(self.sweep):+.0f} deg, '
            f'max XY drift {self.max_drift:.2f} m, yaw rate (rad/s): '
            f'pose {self.yaw_rate:+.2f}, wheels {wheel}')

    def plan(self, now):
        self.command()
        start = self.cell()
        if not inside(self.walk, start) or not self.walk[start]:
            # A new depth observation or settling pose can change this cell.
            # Stay stopped and re-evaluate on subsequent planning ticks.
            if not inside(self.walk, start):
                reason = 'outside map bounds'
            elif self.grid[start] != 0:
                reason = f'raw occupancy {self.grid[start]}'
            else:
                gap = obstacle_distance(self.grid, start, self.resolution, 1.0)
                reason = f'free cell in planning margin, obstacle/edge {gap:.2f} m'
            self.report(f'Waiting for traversable start {start}: {reason}.')
            return
        path = choose_path(self.walk, self.frontier, start, self.resolution,
                           self.scanned, self.failed)
        if path:
            self.confirming = False
            self.frontier_goal = path[-1]
            self.start_drive(observation_path(path, self.grid, self.resolution), now)
            self.report('Following A* path to frontier at (%.2f, %.2f) m.' % self.xy(path[-1]))
        elif not self.confirming:
            connected = reachable_cells(self.walk, start)
            self.get_logger().info(
                f'No goal: reachable free area {connected.sum() * self.resolution**2:.2f} m2, '
                f'reachable frontier cells {np.count_nonzero(connected & self.frontier)}. '
                'Starting confirmation scan.')
            self.begin_scan(confirming=True)
        elif self.failed and choose_path(self.walk, self.frontier, start, self.resolution,
                                         self.scanned, []):
            if self.retried:
                self.phase = 'blocked'
                self.report('Blocked: remaining frontiers could not be reached after retry.')
            else:
                self.failed.clear()
                self.retried = True
                self.report('New scan complete; retrying remaining frontiers.')
        else:
            self.phase = 'complete'
            self.report('Complete: no eligible reachable frontier after confirmation scan.')

    def start_drive(self, path, now):
        self.set_path(path)
        self.phase = 'drive'
        self.progress_time = now
        self.progress_index = self.index = 0
        self.best_heading_error = math.inf

    def route_arrived(self):
        self.begin_scan()

    def drive(self, now, check_frontier=True):
        start = self.cell()
        if not inside(self.walk, start) or not self.walk[start]:
            self.command()
            self.set_path([])
            self.phase = 'plan'
            return
        goal = self.path[-1]
        x, y, yaw = self.pose
        # Settle near the planned observation cell before rotating. A large
        # arrival tolerance can leave the chassis on a doorway's clearance rim.
        if math.dist((x, y), self.xy(goal)) < 0.06:
            self.command()
            self.route_arrived()
            return
        # Recheck the remaining route as obstacles appear during motion.
        # A frontier typically recedes as the camera approaches. Retain the
        # observation target while there is still a frontier in its vicinity.
        r = math.ceil(0.5 / self.resolution)
        lost_frontier = False
        if check_frontier:
            gy, gx = self.frontier_goal
            nearby = self.frontier[max(0, gy-r):gy+r+1, max(0, gx-r):gx+r+1]
            lost_frontier = not nearby.any()
        if lost_frontier or not all(self.walk[c] for c in self.path[self.index:]):
            self.command()
            self.set_path([])
            self.phase = 'plan'
            return
        # Only cut across waypoints if the complete shortcut has clearance.
        target = None
        # Reconsider the last few cells when rounding a bend: a previously
        # visible lookahead may become occluded by the inflated corner.
        lookback = math.ceil(0.30 / self.resolution)
        for i in range(max(0, self.index - lookback), len(self.path)):
            if math.dist((x, y), self.xy(self.path[i])) > 0.24:
                if i > self.index:
                    break
                continue
            if clear_segment(self.walk, start, self.path[i]):
                target = i
        if target is None:
            self.retry_goal('no clear lookahead on route')
            return
        self.index = target
        if self.index > self.progress_index:
            self.progress_index = self.index
            self.progress_time = now
            self.best_heading_error = math.inf
        tx, ty = self.xy(self.path[target])
        error = wrap(math.atan2(ty - y, tx - x) - yaw)
        if self.index == self.progress_index and abs(error) < self.best_heading_error - 0.10:
            self.best_heading_error = abs(error)
            self.progress_time = now
        if now - self.progress_time > 7.0:
            self.retry_goal('no tracking progress for 7 s')
            return
        angular = max(-0.55, min(0.55, 1.8 * error))
        # Align before advancing. A slow skid-steer turn can otherwise round
        # a doorway corner into the inflated wall even on a valid A* path.
        if abs(error) > 0.12:
            self.command(angular=angular)
        elif self.forward_blocked:
            self.retry_goal('obstacle in local forward corridor')
        else:
            taper = min(1., math.hypot(tx - x, ty - y) / .16)
            self.command(self.speed_limit * taper, angular)

    def retry_goal(self, reason):
        self.command()
        self.failed.append(self.path[-1])
        self.set_path([])
        self.phase = 'plan'
        self.report(f'Replanning: {reason}.')


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.command()
        node.destroy_node()
        rclpy.shutdown()
