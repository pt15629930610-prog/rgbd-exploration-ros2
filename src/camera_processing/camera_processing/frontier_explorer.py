"""Closed-loop autonomous exploration for the two-room RGB-D simulation."""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from .frontier_planner import (
    choose_path, clear_segment, frontiers, inside, traversable,
)


def seconds(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        self.velocity = self.create_publisher(Twist, '/model/mobile_robot/cmd_vel', 1)
        self.route = self.create_publisher(Path, '/exploration/path', 1)
        self.create_subscription(OccupancyGrid, '/exploration/map', self.global_map, 1)
        self.create_subscription(OccupancyGrid, '/rgbd_camera/occ_grid', self.local_map, 1)
        self.create_subscription(Odometry, '/model/mobile_robot/odometry', self.odometry, 1)
        self.stamps = dict.fromkeys(('global', 'local', 'odom'), -math.inf)
        self.walk = self.frontier = None
        self.pose = None
        self.path = []
        self.scanned = []
        self.failed = []
        self.retried = False
        self.phase = 'scan'
        self.last_heading = None
        self.sweep = 0.0
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
        self.walk = traversable(grid, self.resolution)
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

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.phase in ('complete', 'blocked'):
            self.command()
            return
        # Clock and sensor messages arrive on separate bridge topics.
        if not all(-0.10 <= now - t <= 1.2 for t in self.stamps.values()):
            self.command()
            if self.phase == 'drive':
                self.phase = 'plan'
                self.set_path([])
            self.last_heading = None
            self.report('Waiting for fresh global map, local OCC and odometry.')
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
        if self.last_heading is None:
            self.progress_time = now
            self.progress_sweep = self.sweep
        else:
            self.sweep += wrap(yaw - self.last_heading)
        self.last_heading = yaw
        if self.sweep >= 2 * math.pi:
            self.command()
            self.phase = 'settle'
            self.stopped_at = now
            return
        if self.sweep > self.progress_sweep + 0.10:
            self.progress_sweep = self.sweep
            self.progress_time = now
        if now - self.progress_time > 4.0:
            self.command()
            self.phase = 'blocked'
            self.report('Blocked: scan commanded but no angular progress.')
            return
        self.report('Scanning surroundings.')
        self.command(angular=0.38)

    def plan(self, now):
        self.command()
        start = self.cell()
        if not inside(self.walk, start) or not self.walk[start]:
            self.phase = 'blocked'
            self.report('Blocked: robot cell is outside known traversable space.')
            return
        path = choose_path(self.walk, self.frontier, start, self.resolution,
                           self.scanned, self.failed)
        if path:
            self.confirming = False
            self.set_path(path)
            self.phase = 'drive'
            self.progress_time = now
            self.progress_index = self.index = 0
            self.report('Following A* path to frontier at (%.2f, %.2f) m.' % self.xy(path[-1]))
        elif not self.confirming:
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

    def drive(self, now):
        start = self.cell()
        goal = self.path[-1]
        x, y, yaw = self.pose
        if math.dist((x, y), self.xy(goal)) < 0.12:
            self.command()
            self.begin_scan()
            return
        # Recheck the remaining route as obstacles appear during motion.
        # A frontier typically recedes as the camera approaches. Retain the
        # observation target while there is still a frontier in its vicinity.
        r = math.ceil(0.5 / self.resolution)
        gy, gx = goal
        nearby = self.frontier[max(0, gy-r):gy+r+1, max(0, gx-r):gx+r+1]
        if not nearby.any() or not all(self.walk[c] for c in self.path[self.index:]):
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
        if target is None or now - self.progress_time > 7.0:
            self.retry_goal()
            return
        self.index = target
        if self.index > self.progress_index:
            self.progress_index = self.index
            self.progress_time = now
        tx, ty = self.xy(self.path[target])
        error = wrap(math.atan2(ty - y, tx - x) - yaw)
        angular = max(-0.55, min(0.55, 1.8 * error))
        if abs(error) > 0.28:
            self.command(angular=angular)
        elif self.forward_blocked:
            self.retry_goal()
        else:
            self.command(min(0.16, math.hypot(tx - x, ty - y)), angular)

    def retry_goal(self):
        self.command()
        self.failed.append(self.path[-1])
        self.set_path([])
        self.phase = 'plan'
        self.report('Route blocked or stalled; trying another frontier.')


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
