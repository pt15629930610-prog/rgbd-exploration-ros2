"""Route binding and sensor-limited closed-loop replay on the actual SDF collision layout."""

from concurrent.futures import Future
import json
import math
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch
from xml.etree import ElementTree as ET

import cv2
import numpy as np

from camera_processing.aisle_tracking import crossed_right_door, observed_path
from camera_processing.frontier_explorer import wrap
from camera_processing.frontier_planner import traversable
from camera_processing.greenhouse_roads import (
    CROSS, ENTRANCE, ENTRANCE_POSE, FRONT, OUTSIDE, RIGHT_SILL, ROWS, compile_route, from_world, to_world,
)
from camera_processing.occ_mapping import OCCMapping
from camera_processing.route_language import extract_step, interpret, literal_action, split_instruction
from camera_processing.route_navigator import RouteNavigator


WORLD = Path(__file__).resolve().parents[1] / 'worlds' / 'greenhouse_routes.sdf'
EXAMPLE = '经过第一个十字路口左拐，然后前面遇到第一条作物行进去走一圈，然后从大棚的右门出'
STEPS = [{'do': 'turn', 'side': 'left', 'at': 1}, {'do': 'loop', 'row': 1, 'toward': 'auto'},
         {'do': 'exit', 'side': 'right'}]
ORIGIN = (-8., -14.)
RESOLUTION = .05


def cell(point):
    return (math.floor((point[1]-ORIGIN[1])/RESOLUTION),
            math.floor((point[0]-ORIGIN[0])/RESOLUTION))


def collision_grid():
    grid = np.zeros((560, 320), np.int8)
    root = ET.parse(WORLD)
    for m in root.findall('world/model'):
        if m.findtext('static') != 'true':
            continue
        x, y, z, _, _, yaw = map(float, m.findtext('pose', '0 0 0 0 0 0').split())
        for collider in m.findall('link/collision'):
            size = collider.findtext('geometry/box/size')
            if size is None:
                continue
            length, width, height = np.fromstring(size, sep=' ')/2
            a, b, h, _, _, angle = map(float, collider.findtext('pose', '0 0 0 0 0 0').split())
            if abs(z+h-.25) > height:
                continue
            cx, cy = to_world((a, b), (x, y, yaw))
            c, s = abs(math.cos(yaw+angle)), abs(math.sin(yaw+angle))
            hx, hy = length*c+width*s, length*s+width*c
            lo, hi = cell((cx-hx, cy-hy)), cell((cx+hx, cy+hy))
            grid[lo[0]:hi[0]+1, lo[1]:hi[1]+1] = 100
    return grid


class SimulatedRoute:
    """Feed the production follower measured poses and a gradually revealed 60-degree fan."""

    activate = RouteNavigator.activate
    begin_leg = RouteNavigator.begin_leg
    describe_leg = RouteNavigator.describe_leg
    follow = RouteNavigator.follow
    look_for_road = RouteNavigator.look_for_road

    def __init__(self):
        self.pose = self.entrance = (-6., 0., 0.)
        self.resolution = RESOLUTION
        self.origin = types.SimpleNamespace(x=ORIGIN[0], y=ORIGIN[1])
        self.steps = STEPS
        self.speed_limit = .16
        self.forward_blocked = False
        self.events = []
        self.mission_pub = None
        self.truth = collision_grid()
        self.grid = np.full_like(self.truth, -1)
        self.observed = np.zeros_like(self.truth, dtype=np.uint8)
        self.walk = self.grid == 0
        angles = np.linspace(-math.pi/6, math.pi/6, 141)[:, None]
        ranges = np.arange(.025, 5., .035)
        self.ray_x, self.ray_y = ranges*np.cos(angles), ranges*np.sin(angles)
        r, c = np.ogrid[-11:12, -11:12]
        self.disk = (r*r+c*c <= 11**2).astype(np.uint8)
        self.command()

    def see(self, now):
        x, y, a = self.pose
        c, s = math.cos(a), math.sin(a)
        cols = np.floor((x+self.ray_x*c-self.ray_y*s-ORIGIN[0])/RESOLUTION).astype(int)
        rows = np.floor((y+self.ray_x*s+self.ray_y*c-ORIGIN[1])/RESOLUTION).astype(int)
        rows = rows.clip(0, self.grid.shape[0]-1)
        cols = cols.clip(0, self.grid.shape[1]-1)
        hit = self.truth[rows, cols] == 100
        occluded = np.maximum.accumulate(hit, axis=1)
        seen = ~np.column_stack((np.zeros(len(hit), bool), occluded[:, :-1]))
        rr, cc = rows[seen], cols[seen]
        self.grid[rr, cc] = self.truth[rr, cc]
        self.observed[rr, cc] = hit[seen]
        cv2.circle(self.grid, self.cell()[::-1], 3, 0, -1)
        self.walk = (self.grid == 0) & (cv2.dilate(self.observed, self.disk) == 0)
        self.stamps = {'global': now}

    def cell(self):
        return cell(self.pose)

    def xy(self, rc):
        return ORIGIN[0]+(rc[1]+.5)*RESOLUTION, ORIGIN[1]+(rc[0]+.5)*RESOLUTION

    def say(self, status, message):
        self.events.append((status, self.pose))

    def publish_points(self, pub, points):
        self.preview = points

    def set_path(self, path):
        self.path = path

    def command(self, linear=0., angular=0.):
        self.linear, self.angular = linear, angular

    def hold(self, status, message):
        self.command()
        self.phase = 'idle'
        self.say(status, message)

    def run(self):
        self.activate(0.)
        for i in range(4300):
            now = i*.15
            if i % 2 == 0:
                self.see(now)
            self.follow(now)
            if self.phase == 'idle':
                break
            x, y, yaw = self.pose
            # Include a small translation during skid-steer turns.
            travel = (self.linear + .012*abs(self.angular))*.15
            self.pose = (x+travel*math.cos(yaw), y+travel*math.sin(yaw),
                         wrap(yaw+self.angular*.15))
            assert self.truth[cell(self.pose)] == 0, self.pose


class GreenhouseRouteTests(unittest.TestCase):
    def test_instructions_and_natural_rewording(self):
        examples = [(EXAMPLE, {'1': 'A', '2': 'B', '3': 'C'}),
                    ('进棚一直走，到头一个十字口往左，顺路碰到的第一排作物绕一圈，最后从右手边的出口离棚。',
                     {'1': 'A', '2': 'B', '3': 'C'})]
        for phrase, labels in examples:
            with self.subTest(phrase=phrase), patch('camera_processing.route_language.build_opener') as net:
                reply = net.return_value.open.return_value.__enter__.return_value
                reply.read.return_value = json.dumps({'choices': [{'message': {
                    'content': json.dumps(labels)}}]}).encode()
                self.assertEqual(interpret('http://localhost', phrase), STEPS)
                self.assertEqual(net.return_value.open.call_count, 1)

    def test_loop_requires_all_sides_and_returns_to_its_own_gate(self):
        points = [s.point for s in compile_route(STEPS)]
        gate = (6., ROWS[0])
        self.assertEqual(points[:6], [CROSS, gate, (FRONT, ROWS[0]), (FRONT, 0.), CROSS, gate])
        self.assertEqual(points[-2:], [RIGHT_SILL, OUTSIDE])
        for bad in [[dict(STEPS[0], side='right'), STEPS[1]], [STEPS[0], dict(STEPS[1], toward='back')],
                    [dict(STEPS[0], at=2)], [dict(STEPS[2], side='left')], [STEPS[1]]]:
            with self.assertRaises(ValueError):
                compile_route(bad)

    def test_entrance_frame_survives_changed_heading_and_exit_needs_crossing(self):
        frame = (3., 4., math.radians(83))
        np.testing.assert_allclose(from_world(to_world(OUTSIDE, frame), frame), OUTSIDE)
        self.assertTrue(crossed_right_door((6.1,-10.96), (6.1,-11.04), RIGHT_SILL))
        self.assertFalse(crossed_right_door((7.,-10.96), (7.,-11.04), RIGHT_SILL))
        self.assertFalse(crossed_right_door((6.,-10.), OUTSIDE, RIGHT_SILL))

    def test_missing_constraints_and_unsupported_actions_stop_before_driving(self):
        for action, clause in [('A', '看一下左边'), ('B', '看看第一排作物'), ('C', '右侧出口'),
                               ('D', '看看入口'), ('E', '去那边看看'), ('E', '停留七十秒')]:
            with self.subTest(clause=clause), self.assertRaises(ValueError):
                extract_step(action, clause)
        for text in ['不要出右门', '停留0.5秒', '如果有路就往左拐']:
            with self.assertRaises(ValueError):
                split_instruction(text)
        self.assertEqual(split_instruction('停两秒后返回入口'), ['停两秒', '返回入口'])
        with self.assertRaises(ValueError):
            literal_action('绕一圈等待两秒')

    def test_replacement_waits_at_actual_pose_then_returns_to_original_entrance(self):
        pose = (3., .30)
        steps = compile_route([{'do': 'wait', 'seconds': 2}, {'do': 'home'}], pose)
        self.assertEqual((steps[0].point, steps[0].wait), (pose, 2))
        self.assertEqual(steps[-1].point, ENTRANCE)
        steps = compile_route([STEPS[0]], (3., 0.))
        self.assertEqual(steps[0].point, CROSS)
        with self.assertRaises(ValueError):
            compile_route([STEPS[2]], (6., -11.6))

    def test_replacing_or_cancelling_a_pending_instruction_brakes_and_ignores_old_reply(self):
        node = Mock(spec=RouteNavigator)
        node.hold = types.MethodType(RouteNavigator.hold, node)
        node.pending = old = Future()
        old.set_running_or_notify_cancel()
        node.mission_pub, node.server, node.worker = Mock(), 'http://localhost', Mock()
        new = node.worker.submit.return_value = Future()
        RouteNavigator.instruction(node, types.SimpleNamespace(data='等两秒'))
        old.set_result([STEPS[2]])
        self.assertIs(node.pending, new)
        RouteNavigator.instruction(node, types.SimpleNamespace(data='停车'))
        self.assertTrue(new.cancelled())
        self.assertEqual(node.phase, 'idle')
        self.assertIsNone(node.pending)
        self.assertEqual(node.command.call_count, 2)

    def test_open_door_lintel_does_not_create_an_occupied_wall(self):
        cloud = np.array([[2., -.2, .12], [2., .3, .05], [2., 0., 1.24]])
        for limit, count in [(.27, 2), (-1., 3)]:
            node = Mock()
            node.max_obstacle_z = limit
            with patch('camera_processing.occ_mapping.convert_pointcloud', side_effect=[cloud[:0], cloud]):
                OCCMapping.cloud_callback(node, Mock(), Mock())
            np.testing.assert_array_equal(node.latest_obstacles, cloud[:count])

    def test_world_assets_and_all_required_roads_after_inflation(self):
        root = ET.parse(WORLD)
        spawn = root.findtext("world/model[@name='mobile_robot']/pose").split()
        self.assertEqual((float(spawn[0]), float(spawn[1]), float(spawn[5])), ENTRANCE_POSE)
        self.assertEqual(len(root.findall("world/model/link/collision[@name='planting_station']")), 12)
        for uri in root.iter('uri'):
            self.assertTrue((WORLD.parent/uri.text).is_file(), uri.text)
        walk = traversable(collision_grid(), RESOLUTION)
        points = [ENTRANCE] + [s.point for s in compile_route(STEPS)]
        for a, b in zip(points, points[1:]):
            for xy in np.linspace(a, b, 140):
                self.assertTrue(walk[cell(to_world(xy, (-6., 0., 0.)))], xy)
        for x, y in [(-1.14, 3.11), (4.82, -6.21)]:
            self.assertFalse(walk[cell((x,y))])
            self.assertTrue(walk[cell((x-.85, y))])

    def test_sensed_fan_follows_each_leg_and_crosses_into_exterior(self):
        robot = SimulatedRoute()
        robot.run()
        self.assertEqual(robot.events[-1][0], 'Route complete.', (robot.events[-3:], robot.pose))
        self.assertTrue(robot.crossed)
        self.assertLess(robot.pose[1], -12.)
        reached = [pose for status, pose in robot.events if status.startswith('Reached checkpoint')]
        self.assertEqual(len(reached), len(robot.mission))
        for step, pose in zip(robot.mission, reached):
            self.assertLessEqual(math.dist(from_world(pose, robot.entrance), step.point), .22)

    def test_sealed_mandatory_road_never_shortcuts_or_reports_success(self):
        robot = SimulatedRoute()
        row, col = cell((-3., 0.))
        robot.truth[row-24:row+25,col:col+4] = 100
        robot.run()
        self.assertIn('Holding:', robot.events[-1][0])
        self.assertLess(robot.pose[0], -3.4)
        self.assertEqual((robot.linear, robot.angular), (0., 0.))

    def test_no_path_from_unknown_cell_and_stale_input_stops_drive(self):
        grid = np.zeros((40, 40), bool)
        self.assertEqual(observed_path(grid, (-1., -1.), .05, (0., 0.), (0., 0.), (1., 0.)), [])
        node = Mock()
        node.phase, node.entrance, node.path = 'follow', (0.,0.,0.), [(2,2)]
        node.get_clock.return_value.now.return_value.nanoseconds = 3_000_000_000
        node.stamps = {'global': 0., 'odom': 3., 'local': 3.}
        RouteNavigator.tick(node)
        node.command.assert_called_once_with()
        node.follow.assert_not_called()
