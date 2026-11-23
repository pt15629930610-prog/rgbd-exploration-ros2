"""Kinematic replay on the greenhouse SDF with a progressively revealed 60° depth fan."""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from camera_processing.online_roads import RoadView
from camera_processing.online_skills import Feature, MotionSkills, crossed_threshold, wrap
from camera_processing.frontier_planner import obstacle_distance
from camera_processing.online_vision import depth_openings
from test_greenhouse_route import ORIGIN, RESOLUTION, SimulatedRoute, cell


class OnlineReplay(MotionSkills, SimulatedRoute):
    crop_point = (2., 1.56)
    search_turn_slip = .008
    def __init__(self):
        super().__init__()
        self.course = self.entrance[2]
        self.now = 0.
        self.path = []
        self.stamps = {'global': 0.}
        self.responses = 0
        self.feature = self.entered = self.loop_trace = self.vision_key = self.portal = None
        self.vision = SimpleNamespace(reset=lambda: None)
        self.events = []

    def note(self, message):
        self.events.append((message, self.here))

    def camera(self):
        self.see(self.now)
        self.view = RoadView(self.grid, self.walk, ORIGIN, RESOLUTION)
        if self.vision_key:
            # Deterministic visual fixture. Only return a crop if its near canopy
            # is within this exposure's field of view; no map landmarks reach the skill.
            exit = self.vision_key[0] == 'exit'
            point = (0., -11.) if exit else self.crop_point
            angle = math.atan2(point[1]-self.pose[1], point[0]-self.pose[0])
            self.responses += 1
            if abs(wrap(angle-self.pose[2])) < math.pi/6 and math.dist(point, self.here) < 4.9:
                if exit and any(self.truth[cell(xy)] for xy in np.linspace(self.here, point, 120)):
                    return
                self.images.append(SimpleNamespace(stamp=self.now, point=point, pose=self.pose,
                                                   heading=-math.pi/2 if exit else None))

    def exercise(self, step, next_step=None):
        self.vision_key = None  # Production stops the previous vision job between skills.
        self.camera()
        skill = self.run_skill(step, next_step)
        for i in range(2200):
            try:
                next(skill)
            except StopIteration:
                return
            x,y,yaw = self.pose
            dt = .18
            slip = self.search_turn_slip if self.vision_key and self.vision_key[0] == 'exit' else .008
            travel = dt*(self.linear+slip*abs(self.angular))
            self.pose = (x+travel*math.cos(yaw), y+travel*math.sin(yaw), wrap(yaw+dt*self.angular))
            if self.portal and crossed_threshold(self.portal, (x, y), self.here):
                self.crossed = True
            if self.loop_trace:
                self.loop_trace.observe(self.here)
            self.now += dt
            if i % 2 == 0:
                self.camera()
            if self.truth[cell(self.here)] == 100:
                raise AssertionError(('kinematic collision', self.pose, step))
        raise AssertionError(('timed out', self.pose, step, self.events))


class DepthExitReplay(OnlineReplay):
    """Require a two-jamb depth fit for exit observations; model class is a fixture."""

    search_turn_slip = .08

    def camera(self):
        checking_exit = self.portal is not None or self.vision_key and self.vision_key[0] == 'exit'
        if checking_exit and obstacle_distance(self.truth, self.cell(), RESOLUTION, .9) < .42:
            raise AssertionError(('chassis clearance', self.pose))
        if not self.vision_key or self.vision_key[0] != 'exit':
            return super().camera()
        self.see(self.now)
        self.view = RoadView(self.grid, self.walk, ORIGIN, RESOLUTION)
        if self.now-getattr(self, 'last_exposure', -10.) < 1.15:
            return
        self.last_exposure = self.now
        self.responses += 1
        x, y, yaw = self.pose
        width = 200
        fx = width*.5/math.tan(math.pi/6)
        lateral = ((width*.5-np.arange(width))/fx)[:, None]
        ranges = np.arange(.05, 5., .04)[None, :]
        xx = x + ranges*(math.cos(yaw)-lateral*math.sin(yaw))
        yy = y + ranges*(math.sin(yaw)+lateral*math.cos(yaw))
        cols = np.floor((xx-ORIGIN[0])/RESOLUTION).astype(int).clip(0, self.grid.shape[1]-1)
        rows = np.floor((yy-ORIGIN[1])/RESOLUTION).astype(int).clip(0, self.grid.shape[0]-1)
        first_hit = np.where(self.truth[rows, cols] == 100, ranges, np.inf).min(axis=1)
        depth = np.tile(first_hit, (96, 1))
        for _, (point, heading) in depth_openings(depth, fx, width*.5, self.pose):
            if math.dist(point, (0., -11.)) < .4:
                self.images.append(SimpleNamespace(stamp=self.now, point=point, heading=heading, pose=self.pose))


class OnlineRouteReplayTest(unittest.TestCase):
    def test_tight_exit_turn_requires_an_observed_retreat(self):
        robot = OnlineReplay()
        robot.pose = (0., 0., 0.)
        robot.grid[:] = 0
        robot.grid[:, cell((.65, 0.))[1]] = 100
        robot.walk = robot.grid == 0
        robot.view = RoadView(robot.grid, robot.walk, ORIGIN, RESOLUTION)
        robot.vision_start('exit')
        self.assertTrue(robot.tight_exit_turn(math.pi))
        robot.approach_track.append((-.4, 0.))
        robot.walk[cell((-.2, 0.))] = False
        with self.assertRaises(StopIteration) as stopped:
            next(robot.retreat_for_turn())
        self.assertFalse(stopped.exception.value)
        self.assertEqual(robot.linear, 0.)
        robot.walk[cell((-.2, 0.))] = True
        retreat = robot.retreat_for_turn()
        next(retreat)
        self.assertLess(robot.linear, 0.)
        robot.pose = (-.37, 0., 0.)
        with self.assertRaises(StopIteration) as finished:
            next(retreat)
        self.assertTrue(finished.exception.value)
        self.assertEqual(robot.linear, 0.)

    def test_first_row_patrol_returns_from_observed_dead_end_then_exits(self):
        robot = DepthExitReplay()
        robot.speed_limit = .3
        robot.crop_point = (-3.8, 1.9)
        robot.exercise({'do': 'turn', 'side': 'left'})
        robot.exercise({'do': 'seek', 'target': 'row', 'ordinal': 1})
        gate = robot.here
        self.assertLess(math.dist(gate, (-6., 3.11)), .5)
        robot.exercise({'do': 'enter'})
        robot.exercise({'do': 'patrol'})
        self.assertLess(math.dist(robot.here, gate), .3)
        self.assertTrue(any('完成死胡同巡检' in m for m, _ in robot.events))
        robot.exercise({'do': 'seek', 'target': 'door', 'side': 'right'}, next_step='exit')
        robot.exercise({'do': 'exit'})
        self.assertTrue(robot.crossed)

    def test_first_cross_moves_with_unseen_aisle(self):
        for at in (-1.5, 1.8):
            with self.subTest(at=at):
                robot = OnlineReplay()
                rows, cols = np.indices(robot.truth.shape)
                x = ORIGIN[0] + (cols+.5)*RESOLUTION
                y = ORIGIN[1] + (rows+.5)*RESOLUTION
                robot.truth[:] = np.where((abs(y) < .93) | (abs(x-at) < .93), 0, 100)
                self.assertTrue(np.all(robot.grid == -1))
                robot.exercise({'do': 'seek', 'target': 'crossroad'})
                self.assertLess(math.dist(robot.here, (at, 0.)), .35)

    def test_resumed_patrol_counts_only_finished_trips(self):
        robot = OnlineReplay()
        robot.speed_limit = .3
        robot.crop_point = (-3.8, 1.9)
        robot.exercise({'do':'turn','side':'left'})
        robot.exercise({'do':'seek','target':'row','ordinal':1})
        robot.exercise({'do':'enter'})
        robot.exercise({'do':'patrol'})
        robot.feature = None
        robot.exercise({'do':'resume_row'})
        robot.exercise({'do':'patrol','count':2,'total':True})
        self.assertEqual(robot.row_laps, 2)
        self.assertLess(math.dist(robot.here, robot.last_row.point), .3)
        robot.exercise({'do':'patrol','count':2,'total':True})
        self.assertEqual(robot.row_laps, 2)

    def test_immediate_left_turn_at_entrance(self):
        robot = OnlineReplay()
        robot.exercise({'do': 'turn', 'side': 'left'})
        self.assertLess(math.dist(robot.here, robot.entrance[:2]), .2)
        self.assertLess(abs(wrap(robot.pose[2]-math.pi/2)), .1)

    def test_crossing_uses_observed_gate_normal_and_clear_span(self):
        portal = Feature('exit', (5., -3.), math.pi/2)
        self.assertTrue(crossed_threshold(portal, (5.1, -3.1), (5.1, -2.9)))
        self.assertFalse(crossed_threshold(portal, (5.1, -2.9), (5.1, -3.1)))
        self.assertFalse(crossed_threshold(portal, (5.7, -3.1), (5.7, -2.9)))
        self.assertFalse(crossed_threshold(portal, (5., -4.), (5., -2.9)))

    def test_left_turn_observed_row_and_circuit_in_progressively_observed_world(self):
        robot = OnlineReplay()
        robot.exercise({'do':'seek', 'target':'crossroad'})
        self.assertLess(math.dist(robot.here, (0., 0.)), .55)
        robot.exercise({'do':'turn','side':'left'})
        robot.exercise({'do':'seek','target':'row'})
        self.assertLess(math.dist(robot.here, (0., 3.11)), .6)
        robot.exercise({'do':'enter'})
        robot.exercise({'do':'loop'})
        self.assertLess(math.dist(robot.here, (0., 3.11)), .6)
        robot.exercise({'do':'seek', 'target':'door', 'side':'right'}, next_step='exit')
        robot.exercise({'do':'exit'})
        self.assertTrue(robot.crossed)
        self.assertLess(robot.here[1], -11.55)
