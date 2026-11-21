"""Kinematic replay on the greenhouse SDF with a progressively revealed 60° depth fan."""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from camera_processing.online_roads import RoadView
from camera_processing.online_skills import Feature, MotionSkills, crossed_threshold, wrap
from test_greenhouse_route import ORIGIN, RESOLUTION, SimulatedRoute, cell


class OnlineReplay(MotionSkills, SimulatedRoute):
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
            point = (0., -11.) if exit else (2.0, 1.56)
            angle = math.atan2(point[1]-self.pose[1], point[0]-self.pose[0])
            self.responses += 1
            if abs(wrap(angle-self.pose[2])) < math.pi/6 and math.dist(point, self.here) < 4.9:
                if exit and any(self.truth[cell(xy)] for xy in np.linspace(self.here, point, 120)):
                    return
                self.images.append(SimpleNamespace(stamp=self.now, point=point, pose=self.pose,
                                                   heading=-math.pi/2 if exit else None))

    def exercise(self, step, next_step=None):
        self.camera()
        skill = self.run_skill(step, next_step)
        for i in range(2200):
            try:
                next(skill)
            except StopIteration:
                return
            x,y,yaw = self.pose
            dt = .18
            travel = dt*(self.linear+.008*abs(self.angular))
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


class OnlineRouteReplayTest(unittest.TestCase):
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
