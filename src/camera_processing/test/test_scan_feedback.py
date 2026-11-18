"""Scan completion must follow chassis motion, including at the yaw wrap."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from camera_processing.frontier_explorer import FrontierExplorer


class ScanFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.scan = SimpleNamespace(
            pose=(0.0, 0.0, 0.0), phase='scan', last_heading=None,
            scan_start_xy=None, sweep=0.0, wheel_sample=(0.0, 0.38),
            command=Mock(), report=Mock(), log_scan=Mock(),
            mapped_turn_obstacle=Mock(return_value=False),
        )
        FrontierExplorer.scan(self.scan, 0.0)

    def test_spinning_wheels_stationary_chassis(self):
        self.scan.wheel_sample = (4.1, 0.38)
        FrontierExplorer.scan(self.scan, 4.1)
        self.assertEqual(self.scan.phase, 'blocked')
        self.assertEqual(self.scan.sweep, 0.0)
        self.scan.command.assert_called_with()

    def test_rocking_across_scan_origin_at_101_degrees(self):
        # Only the radial maximum is logged. These synthetic XY positions
        # reproduce both a 0.15 m maximum and a >0.20 m segment displacement.
        for angle in range(1, 362):
            if angle <= 91:
                x = -0.12 * angle / 91
            else:
                x = -0.12 + 0.27 * min(1, (angle - 91) / 10)
            yaw = math.radians(angle)
            self.scan.pose = (x, 0., math.atan2(math.sin(yaw), math.cos(yaw)))
            FrontierExplorer.scan(self.scan, angle / 8.)
            if angle == 101:
                self.assertAlmostEqual(self.scan.max_drift, .15)
            if self.scan.phase == 'settle':
                break
            self.assertEqual(self.scan.phase, 'scan')
        self.assertEqual(self.scan.phase, 'settle')

    def test_actual_turn_enters_settling_phase(self):
        for i in range(1, 65):
            heading = i * 0.1
            self.scan.pose = (0.0, 0.0, math.atan2(math.sin(heading), math.cos(heading)))
            FrontierExplorer.scan(self.scan, i * 0.25)
            if self.scan.phase == 'settle':
                break
        self.assertEqual(self.scan.phase, 'settle')
        self.assertGreaterEqual(self.scan.sweep, math.tau)
        self.scan.command.assert_called_with()

    def test_reported_drift_at_173_degrees_then_full_turn_and_plan(self):
        # The log gives maximum radial drift, not the XY trajectory. Use one
        # trajectory consistent with those readings, then continue past 360.
        readings = [(0, 0), (20, .12), (48, .12), (72, .12), (97, .15),
                    (118, .15), (149, .15), (170, .15), (173, .21),
                    (210, .23), (270, .24), (330, .22), (361, .21)]
        for (a0, x0), (a1, x1) in zip(readings, readings[1:]):
            for angle in range(a0 + 1, a1 + 1):
                x = x0 + (x1 - x0) * (angle - a0) / (a1 - a0)
                yaw = math.radians(angle)
                self.scan.pose = (x, 0.0, math.atan2(math.sin(yaw), math.cos(yaw)))
                FrontierExplorer.scan(self.scan, angle / 8)
                if self.scan.phase == 'settle':
                    break
                self.assertEqual(self.scan.phase, 'scan')
        self.assertEqual(self.scan.phase, 'settle')
        self.scan.mapped_turn_obstacle.assert_called()
        now = self.scan.stopped_at + 0.4
        self.scan.get_clock = lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=round(now * 1e9)))
        self.scan.stamps = dict.fromkeys(('global', 'local', 'odom'), now)
        self.scan.turn_blocked = False
        self.scan.scanned = []
        self.scan.cell = lambda: (120, 124)
        FrontierExplorer.tick(self.scan)
        self.assertEqual(self.scan.phase, 'plan')

    def test_gradual_drift_towards_mapped_obstacle(self):
        self.scan.mapped_turn_obstacle.return_value = True
        for i in range(1, 51):
            self.scan.pose = (0.005 * i, 0.0, 0.05 * i)
            FrontierExplorer.scan(self.scan, i * 0.2)
            if self.scan.phase == 'blocked':
                break
        self.assertEqual(self.scan.phase, 'blocked')
        self.scan.command.assert_called_with()

    def test_mapped_clearance_includes_obstacle_behind_robot(self):
        grid = np.zeros((80, 80), dtype=np.int8)
        state = SimpleNamespace(grid=grid, cell=lambda: (20, 40),
                                origin=SimpleNamespace(x=-2.0, y=-1.0),
                                resolution=0.05, pose=(0.025, 0.025, 0.0))
        grid[20, 31] = 100  # 0.45 m behind
        self.assertFalse(FrontierExplorer.mapped_turn_obstacle(state))
        grid[20, 32] = 100  # 0.40 m behind
        self.assertTrue(FrontierExplorer.mapped_turn_obstacle(state))
