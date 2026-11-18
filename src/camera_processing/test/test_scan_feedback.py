"""Scan completion must follow chassis motion, including at the yaw wrap."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from camera_processing.frontier_explorer import FrontierExplorer


class ScanFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.scan = SimpleNamespace(
            pose=(0.0, 0.0, 0.0), phase='scan', last_heading=None,
            scan_start_xy=None, sweep=0.0, wheel_sample=(0.0, 0.38),
            command=Mock(), report=Mock(), log_scan=Mock(),
        )
        FrontierExplorer.scan(self.scan, 0.0)

    def test_spinning_wheels_stationary_chassis(self):
        self.scan.wheel_sample = (4.1, 0.38)
        FrontierExplorer.scan(self.scan, 4.1)
        self.assertEqual(self.scan.phase, 'blocked')
        self.assertEqual(self.scan.sweep, 0.0)
        self.scan.command.assert_called_with()

    def test_translation_during_turn(self):
        self.scan.pose = (0.18, 0.12, 0.1)
        FrontierExplorer.scan(self.scan, 1.0)
        self.assertEqual(self.scan.phase, 'blocked')
        self.scan.command.assert_called_with()

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
