"""Decision replay for arrival, cancellation and occlusion without a running simulator."""

from concurrent.futures import Future
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from camera_processing.door_navigator import DoorNavigator


class DoorDecisions(unittest.TestCase):
    def state(self):
        node = SimpleNamespace(command=Mock(), set_path=Mock(), report=Mock(),
                               generation=2, request_id=2, pending=Future(),
                               pose=(0., 4., math.pi/2), camera_pose=(0., 4., math.pi/2),
                               entrance=(-4., 0., 0.),
                               direction=math.pi/2, task=(math.pi/2, 1.),
                               verifying=True, landmark=np.array([0., 5.]),
                               next_view=Mock(), phase='vision')
        node.near_parking_point = lambda point: DoorNavigator.near_parking_point(node, point)
        node.halt = lambda reason: DoorNavigator.halt(node, reason)
        return node

    def test_distance_and_lateral_error_then_visual_confirmation(self):
        node = self.state()
        node.pending.set_result(np.array([.03, 5.02]))
        DoorNavigator.finish_view(node)
        self.assertEqual(node.phase, 'idle')
        node.command.assert_called_once_with()
        self.assertIn('Arrived', node.report.call_args.args[0])
        node.pose = (1., 4., math.pi/2)
        self.assertFalse(node.near_parking_point(node.landmark))

    def test_no_confirmation_when_hidden_or_cancelled(self):
        node = self.state()
        node.pending.set_result(None)
        DoorNavigator.finish_view(node)
        node.next_view.assert_called_once()
        node.report.assert_not_called()
        DoorNavigator.new_task(node, SimpleNamespace(data='停车'))
        self.assertEqual(node.phase, 'idle')
        self.assertIsNone(node.landmark)
        DoorNavigator.finish_view(node)
        self.assertEqual(node.phase, 'idle')
        node.next_view.assert_called_once()
