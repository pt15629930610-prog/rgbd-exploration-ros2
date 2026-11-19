"""Decision replay for arrival, cancellation and occlusion without a running simulator."""

from concurrent.futures import Future
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from camera_processing.door_navigator import DoorNavigator
from camera_processing.frontier_explorer import FrontierExplorer


class DoorDecisions(unittest.TestCase):
    def test_forward_command_obeys_speed_limit_and_obstacle_stop(self):
        node = SimpleNamespace(walk=np.ones((3, 10), dtype=bool), cell=lambda: (1, 0),
                               xy=lambda c: (c[1]*.05, c[0]*.05), pose=(0., .05, 0.),
                               path=[(1, c) for c in range(8)], resolution=.05, index=0,
                               progress_index=0, progress_time=0., speed_limit=.08,
                               command=Mock(), forward_blocked=False, retry_goal=Mock())
        FrontierExplorer.drive(node, 1., check_frontier=False)
        node.command.assert_called_with(.08, 0.)
        node.forward_blocked = True
        FrontierExplorer.drive(node, 1.1, check_frontier=False)
        node.retry_goal.assert_called_with('obstacle in local forward corridor')

    def test_speed_change_preserves_task_and_pending_observation(self):
        node = self.state()
        DoorNavigator.new_task(node, SimpleNamespace(data='速度改为0.10米/秒'))
        self.assertEqual(node.speed_limit, .1)
        self.assertEqual((node.generation, node.phase), (2, 'vision'))
        self.assertIsNotNone(node.landmark)
        node.set_path.assert_not_called()

    def test_only_consecutive_approach_failures_stop_task(self):
        node = Mock(spec=DoorNavigator)
        node.route_failures = 2
        node.failed = []
        node.parking_route = False
        node.path = [(10, 12), (10, 13)]
        DoorNavigator.retry_goal(node, 'no clear lookahead on route')
        self.assertEqual(node.route_failures, 3)
        DoorNavigator.route_arrived(node)
        self.assertEqual(node.route_failures, 0)
        for i in range(4):
            DoorNavigator.retry_goal(node, 'obstacle in local forward corridor')
        node.halt.assert_called_once()
        self.assertIn('local forward corridor', node.halt.call_args.args[0])

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
