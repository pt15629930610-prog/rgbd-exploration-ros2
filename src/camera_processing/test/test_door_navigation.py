"""Decision replay for arrival, cancellation and occlusion without a running simulator."""

from concurrent.futures import Future
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from camera_processing.door_navigator import DoorNavigator, observe
from camera_processing.frontier_explorer import FrontierExplorer


class DoorDecisions(unittest.TestCase):
    def test_busy_worker_skips_intermediate_frames_then_takes_newest_once(self):
        node = Mock(spec=DoorNavigator)
        node.pending = Future()
        node.intrinsics = (55., 32.)
        node.poses = [(1.0, (0., 0., 0.))]
        node.last_frame_stamp = .8
        node.phase, node.view_number, node.generation = 'drive', 1, 1
        node.direction = 0.
        node.server, node.vision_worker, node.frame_pub = '', Mock(), Mock()
        node.bridge = Mock()
        node.pair = (.9, Mock(), Mock())
        DoorNavigator.capture(node, 1., moving=True)
        node.vision_worker.submit.assert_not_called()
        rgb, depth = Mock(), SimpleNamespace(encoding='32FC1')
        node.pair = (1., rgb, depth)
        node.bridge.imgmsg_to_cv2.side_effect = [
            np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32)]
        node.pending = None
        DoorNavigator.capture(node, 1., moving=True)
        node.frame_pub.publish.assert_called_once_with(rgb)
        node.command.assert_not_called()
        self.assertEqual(node.phase, 'drive')
        node.pending = None
        DoorNavigator.capture(node, 1.1, moving=True)
        node.vision_worker.submit.assert_called_once()

    def test_drives_while_visual_inference_is_pending(self):
        node = Mock(spec=DoorNavigator)
        node.get_clock = Mock()
        node.get_clock.return_value.now.return_value.nanoseconds = 1_000_000_000
        node.phase, node.entrance, node.stamps = 'drive', (0, 0, 0), {'odom': 1.}
        node.turn_blocked = node.parking_route = False
        node.mapped_turn_obstacle.return_value = False
        node.pending, node.pair = Future(), (1., Mock(), Mock())
        DoorNavigator.tick(node)
        node.drive.assert_called_once_with(1., check_frontier=False)
        node.command.assert_not_called()

    @patch('camera_processing.door_navigator.query_image')
    def test_records_both_model_answers_and_invalid_depth(self, query):
        query.side_effect = ['看见', '[200,100,800,900]']
        point, detail = observe('', b'jpeg',
                                np.full((48, 64), np.nan), 55., 32., (0., 0., 0.), 0.)
        self.assertIsNone(point)
        for part in ('看见', '[200,100,800,900]', '0/45', '无法定位'):
            self.assertIn(part, detail)

    def test_in_motion_negative_keeps_route_and_stall_timer(self):
        node = self.state()
        node.moving_view, node.phase, node.progress_time = True, 'drive', 1.
        node.pending.set_result((None, '未见'))
        DoorNavigator.finish_view(node)
        self.assertEqual(node.phase, 'drive')
        self.assertEqual(node.progress_time, 1.)
        node.next_view.assert_not_called()
        node.set_path.assert_not_called()

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
                               verifying=True, moving_view=False, landmark=np.array([0., 5.]),
                               next_view=Mock(), phase='vision', log_view=Mock())
        node.near_parking_point = lambda point: DoorNavigator.near_parking_point(node, point)
        node.halt = lambda reason: DoorNavigator.halt(node, reason)
        return node

    def test_distance_and_lateral_error_then_visual_confirmation(self):
        node = self.state()
        node.pending.set_result((np.array([.03, 5.02]), 'accepted depth'))
        DoorNavigator.finish_view(node)
        self.assertEqual(node.phase, 'idle')
        node.command.assert_called_once_with()
        self.assertIn('Arrived', node.report.call_args.args[0])
        node.pose = (1., 4., math.pi/2)
        self.assertFalse(node.near_parking_point(node.landmark))

    def test_no_confirmation_when_hidden_or_cancelled(self):
        node = self.state()
        node.pending.set_result((None, '未见'))
        DoorNavigator.finish_view(node)
        node.next_view.assert_called_once()
        node.report.assert_not_called()
        DoorNavigator.new_task(node, SimpleNamespace(data='停车'))
        self.assertEqual(node.phase, 'idle')
        self.assertIsNone(node.landmark)
        DoorNavigator.finish_view(node)
        self.assertEqual(node.phase, 'idle')
        node.next_view.assert_called_once()

    def test_new_door_on_search_route_switches_to_parking_plan(self):
        node = self.state()
        node.moving_view = True
        node.pose = (0., 4.3, math.pi/2)
        node.landmark = None
        node.verifying = False
        node.pending.set_result((np.array([0., 5.]), '看见'))
        DoorNavigator.finish_view(node)
        self.assertEqual(node.phase, 'plan')
        node.command.assert_called_once_with()
        node.set_path.assert_called_once_with([])
        node.next_view.assert_not_called()
