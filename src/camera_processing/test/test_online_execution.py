"""Exercise the ROS control tick independently of simulator and network timing."""

import unittest
from unittest.mock import Mock

import numpy as np
from camera_processing.online_navigator import OnlineNavigator
from camera_processing.online_skills import Unmet


class ExecutionTests(unittest.TestCase):
    def test_clearance_stops_immediately_and_requires_both_fresh_maps(self):
        node = self.robot()
        node.pose = (0., 0., 0.)
        node.cell.return_value = (15, 15)
        node.grid = np.zeros((31, 31), np.int8)
        node.resolution = .05
        node.local_obstacle_distance = .39
        node.mapped_turn_obstacle.return_value = False
        node.turn_blocked = True
        node.clearance_hold = None
        node.stamps = {'local': 12., 'global': 12.}
        self.assertFalse(OnlineNavigator.clearance_ready(node))
        node.command.assert_called_once_with()
        node.turn_blocked = False
        node.stamps['local'] = 12.4
        self.assertFalse(OnlineNavigator.clearance_ready(node))
        node.stamps['global'] = 12.5
        self.assertTrue(OnlineNavigator.clearance_ready(node))
        node.mapped_turn_obstacle.return_value = True
        self.assertFalse(OnlineNavigator.clearance_ready(node))
        node.now += 3.1
        with self.assertRaisesRegex(Unmet, '全局触发=True'):
            OnlineNavigator.clearance_ready(node)

    def robot(self):
        node = Mock(spec=OnlineNavigator)
        node.here = (0., 0.)
        node.now = 12.
        node.vision = Mock()
        node.speed_limit = .16
        node.skipped = []
        node.run_skill.side_effect = lambda step, following: iter(())
        return node

    def test_independent_unsupported_action_does_not_drop_home_or_speed(self):
        node = self.robot()
        route = [{'do': 'advance', 'speed': .12}, {'do': 'unsupported', 'reason': '唱歌', 'dependent': False},
                 {'do': 'home'}]
        list(OnlineNavigator.execute(node, route))
        self.assertEqual([c.args[0]['do'] for c in node.run_skill.call_args_list], ['advance', 'home'])
        self.assertEqual(node.skipped, ['唱歌'])
        self.assertEqual(node.speed_limit, .12)

    def test_unverified_condition_stops_before_dependent_motion(self):
        node = self.robot()
        node.run_skill.side_effect = Unmet('无法判断是否有虫')
        with self.assertRaises(Unmet):
            list(OnlineNavigator.execute(node, [{'do': 'unsupported', 'dependent': True}, {'do': 'exit'}]))
        node.run_skill.assert_called_once()

    def test_stale_data_brakes_without_advancing_program(self):
        node = self.robot()
        node.phase = 'active'
        node.get_clock = Mock()
        node.get_clock.return_value.now.return_value.nanoseconds = 12_000_000_000
        node.pose = (0., 0., 0.)
        node.stamps = {'global': 12., 'odom': 12., 'local': 9.}
        OnlineNavigator.tick(node)
        node.command.assert_called_once_with()
        node.clear_path.assert_called_once()
        node.vision_tick.assert_not_called()

    def test_cancellation_invalidates_pending_plan_and_visual_request(self):
        node = self.robot()
        pending = node.pending = Mock()
        node.program = iter((1, 2))
        OnlineNavigator.stop(node)
        self.assertEqual(node.phase, 'idle')
        self.assertIsNone(node.program)
        self.assertIsNone(node.vision_key)
        self.assertIsNone(node.portal)
        pending.cancel.assert_called_once()
        node.vision.reset.assert_called_once()
        node.command.assert_called_once_with()
