import math
import unittest
from types import SimpleNamespace

from camera_processing.online_skills import MotionSkills, RouteHistory, Unmet


class ReturnDrive(MotionSkills):
    """A door behind the robot; global planning reconnects only on the inside."""

    def __init__(self, blocked=False):
        self.route_history = RouteHistory()
        for i in range(21):
            self.route_history.observe((i/10, 0.))
        self.route_history.checkpoint()
        self.route_history.begin()
        self.pose = (2., 0., 0.)
        self.path = []
        self.now = 0.
        self.stamps = {'global': 0}
        self.forward_blocked = False
        self.speed_limit = .2
        self.resolution = .05
        self.velocity = 0.
        self.messages = []
        self.view = SimpleNamespace(path_to=self.plan, cell=lambda p:p,
                                    straight_clear=lambda *args: not blocked)

    def plan(self, start, goal):
        return [goal] if start[0] < 1.13 and goal == (0., 0.) else []

    def note(self, message):
        self.messages.append(message)

    def look(self, yaw):
        yield

    def command(self, linear=0., angular=0.):
        self.velocity = linear

    def drive_path(self, path, **kwargs):
        if path:
            self.pose = (*path[-1], 0.)
        yield
        return bool(path)

    def cell(self):
        return self.here

    def run_home(self):
        program = self.travel((0., 0.))
        for _ in range(1000):
            try:
                next(program)
            except StopIteration:
                return
            self.now += .2
            self.stamps['global'] += 1
            self.pose = (self.pose[0]+.2*self.velocity, 0., 0.)
            self.route_history.observe(self.here)
        raise AssertionError('Recovery did not terminate')


class RecoveryTests(unittest.TestCase):
    def test_midpoint_is_on_measured_bend(self):
        history = RouteHistory()
        for point in [(0,0),(1,0),(1,1)]:
            history.observe(point)
        history.checkpoint()
        self.assertEqual(history.candidates((1,1))[0][1], (1,0))

    def test_reverse_to_previous_stage_then_replan_home(self):
        drive = ReturnDrive()
        drive.run_home()
        self.assertLess(math.dist(drive.here, (0.,0.)), .2)
        self.assertTrue(any('已回到节点' in msg for msg in drive.messages))

    def test_obstructed_return_trail_does_not_move(self):
        drive = ReturnDrive(blocked=True)
        with self.assertRaises(Unmet):
            drive.run_home()
        self.assertEqual(drive.here, (2., 0.))
