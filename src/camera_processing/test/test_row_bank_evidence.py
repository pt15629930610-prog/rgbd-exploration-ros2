import unittest
from types import SimpleNamespace

from camera_processing.online_skills import MotionSkills


class RowBankEvidenceTests(unittest.TestCase):
    def test_logged_detections_on_same_bank(self):
        gate = (-5.85, -3.15)
        first = SimpleNamespace(stamp=91.6, point=(-3.2743, -1.9967))
        second = SimpleNamespace(stamp=95.6, point=(-3.9323, -2.0793))
        robot = SimpleNamespace(now=96., images=[first, second])
        self.assertIs(MotionSkills.row_bank_match(robot, gate, 0.), second)
        for stamp, point in [(95.6, (-3.93, -4.31)), (91.6, second.point),
                             (95.6, None), (95.6, (-1.2, -2.08))]:
            robot.images = [first, SimpleNamespace(stamp=stamp, point=point)]
            self.assertIsNone(MotionSkills.row_bank_match(robot, gate, 0.))
