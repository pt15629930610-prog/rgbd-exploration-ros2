"""Geometry tests for visual navigation; no world file or model server required."""

import math
import unittest

import numpy as np

from camera_processing.online_vision import locate_door_face, locate_opening, locate_visible


class VisualGeometryTests(unittest.TestCase):
    box = (60, 15, 180, 95)
    pose = (8., -1., math.pi/2)

    def test_parking_uses_face_normal_in_an_oblique_view(self):
        angle = .39
        columns = np.arange(240)
        depth = 2. / (math.cos(angle) + (120.-columns)/140.*math.sin(angle))
        depth = np.tile(depth, (110, 1))
        _, heading = locate_door_face(self.box, depth, 140., 120., self.pose)
        self.assertAlmostEqual(heading, self.pose[2]+angle, places=3)
        self.assertIsNone(locate_door_face(self.box, depth*np.nan, 140., 120., self.pose))

    def test_threshold_and_normal_are_measured_in_the_frame_of_the_camera(self):
        depth = np.full((110, 240), 2., dtype=float)
        depth[:, 60:180] = np.inf
        point, heading = locate_opening(self.box, depth, 140., 120., self.pose)
        np.testing.assert_allclose(point, (8., 1.), atol=.05)
        self.assertAlmostEqual(heading, math.pi/2)
        depth[:, 60:180] = 2.
        self.assertIsNone(locate_opening(self.box, depth, 140., 120., self.pose))

    def test_obstacle_in_gap_and_missing_depth_are_not_a_valid_exit(self):
        for middle in [2., np.nan, 0.]:
            with self.subTest(middle=middle):
                depth = np.full((110, 240), 2., dtype=float)
                depth[:, 60:180] = 4.
                depth[:, 104:136] = middle
                self.assertIsNone(locate_opening(self.box, depth, 140., 120., self.pose))

    def test_openings_narrower_than_planning_clearance_are_rejected(self):
        depth = np.full((110, 240), 2., dtype=float)
        depth[:, 90:150] = 4.
        self.assertIsNone(locate_opening((90,15,150,95), depth, 140., 120., self.pose))

    def test_rgb_without_depth_never_produces_a_landmark(self):
        depth = np.full((110, 240), np.nan)
        self.assertIsNone(locate_visible(self.box, depth, 140., 120., self.pose))
        depth[:, 60:180] = 2.
        np.testing.assert_allclose(locate_visible(self.box, depth, 140., 120., self.pose), (8., 1.), atol=.04)
