"""Regression tests for doorway clearance and exploration goal selection."""

import unittest

import numpy as np

from camera_processing.frontier_planner import (
    astar, choose_path, clear_segment, frontiers, traversable,
)


class PlannerTests(unittest.TestCase):
    def test_only_measured_free_start_can_exit_margin(self):
        grid = np.zeros((80, 80), dtype=np.int8)
        grid[:, 20] = 100
        start = (40, 31)
        baseline = traversable(grid, .05)
        walk = traversable(grid, .05, start)
        self.assertFalse(baseline[start])
        self.assertTrue(walk[start])
        self.assertEqual(np.count_nonzero(baseline != walk), 1)
        self.assertFalse(traversable(grid, .05, (40, 30))[40, 30])
        grid[start] = -1
        self.assertFalse(traversable(grid, .05, start)[start])

    def test_two_rooms_and_doorway(self):
        grid = np.zeros((62, 82), dtype=np.int8)
        grid[[0, -1], :] = 100
        grid[:, [0, -1]] = 100
        grid[:, 36] = 100
        grid[34:48, 36] = 0
        grid[1:-1, 60:-1] = -1
        start = (25, 20)
        walk = traversable(grid, 0.1)
        edge = frontiers(grid)
        path = choose_path(walk, edge, start, 0.1, [start], [])
        self.assertEqual(path[0], start)
        self.assertTrue(edge[path[-1]])
        self.assertTrue(any(x == 36 and 34 <= y < 48 for y, x in path))
        self.assertTrue(all(clear_segment(walk, a, b) for a, b in zip(path, path[1:])))

        grid[:, 36] = 100
        self.assertFalse(choose_path(traversable(grid, 0.1), frontiers(grid),
                                     start, 0.1, [start], []))

    def test_cannot_cut_diagonal_corner_or_track_across_unknown(self):
        walk = np.eye(4, dtype=bool)
        self.assertFalse(astar(walk, (0, 0), (3, 3)))
        self.assertFalse(clear_segment(walk, (0, 0), (3, 3)))
        walk[:] = True
        walk[2, 2] = False
        self.assertFalse(clear_segment(walk, (0, 2), (3, 2)))
        self.assertTrue(clear_segment(walk, (3, 0), (0, 2)))

    def test_pinhole_and_fully_known_map(self):
        grid = np.zeros((40, 60), dtype=np.int8)
        grid[20, 20] = -1
        self.assertFalse(frontiers(grid).any())
        self.assertFalse(choose_path(traversable(grid, 0.1), frontiers(grid),
                                     (20, 30), 0.1, [], []))
