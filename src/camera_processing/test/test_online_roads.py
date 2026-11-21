"""Online road reasoning with shifted junctions, T branches and partial observations."""

import math
import unittest

import numpy as np

from camera_processing.frontier_planner import traversable
from camera_processing.online_roads import JunctionTracker, RoadView


RES = .05
ORIGIN = (-6.5, -5.5)


def crossing(at, *, right=True, blocked=False, blind_right=False, yaw=0.):
    r, c = np.indices((220, 260))
    x, y = ORIGIN[0]+(c+.5)*RES, ORIGIN[1]+(r+.5)*RES
    along = x*math.cos(yaw)+y*math.sin(yaw)
    across = -x*math.sin(yaw)+y*math.cos(yaw)
    corridor = abs(across) <= .93
    branches = (abs(along-at) <= .93) & ((across > 0) | right)
    grid = np.where(corridor | branches, 0, 100).astype(np.int8)
    if blind_right:
        grid[(across < -.52) & (abs(along-at) <= .93)] = -1
    if blocked:
        grid[abs(along-(at-1.15)) <= .12] = 100
    return RoadView(grid, traversable(grid, RES), ORIGIN, RES)


class OnlineRoadTest(unittest.TestCase):
    def test_moved_crossing_is_discovered_at_new_position(self):
        for at in (0., 1.73, 3.04):
            with self.subTest(at=at):
                road = crossing(at)
                here = (at-1.4, 0.)
                junction, = road.junctions(here, 0.)
                self.assertAlmostEqual(junction.point[0], at, delta=.15)
                self.assertEqual(junction.branches, {'forward', 'back', 'left', 'right'})
                self.assertEqual(junction.crop_sides, {'left', 'right'})
                if at >= 1.73:
                    self.assertFalse(road.junctions((-2.8, 0.), 0., horizon=1.7))

    def test_robot_reference_rotates_with_aisle(self):
        yaw = math.radians(34.)
        road = crossing(.3, right=False, yaw=yaw)
        here = (-math.cos(yaw)*1.4, -math.sin(yaw)*1.4)
        j, = road.junctions(here, yaw)
        self.assertEqual(j.branches, {'forward', 'back', 'left'})
        self.assertLess(math.dist(j.point, (.3*math.cos(yaw), .3*math.sin(yaw))), .18)

    def test_unknown_arm_requires_observation_before_confirming_crossroad(self):
        road = crossing(.4, blind_right=True)
        here = (.35, 0.)
        partial, = road.junctions(here, 0.)
        self.assertNotIn('right', partial.branches)
        self.assertIn('right', road.probe_sides(here, 0.))
        complete = crossing(.4).junctions(here, 0.)
        tracker = JunctionTracker()
        self.assertFalse(tracker.update([partial], 1.0))
        self.assertFalse(tracker.update(complete, 1.0))
        self.assertNotIn('right', tracker.update(complete, 1.1)[0].branches)
        self.assertIn('right', tracker.update(complete, 1.2)[0].branches)

    def test_open_room_and_inaccessible_cross_do_not_generate_events(self):
        grid = np.zeros((100, 100), np.int8)
        v = RoadView(grid, grid == 0, (-2.5,-2.5), RES)
        self.assertFalse(v.junctions((0., 0.), 0.))
        walled = crossing(.0, blocked=True)
        self.assertFalse(walled.junctions((-2.1, 0.), 0.))

    def test_frontier_direction_and_visited_filter(self):
        grid = np.full((120, 160), -1, np.int8)
        grid[31:89,45:115] = 0
        v = RoadView(grid, grid == 0, (-4., -3.), RES)
        for yaw in (0., math.pi/2, math.pi):
            path = v.frontier_path((0.,0.), yaw)
            self.assertTrue(path)
            a = v.xy(path[-1])
            self.assertLess(abs(math.remainder(math.atan2(a[1], a[0])-yaw, 2*math.pi)), .15)
            another = v.frontier_path((0.,0.), yaw, visited=[a])
            self.assertGreater(math.dist(v.xy(another[-1]), a), .64)
            self.assertTrue(all(v.walk[p] for p in another))

    def test_advance_uses_only_seen_safe_cells(self):
        road = crossing(0., blocked=True)
        here = (-2.1, 0.)
        self.assertLess(road.clearance(here, 0.), .45)
        self.assertFalse(road.path_to(here, (0.,0.)))
        path = road.advance(here, 0.)
        self.assertTrue(all(road.walk[p] for p in path))
        self.assertLess(road.xy(path[-1])[0], -1.65)
