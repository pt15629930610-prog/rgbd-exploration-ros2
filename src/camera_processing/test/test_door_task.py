"""Entrance-relative task parsing, RGB-D checks and four-door layout."""

import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from camera_processing.door_task import (
    door_box, locate_panel, parking_path, parse_command, parse_task, search_path,
)
from camera_processing.frontier_planner import astar, traversable


class DoorTaskTests(unittest.TestCase):
    def test_commands(self):
        self.assertEqual(parse_task('抵达对面的门'), (0, 1))
        self.assertEqual(parse_task('去对面的棕色门，在门前1米停下'), (0, 1))
        self.assertEqual(parse_task('去左边的那扇棕色房门'), (math.pi/2, 1))
        self.assertEqual(parse_task('去左边的门，在它正前方120厘米停下'), (math.pi/2, 1.2))
        self.assertEqual(parse_task('前往右侧出口，门前0.7米停车'), (-math.pi/2, .7))
        for text in ('去入口的门', '去左边或右边的门', '去左边的门，在门前0.25米停下'):
            with self.assertRaises(ValueError):
                parse_task(text)

    def test_project_panel_and_reject_other_wall(self):
        depth = np.full((240, 320), 2., dtype=np.float32)
        box = door_box('```json\n[{"bbox_2d":[300,50,700,960]}]\n```')
        current = (-1, 2, math.pi/2)
        point = locate_panel(box, depth, 277., 160., current, math.pi/2)
        np.testing.assert_allclose(point, (-1, 4), atol=.02)
        self.assertIsNone(locate_panel(box, depth, 277., 160., current, 0))
        self.assertIsNone(door_box('[]'))
        with self.assertRaises(ValueError):
            door_box('[2,5,2,90]')
        depth.fill(np.inf)
        self.assertIsNone(locate_panel(box, depth, 277., 160., current, math.pi/2))

    def test_speed_clause_combined_and_standalone(self):
        self.assertEqual(parse_command('以0.12米/秒去前面这个门，在门前1米停下'), ((0, 1), .12))
        self.assertEqual(parse_command('去左边的门，以10cm/s行驶，在门前1米停下'),
                         ((math.pi/2, 1), .1))
        self.assertEqual(parse_command('速度改为0.08米/秒'), (None, .08))
        for text in ('速度改为0米/秒', '速度改为0.35米/秒', '以-0.2m/s去对面的门'):
            with self.assertRaises(ValueError):
                parse_command(text)

    def test_long_frontier_route_excludes_actual_failed_frontier(self):
        walk = np.ones((100, 120), dtype=bool)
        frontier = np.zeros_like(walk)
        frontier[50, 100] = frontier[80, 90] = True
        path = search_path(walk, frontier, (50, 10), .05, 0, [])
        self.assertEqual(path[-1], (50, 100))
        alternate = search_path(walk, frontier, (50, 10), .05, 0, [path[-1]])
        self.assertEqual(alternate[-1], (80, 90))

    def test_direction_changes_frontier(self):
        walk = np.ones((61, 61), dtype=bool)
        frontier = np.zeros_like(walk)
        frontier[[3, 57], 30] = True
        self.assertGreater(search_path(walk, frontier, (30, 30), .05, math.pi/2, [])[-1][0], 30)
        self.assertLess(search_path(walk, frontier, (30, 30), .05, -math.pi/2, [])[-1][0], 30)

    def test_locked_goal_frontier_must_advance_towards_parking(self):
        walk = np.ones((80, 80), bool)
        frontier = np.zeros_like(walk)
        frontier[20, 35] = True
        start, hint = (45, 35), (70, 35)
        self.assertEqual(search_path(walk, frontier, start, .05, 0., [], hint), [])
        frontier[61, 35] = True
        self.assertEqual(search_path(walk, frontier, start, .05, 0., [], hint)[-1], (61, 35))

    def test_failed_parking_cell_allows_clear_neighbour_on_same_door(self):
        walk = np.ones((80, 80), bool)
        start, goal, origin = (40, 26), (.025, .025), (-2., -2.)
        failed = {(40, 40)}
        walk[40, 41] = False
        path = parking_path(walk, start, .05, origin, goal, math.pi/2, failed, local=True, roomy=walk)
        self.assertEqual(path[0], start)
        self.assertLessEqual(math.dist(path[-1], (40, 40)), 1.1)
        self.assertNotIn(path[-1], failed)
        self.assertTrue(all(walk[cell] for cell in path))

    def test_greenhouse_aisles_and_parking_access_from_entrance(self):
        sdf = Path(__file__).resolve().parents[1]/'worlds'/'four_doors.sdf'
        world = ET.parse(sdf).getroot().find('world')
        resolution = .05
        origin = np.array([-8., -14.])
        grid = np.zeros((560, 320), dtype=np.int8)

        def pose(element):
            return np.fromstring(element.findtext('pose', '0 0 0 0 0 0'), sep=' ')

        def cell(xy):
            return tuple(np.floor((xy - origin) / resolution).astype(int)[::-1])

        def rotation(yaw):
            c, s = math.cos(yaw), math.sin(yaw)
            return np.array([[c, -s], [s, c]])

        models = {m.get('name'): m for m in world.findall('model')}
        start = pose(models['mobile_robot'])
        np.testing.assert_allclose(start[:2], pose(models['entrance_wall_and_door'])[:2] + [1, 0])
        self.assertAlmostEqual(start[5], 0.)
        beds = [m for m in models.values() if m.find("link/collision[@name='planting_station']") is not None]
        for row_sign in (-1, 1):
            row = [m for m in beds if np.sign(pose(m)[0]) == row_sign]
            self.assertEqual([m.findtext("link/visual/geometry/mesh/uri") for m in
                              sorted(row, key=lambda m: pose(m)[1])],
                             ['meshes/potted_bench.dae', 'meshes/lettuce_bed.dae', 'meshes/strawberry_bed.dae',
                              'meshes/strawberry_bed.dae', 'meshes/lettuce_bed.dae', 'meshes/potted_bench.dae'])

        # Rasterize the actual static walls and planting hardware at chassis height.
        for model in models.values():
            if model.findtext('static') != 'true':
                continue
            placement = pose(model)
            for collider in model.findall('link/collision'):
                size = collider.findtext('geometry/box/size')
                if size is None:
                    continue
                half_size = np.fromstring(size, sep=' ') / 2
                local = pose(collider)
                if abs(.25 - placement[2] - local[2]) > half_size[2]:
                    continue
                center = placement[:2] + rotation(placement[5]) @ local[:2]
                half = np.abs(rotation(placement[5] + local[5])) @ half_size[:2]
                lo, hi = cell(center - half), cell(center + half)
                grid[lo[0]:hi[0]+1, lo[1]:hi[1]+1] = 100
        walk = traversable(grid, resolution)
        # Check every cell along the specified loop and right exit, with planner inflation.
        route = [(-6, 0), (0, 0), (0, 3.11), (5.94, 3.11), (5.94, 0),
                 (0, 0), (0, -9.7), (2.62, -9.7), (2.62, -12.5)]
        for a, b in zip(route, route[1:]):
            for xy in np.linspace(a, b, 160):
                self.assertTrue(walk[cell(xy)], xy)
        self.assertTrue(walk[cell(np.array([3.8, -6.21]))])
        self.assertFalse(walk[cell(np.array([4.82, -6.21]))])
        for name, expected in [('front', (7, 0)), ('left', (0, 11)), ('right', (0, -11))]:
            wall = models[f'{name}_wall_and_door']
            door = pose(wall)[:2]
            np.testing.assert_allclose(door, expected)
            normal = rotation(pose(wall)[5])[:, 0]
            self.assertTrue(np.allclose(normal, door / np.linalg.norm(door)))
            thickness = float(wall.findtext("link/collision[@name='door']/geometry/box/size").split()[0])
            for distance in (.7, 1.2, 3.):
                parked = door - (distance + thickness/2) * normal
                with self.subTest(door=name, distance=distance):
                    self.assertTrue(astar(walk, cell(start[:2]), cell(parked)))
