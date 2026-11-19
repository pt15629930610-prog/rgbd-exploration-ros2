"""Entrance-relative task parsing, RGB-D checks and four-door layout."""

import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from camera_processing.door_task import door_box, locate_panel, parse_command, parse_task, search_path
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

    def test_initial_occlusion_and_parking_access_from_entrance(self):
        sdf = Path(__file__).resolve().parents[1]/'worlds'/'four_doors.sdf'
        world = ET.parse(sdf).getroot().find('world')
        def xy_pose(model):
            return np.fromstring(model.findtext('pose'), sep=' ')[:2]
        def cell(xy):
            return tuple(np.floor((xy[::-1]+6)/.05).astype(int))
        start = xy_pose(world.find("model[@name='mobile_robot']"))
        boxes = []
        grid = np.zeros((240, 240), dtype=np.int8)
        for m in world.findall('model'):
            if m.get('name') not in ('center_screen', 'left_cabinet', 'right_cabinet'):
                continue
            collider = m.find('link/collision')
            center = xy_pose(collider)
            half = np.fromstring(collider.findtext('geometry/box/size'), sep=' ')[:2]/2
            boxes.append((center, half))
            a,b = cell(center-half), cell(center+half)
            grid[a[0]:b[0]+1, a[1]:b[1]+1] = 100
        grid[:21,:] = grid[219:,:] = grid[:,:21] = grid[:,219:] = 100
        walk = traversable(grid, .05)
        for name in ('front', 'left', 'right'):
            door = xy_pose(world.find(f"model[@name='{name}_wall_and_door']"))
            line = np.linspace(start, door, 201)
            self.assertTrue(any(np.any(np.all(abs(line-c) <= h, axis=1)) for c,h in boxes))
            for distance in (.7, 1.2, 3.):
                parked = door-distance*door/np.linalg.norm(door)
                self.assertTrue(astar(walk, cell(start), cell(parked)))
