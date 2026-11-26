"""Kinematic replay on the greenhouse SDF with a progressively revealed 60° depth fan."""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from camera_processing.online_roads import RoadView
from camera_processing.online_skills import Feature, MotionSkills, Unmet, crossed_threshold, wrap
from camera_processing.frontier_planner import obstacle_distance
from camera_processing.online_vision import depth_openings
from test_greenhouse_route import ORIGIN, RESOLUTION, SimulatedRoute, cell


class OnlineReplay(MotionSkills, SimulatedRoute):
    crop_point = (2., 1.56)
    search_turn_slip = .008
    def __init__(self):
        super().__init__()
        self.course = self.entrance[2]
        self.now = 0.
        self.path = []
        self.stamps = {'global': 0.}
        self.responses = 0
        self.feature = self.entered = self.loop_trace = self.vision_key = self.portal = None
        self.vision = SimpleNamespace(reset=lambda: None)
        self.events = []
        self.observed_targets = []

    def note(self, message):
        self.events.append((message, self.here))

    def camera(self):
        self.see(self.now)
        self.view = RoadView(self.grid, self.walk, ORIGIN, RESOLUTION)
        if self.vision_key:
            # Deterministic visual fixture. Only return a crop if its near canopy
            # is within this exposure's field of view; no map landmarks reach the skill.
            exit = self.vision_key[0] == 'exit'
            point = (0., -11.) if exit else self.crop_point
            angle = math.atan2(point[1]-self.pose[1], point[0]-self.pose[0])
            self.responses += 1
            if abs(wrap(angle-self.pose[2])) < math.pi/6 and math.dist(point, self.here) < 4.9:
                if exit and any(self.truth[cell(xy)] for xy in np.linspace(self.here, point, 120)):
                    return
                self.images.append(SimpleNamespace(stamp=self.now, point=point, pose=self.pose,
                                                   heading=-math.pi/2 if exit else None))

    def exercise(self, step, next_step=None):
        self.vision_key = None  # Production stops the previous vision job between skills.
        self.camera()
        skill = self.run_skill(step, next_step)
        for i in range(2200):
            try:
                next(skill)
            except StopIteration:
                return
            x,y,yaw = self.pose
            dt = .18
            slip = self.search_turn_slip if self.vision_key and self.vision_key[0] == 'exit' else .008
            travel = dt*(self.linear+slip*abs(self.angular))
            self.pose = (x+travel*math.cos(yaw), y+travel*math.sin(yaw), wrap(yaw+dt*self.angular))
            if self.portal and crossed_threshold(self.portal, (x, y), self.here):
                self.crossed = True
            if self.loop_trace:
                self.loop_trace.observe(self.here)
            self.now += dt
            if i % 2 == 0:
                self.camera()
            if self.truth[cell(self.here)] == 100:
                raise AssertionError(('kinematic collision', self.pose, step))
        raise AssertionError(('timed out', self.pose, step, self.events))


class DepthExitReplay(OnlineReplay):
    """Require a two-jamb depth fit for exit observations; model class is a fixture."""

    search_turn_slip = .08

    def camera(self):
        checking_exit = self.portal is not None or self.vision_key and self.vision_key[0] == 'exit'
        if checking_exit and obstacle_distance(self.truth, self.cell(), RESOLUTION, .9) < .42:
            raise AssertionError(('chassis clearance', self.pose))
        if not self.vision_key or self.vision_key[0] != 'exit':
            return super().camera()
        self.see(self.now)
        self.view = RoadView(self.grid, self.walk, ORIGIN, RESOLUTION)
        if self.now-getattr(self, 'last_exposure', -10.) < 1.15:
            return
        self.last_exposure = self.now
        self.responses += 1
        x, y, yaw = self.pose
        width = 200
        fx = width*.5/math.tan(math.pi/6)
        lateral = ((width*.5-np.arange(width))/fx)[:, None]
        ranges = np.arange(.05, 5., .04)[None, :]
        xx = x + ranges*(math.cos(yaw)-lateral*math.sin(yaw))
        yy = y + ranges*(math.sin(yaw)+lateral*math.cos(yaw))
        cols = np.floor((xx-ORIGIN[0])/RESOLUTION).astype(int).clip(0, self.grid.shape[1]-1)
        rows = np.floor((yy-ORIGIN[1])/RESOLUTION).astype(int).clip(0, self.grid.shape[0]-1)
        first_hit = np.where(self.truth[rows, cols] == 100, ranges, np.inf).min(axis=1)
        depth = np.tile(first_hit, (96, 1))
        for _, (point, heading) in depth_openings(depth, fx, width*.5, self.pose):
            if math.dist(point, (0., -11.)) < .4:
                self.images.append(SimpleNamespace(stamp=self.now, point=point, heading=heading, pose=self.pose))


class OnlineRouteReplayTest(unittest.TestCase):
    def test_record_and_stop_on_the_same_sightline(self):
        for effect in ('record', 'stop'):
            with self.subTest(effect=effect):
                robot = OnlineReplay()
                robot.truth[:] = 0
                robot.crop_point = (-1., 0.)
                robot.speed_limit = .35
                robot.exercise({'do':'advance', 'distance':3.,
                                'watch':{'target':'door', 'effect':effect, 'side':'front'}})
                self.assertTrue(any('途中确认 door' in msg for msg, _ in robot.events))
                if effect == 'record':
                    self.assertAlmostEqual(robot.here[0], -3., delta=.22)
                    robot.exercise({'do':'seek','target':'door','side':'front'}, 'approach')
                    self.assertEqual(robot.feature.point, robot.crop_point)
                else:
                    self.assertLess(robot.here[0], -5.)
                    self.assertEqual(robot.linear, 0.)

    def test_watch_crossroad_at_different_locations_and_reject_t(self):
        for at, four_way in [(-3., True), (-1.5, True), (-3., False)]:
            with self.subTest(at=at, four_way=four_way):
                robot = OnlineReplay()
                robot.speed_limit = .45
                rows, cols = np.indices(robot.truth.shape)
                x = ORIGIN[0]+(cols+.5)*RESOLUTION
                y = ORIGIN[1]+(rows+.5)*RESOLUTION
                branch = (abs(x-at)<.93) & ((y>0) | four_way)
                robot.truth[:] = np.where((abs(y)<.93) | branch, 0, 100)
                step = {'do':'advance', 'distance':6.,
                        'watch':{'target':'crossroad','effect':'turn','turn_side':'left'}}
                if four_way:
                    robot.exercise(step)
                    self.assertLess(math.dist(robot.here, (at,0.)), .4)
                    self.assertAlmostEqual(robot.pose[2], math.pi/2, delta=.15)
                else:
                    with self.assertRaisesRegex(Unmet, '0/1'):
                        robot.exercise(step)
                    self.assertFalse(any('途中确认' in s for s,_ in robot.events))

    def test_reverse_along_bent_trail_to_a_distant_wide_point(self):
        from camera_processing.frontier_planner import traversable
        robot = OnlineReplay()
        robot.pose = (0.,0.,0.)
        rows,cols = np.indices(robot.grid.shape)
        x = ORIGIN[0]+(cols+.5)*RESOLUTION
        y = ORIGIN[1]+(rows+.5)*RESOLUTION
        midline = lambda x: .24*np.maximum(0.,-x-.3)
        robot.grid[:] = np.where((abs(y-midline(x)) < .58) | (x < -2.5),0,100)
        robot.walk = traversable(robot.grid,RESOLUTION,robot.cell())
        robot.view = RoadView(robot.grid,robot.walk,ORIGIN,RESOLUTION)
        robot.approach_track = [(s,midline(s)) for s in np.arange(-3.5,.01,.055)]
        recovery = robot.retreat_for_turn()
        for _ in range(420):
            try:
                next(recovery)
            except StopIteration as done:
                self.assertTrue(done.value)
                break
            x,y,yaw = robot.pose
            dt = .18
            robot.pose = (x+dt*robot.linear*math.cos(yaw), y+dt*robot.linear*math.sin(yaw),
                          wrap(yaw+dt*robot.angular))
            robot.now += dt
            self.assertTrue(robot.view.footprint_clear(robot.here,robot.pose[2],known=True))
        else:
            self.fail('recovery timed out')
        self.assertLess(robot.here[0],-2.5)
        self.assertTrue(any('分段找宽处' in msg for msg,_ in robot.events))

    def test_non_exit_turn_recovers_from_inside_planning_margin(self):
        robot = OnlineReplay()
        robot.pose = (0., 0., 0.)
        robot.grid[:] = 0
        # An obstacle beside the front wheel blocks rotation but permits reverse.
        robot.grid[cell((.16, .36))] = 100
        from camera_processing.frontier_planner import traversable
        robot.walk = traversable(robot.grid, RESOLUTION, robot.cell())
        robot.view = RoadView(robot.grid, robot.walk, ORIGIN, RESOLUTION)
        robot.vision_start('door')
        robot.approach_track.append((-.65, 0.))
        self.assertTrue(robot.tight_turn(math.pi/2))
        move = robot.retreat_for_turn()
        next(move)
        self.assertLess(robot.linear, 0.)
        self.assertEqual(robot.angular, 0.)
        robot.pose = (-.62, 0., 0.)
        with self.assertRaises(StopIteration) as finished:
            next(move)
        self.assertTrue(finished.exception.value)

    def crop_and_door_robot(self):
        class SceneReplay(OnlineReplay):
            def camera(self):
                if self.vision_key and self.vision_key[0] == 'crop':
                    plants = [(x,y) for x in (-4.7,-3.,-1.2,1.2,3.,4.7)
                              for y in (-7.65,-4.65,-1.55,1.55,4.65,7.65)]
                    visible = [p for p in plants if abs(wrap(math.atan2(
                        p[1]-self.pose[1], p[0]-self.pose[0])-self.pose[2])) < math.pi/6]
                    if visible:
                        self.crop_point = min(visible, key=lambda p: math.dist(self.here, p))
                if self.vision_key and self.vision_key[0] == 'door':
                    self.crop_point = (0.,11.)
                super().camera()
        robot = SceneReplay()
        robot.speed_limit = 1.
        return robot

    def test_logged_second_row_turn_keeps_outgoing_aisle(self):
        robot = self.crop_and_door_robot()
        robot.exercise({'do':'turn','side':'left'})
        second = {'do':'seek','target':'row','ordinal':2}
        robot.exercise(second)
        robot.exercise({'do':'enter'})
        robot.exercise({'do':'traverse_row','watch':{
            'target':'crossroad','ordinal':1,'effect':'turn','turn_side':'right'}}, 'seek')
        center = robot.feature.point
        self.assertLess(robot.here[1], center[1]-.65)
        robot.exercise(dict(second, search_axis='front'))
        gates = [point for msg,point in robot.events if '现场确认 row 第' in msg][-2:]
        self.assertTrue(all(abs(point[0]-center[0]) < .75 for point in gates))
        self.assertLess(gates[1][1], gates[0][1]-1.)
        self.assertAlmostEqual(wrap(robot.feature.heading), 0., delta=.15)
        robot.exercise({'do':'enter'})
        robot.exercise({'do':'patrol'})
        robot.exercise({'do':'seek','target':'door','side':'left'})
        robot.exercise({'do':'approach','distance':1.})
        self.assertAlmostEqual(math.dist(robot.here,(0.,11.)), 1., delta=.10)

    def test_door_search_moves_on_fully_known_map(self):
        robot = OnlineReplay()
        robot.truth[:] = robot.grid[:] = 0
        robot.speed_limit = .5
        robot.crop_point = (.2, 0.)
        robot.exercise({'do':'seek','target':'door','side':'front'})
        self.assertTrue(any('换到观察位' in msg for msg,_ in robot.events))
        self.assertEqual(robot.feature.point, robot.crop_point)

    def test_front_second_row_patrol_then_left_door(self):
        robot = self.crop_and_door_robot()
        robot.exercise({'do':'seek','target':'row','search_axis':'front','ordinal':2})
        gates = [point for msg,point in robot.events if '现场确认 row 第' in msg]
        self.assertEqual(len(gates), 2)
        self.assertGreater(math.dist(*gates), 1.)
        self.assertGreaterEqual(gates[1][0], gates[0][0]-.75)
        robot.exercise({'do':'enter'})
        robot.exercise({'do':'patrol'})
        robot.exercise({'do':'seek','target':'door','side':'left'})
        robot.exercise({'do':'approach','distance':1.})
        self.assertAlmostEqual(math.dist(robot.here,(0.,11.)), 1., delta=.22)

    def test_tight_exit_turn_requires_an_observed_retreat(self):
        robot = OnlineReplay()
        robot.pose = (0., 0., 0.)
        robot.grid[:] = 0
        robot.grid[:, cell((.65, 0.))[1]] = 100
        robot.walk = robot.grid == 0
        robot.view = RoadView(robot.grid, robot.walk, ORIGIN, RESOLUTION)
        robot.vision_start('exit')
        self.assertTrue(robot.tight_turn(math.pi))
        robot.approach_track.append((-.4, 0.))
        robot.walk[cell((-.2, 0.))] = False
        with self.assertRaises(StopIteration) as stopped:
            next(robot.retreat_for_turn())
        self.assertFalse(stopped.exception.value)
        self.assertEqual(robot.linear, 0.)
        robot.walk[cell((-.2, 0.))] = True
        retreat = robot.retreat_for_turn()
        next(retreat)
        self.assertLess(robot.linear, 0.)
        robot.pose = (-.37, 0., 0.)
        with self.assertRaises(StopIteration) as finished:
            next(retreat)
        self.assertTrue(finished.exception.value)
        self.assertEqual(robot.linear, 0.)

    def test_first_row_patrol_returns_from_observed_dead_end_then_exits(self):
        robot = DepthExitReplay()
        robot.speed_limit = .3
        robot.crop_point = (-3.8, 1.9)
        robot.exercise({'do': 'turn', 'side': 'left'})
        robot.exercise({'do': 'seek', 'target': 'row', 'ordinal': 1})
        gate = robot.here
        self.assertLess(math.dist(gate, (-6., 3.11)), .5)
        robot.exercise({'do': 'enter'})
        robot.exercise({'do': 'patrol'})
        self.assertLess(math.dist(robot.here, gate), .3)
        self.assertTrue(any('完成死胡同巡检' in m for m, _ in robot.events))
        robot.exercise({'do': 'seek', 'target': 'door', 'side': 'right'}, next_step='exit')
        robot.exercise({'do': 'exit'})
        self.assertTrue(robot.crossed)

    def test_cross_row_takes_door_branch_only_at_verified_dead_end(self):
        for side, ordinal, plant, dead_end in [('left',1,(-3.8,1.9),True),
                                               ('right',2,(-3.,-4.65),False)]:
            with self.subTest(side=side):
                class DoorReplay(OnlineReplay):
                    def camera(self):
                        if self.vision_key and self.vision_key[0]=='door':
                            self.crop_point = (0.,11.)
                        super().camera()
                robot = DoorReplay()
                robot.speed_limit = .6
                robot.crop_point = plant
                robot.exercise({'do':'turn','side':side})
                robot.exercise({'do':'seek','target':'row','ordinal':ordinal})
                robot.exercise({'do':'enter'})
                robot.exercise({'do':'traverse_row','if_blocked':[
                    {'do':'seek','target':'door','side':'left'}, {'do':'approach','distance':1.5}]})
                self.assertEqual(any('确认行末封堵' in m for m,_ in robot.events), dead_end)
                if dead_end:
                    self.assertAlmostEqual(math.dist(robot.here,(0.,11.)),1.5,delta=.22)
                else:
                    self.assertTrue(any('完成穿行' in m for m,_ in robot.events))
                    self.assertFalse(any('核实 door' in m for m,_ in robot.events))

    def test_crossroad_at_exit_of_first_right_row(self):
        robot = OnlineReplay()
        robot.speed_limit = 1.
        robot.crop_point = (-3.9, -2.2)
        robot.exercise({'do': 'turn', 'side': 'right'})
        robot.exercise({'do': 'seek', 'target': 'row', 'ordinal': 1})
        robot.exercise({'do': 'enter'})
        robot.exercise({'do': 'traverse_row'})
        gate = robot.here
        self.assertLess(math.dist(gate, (0., -3.1)), .45)
        robot.exercise({'do': 'seek', 'target': 'crossroad'})
        self.assertLess(math.dist(robot.here, gate), .45)
        self.assertEqual(robot.feature.kind, 'crossroad')
        robot.exercise({'do': 'turn', 'side': 'left'})
        self.assertLess(abs(wrap(robot.pose[2] - robot.direction('left'))), .15)

    def test_monitor_during_actual_row_traversal(self):
        robot = OnlineReplay()
        robot.speed_limit = .6
        robot.crop_point = (-3.9, -2.2)
        robot.exercise({'do':'turn','side':'right'})
        robot.exercise({'do':'seek','target':'row','ordinal':1})
        robot.exercise({'do':'enter'})
        robot.exercise({'do':'traverse_row', 'watch':{
            'target':'crossroad','effect':'turn','turn_side':'left'}})
        self.assertLess(math.dist(robot.here, (0., -3.1)), .45)
        self.assertAlmostEqual(robot.pose[2], robot.direction('left'), delta=.15)
        self.assertIsNone(robot.loop_trace)

    def test_first_cross_moves_with_unseen_aisle(self):
        for at in (-1.5, 1.8):
            with self.subTest(at=at):
                robot = OnlineReplay()
                rows, cols = np.indices(robot.truth.shape)
                x = ORIGIN[0] + (cols+.5)*RESOLUTION
                y = ORIGIN[1] + (rows+.5)*RESOLUTION
                robot.truth[:] = np.where((abs(y) < .93) | (abs(x-at) < .93), 0, 100)
                self.assertTrue(np.all(robot.grid == -1))
                robot.exercise({'do': 'seek', 'target': 'crossroad'})
                self.assertLess(math.dist(robot.here, (at, 0.)), .35)

    def test_resumed_patrol_counts_only_finished_trips(self):
        robot = OnlineReplay()
        robot.speed_limit = .3
        robot.crop_point = (-3.8, 1.9)
        robot.exercise({'do':'turn','side':'left'})
        robot.exercise({'do':'seek','target':'row','ordinal':1})
        robot.exercise({'do':'enter'})
        robot.exercise({'do':'patrol'})
        robot.feature = None
        robot.exercise({'do':'resume_row'})
        robot.exercise({'do':'patrol','count':2,'total':True})
        self.assertEqual(robot.row_laps, 2)
        self.assertLess(math.dist(robot.here, robot.last_row.point), .3)
        robot.exercise({'do':'patrol','count':2,'total':True})
        self.assertEqual(robot.row_laps, 2)

    def test_immediate_left_turn_at_entrance(self):
        robot = OnlineReplay()
        robot.exercise({'do': 'turn', 'side': 'left'})
        self.assertLess(math.dist(robot.here, robot.entrance[:2]), .2)
        self.assertLess(abs(wrap(robot.pose[2]-math.pi/2)), .1)

    def test_crossing_uses_observed_gate_normal_and_clear_span(self):
        portal = Feature('exit', (5., -3.), math.pi/2)
        self.assertTrue(crossed_threshold(portal, (5.1, -3.1), (5.1, -2.9)))
        self.assertFalse(crossed_threshold(portal, (5.1, -2.9), (5.1, -3.1)))
        self.assertFalse(crossed_threshold(portal, (5.7, -3.1), (5.7, -2.9)))
        self.assertFalse(crossed_threshold(portal, (5., -4.), (5., -2.9)))

    def test_left_turn_observed_row_and_circuit_in_progressively_observed_world(self):
        robot = OnlineReplay()
        robot.exercise({'do':'seek', 'target':'crossroad'})
        self.assertLess(math.dist(robot.here, (0., 0.)), .55)
        robot.exercise({'do':'turn','side':'left'})
        robot.exercise({'do':'seek','target':'row'})
        self.assertLess(math.dist(robot.here, (0., 3.11)), .6)
        robot.exercise({'do':'enter'})
        robot.exercise({'do':'loop'})
        self.assertLess(math.dist(robot.here, (0., 3.11)), .6)
        robot.exercise({'do':'seek', 'target':'door', 'side':'right'}, next_step='exit')
        robot.exercise({'do':'exit'})
        self.assertTrue(robot.crossed)
        self.assertLess(robot.here[1], -11.55)
