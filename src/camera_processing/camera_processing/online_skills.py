"""Observation-driven driving skills. All distances and bearings are measured in odom."""

from collections import deque
from dataclasses import dataclass
import math

from .frontier_planner import astar, clear_segment, inside, observation_path
from .online_roads import JunctionTracker


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def offset(point, yaw, distance):
    return point[0]+distance*math.cos(yaw), point[1]+distance*math.sin(yaw)


def components(start, end, yaw):
    x, y = end[0]-start[0], end[1]-start[1]
    return x*math.cos(yaw)+y*math.sin(yaw), -x*math.sin(yaw)+y*math.cos(yaw)


def bearing(start, end):
    return math.atan2(end[1]-start[1], end[0]-start[0])


class RouteHistory:
    """Measured odometry polyline with arc-length addresses for recovery nodes."""

    def __init__(self):
        self.track = deque(maxlen=6000)
        self.nodes = deque(maxlen=256)
        self.stage_start = 0.

    def observe(self, point):
        if not self.track:
            self.track.append((0., point))
            self.nodes.append((0., point))
            return
        arc, last = self.track[-1]
        distance = math.dist(last, point)
        if distance >= .08:
            self.track.append((arc+distance, point))

    def begin(self):
        if self.track:
            self.stage_start = self.track[-1][0]

    def checkpoint(self):
        if not self.track or self.track[-1][0]-self.stage_start < .8:
            return
        mid = (self.stage_start+self.track[-1][0])/2
        if mid < self.track[0][0]:
            return
        node = min(self.track, key=lambda entry: abs(entry[0]-mid))
        if not self.nodes or node[0] > self.nodes[-1][0]+.3:
            self.nodes.append(node)

    def candidates(self, here):
        return [node for node in reversed(self.nodes) if node[0] >= self.track[0][0]
                and math.dist(node[1], here) >= .45] if self.track else []


class Unmet(RuntimeError):
    """A requested action could not be verified from the live observations."""


class SearchExhausted(Unmet):
    """No more reachable candidates along this search course."""


@dataclass
class Feature:
    kind: str
    point: tuple
    heading: float
    crop: tuple | None = None
    approach_yaw: float | None = None
    row_exit: tuple | None = None


def crossed_threshold(portal, previous, current):
    before, ay = components(portal.point, previous, portal.heading)
    after, by = components(portal.point, current, portal.heading)
    if not before < 0. <= after or math.dist(previous[:2], current[:2]) >= .45:
        return False
    lateral = ay + (by-ay)*(-before)/(after-before)
    return abs(lateral) < .38


class LoopTrace:
    """A genuine circuit winds around the observed plant and returns to its own gate."""

    def __init__(self, gate, plant):
        self.trail = [gate]
        self.gate, self.plant, self.last = gate, plant, gate
        self.angle = self.distance = 0.

    def observe(self, point):
        if math.dist(self.trail[-1], point) >= .45:
            self.trail.append(point)
        step = math.dist(self.last, point)
        if step > .65:
            raise Unmet('绕行时位姿不连续，无法核实闭环')
        self.angle += wrap(bearing(self.plant, point)-bearing(self.plant, self.last))
        self.distance += step
        self.last = point

    def closed(self):
        return (self.distance > 3. and math.dist(self.last, self.gate) <= .30
                and abs(self.angle) > 1.85*math.pi)


class MotionSkills:
    """Generators yield at each control tick so camera, map and stop callbacks keep running."""

    def direction(self, side):
        return wrap(self.entrance[2]+{'front': 0., 'back': math.pi,
                                     'left': math.pi/2, 'right': -math.pi/2}[side])

    def in_region(self, point, directions):
        return all(components(self.entrance, point, self.direction(side))[0] > .35 for side in directions)

    @property
    def here(self):
        return self.pose[:2]

    def clear_path(self):
        if self.path:
            self.set_path([])

    def remember_position(self):
        if not hasattr(self, 'approach_track'):
            self.approach_track = deque(maxlen=240)
        if not self.approach_track or math.dist(self.here, self.approach_track[-1]) >= .055:
            self.approach_track.append(self.here)

    def tight_turn(self, error):
        if abs(error) < .15:
            return False
        clearance = self.view.clearance_map[self.cell()]
        if self.vision_key and self.vision_key[0] == 'exit' and abs(error) >= .5:
            if clearance < .46 + .10*abs(error):
                return True
        if clearance > .62:
            return False
        # Include chassis drift during a significant rotation.
        if clearance < .46+.04*abs(error):
            return True
        count = math.ceil(abs(error)/.10)
        return any(not self.view.footprint_clear(self.here, self.pose[2]+error*i/count, known=False)
                   for i in range(count+1))

    def retreat_for_turn(self):
        """Check the swept chassis on short, fully observed forward and reverse segments."""
        self.command()
        self.clear_path()
        origin, yaw = self.here, self.pose[2]
        history = list(getattr(self, 'approach_track', ()))
        if not inside(self.walk, self.cell()):
            return False
        # Prefer the measured approach, then consider an observed forward pocket.
        distances = []
        for point in reversed(getattr(self, 'approach_track', ())):
            along, lateral = components(origin, point, yaw)
            if -.15 >= along >= -1.6 and abs(lateral) < .12*(-along):
                distances.append(along)
        distances += [.25, -.25, .45, -.45, .65, -.65, .85, -.85, 1.1, -1.1]
        exit_search = self.vision_key is not None and self.vision_key[0] == 'exit'
        minimums = (.78,) if exit_search else (.78, .65)
        for margin, distance in ((m, d) for m in minimums for d in distances):
            point = offset(origin, yaw, distance)
            cell = self.view.cell(point)
            remaining, lateral = components(self.here, point, self.pose[2])
            if abs(lateral) > .09 or remaining*distance <= 0:
                continue
            if (inside(self.walk, cell) and self.walk[cell] and self.view.clearance_map[cell] >= margin
                    and (not self.walk[self.cell()] or clear_segment(self.walk, self.cell(), cell))
                    and self.view.straight_clear(self.here, self.pose[2], remaining)):
                self.note(f'转向前沿已观测直线挪车 {distance:+.2f} m，目标宽处 {point}')
                begin = self.now
                sign = 1 if distance > 0 else -1
                while math.dist(self.here, point) > .08:
                    remaining, lateral = components(self.here, point, self.pose[2])
                    if (self.now-begin > 4.+abs(distance)/min(.09, self.speed_limit)
                            or sign*remaining <= 0 or abs(lateral) > .09
                            or abs(wrap(self.pose[2]-yaw)) > .12
                            or (sign > 0 and self.forward_blocked)
                            or not self.local_translation_clear(remaining)
                            or not self.view.straight_clear(self.here, self.pose[2], remaining)):
                        break
                    self.command(sign*min(.09, self.speed_limit))
                    yield
                self.command()
                if math.dist(self.here, point) <= .08 and self.view.clearance_map[self.cell()] >= margin-.05:
                    return True
        if history and math.dist(origin,self.here) > .2:
            history = list(self.approach_track)
        return (yield from self.retrace_to_room(history, minimums[-1]))

    def slow_turn_clear(self, error):
        cell = self.cell()
        margin = getattr(self, 'slow_turn_clearance', .48)
        if (not inside(self.walk,cell) or self.view.clearance_map[cell] < margin
                or getattr(self, 'turn_blocked', False)):
            return False
        hits = getattr(self, 'local_hits', None)
        if hits is not None:
            x,y,res = hits
            if ((x*x+y*y) < (margin+.71*res)**2).any():
                return False
        n = max(1, math.ceil(abs(error)/.09))
        return all(self.view.footprint_clear(self.here,self.pose[2]+error*i/n,known=True)
                   for i in range(n+1))

    def retrace_to_room(self, history, margin, *, goal=None):
        """Follow the recent driven polyline in short verified steps, including bends."""
        near = [i for i,p in enumerate(history) if math.dist(self.here,p) <= .20]
        if not near:
            self.note('车迹恢复未找到与当前位姿相连的起点')
            return False
        points = []
        last = self.here
        for p in reversed(history[:near[-1]+1]):
            length = math.dist(last,p)
            if length < .16:
                continue
            if length > .45:
                break
            points.append(p)
            last = p
        if goal is not None and math.dist(last, goal) > .075 and math.dist(last, goal) <= .45:
            points.append(goal)
        if goal is None:
            self.note(f'直线换位未成功，沿近期车迹分段找宽处，可检查 {len(points)} 段')
        for target in points:
            start = self.now
            while math.dist(self.here,target) > .075:
                distance = math.dist(self.here,target)
                error = wrap(bearing(self.here,target)-self.pose[2])
                sign = 1 if abs(error) < math.pi/2 else -1
                if sign < 0:
                    error = wrap(error+math.pi)
                yaw = self.pose[2]
                if abs(error) >= .55 and self.now-start < 10. and self.slow_turn_clear(error):
                    self.command(angular=max(-.22,min(.22,error)))
                    yield
                    continue
                # Look ahead only as far as this control segment. Revalidate
                # both the translation and the incremental change of heading.
                hop = sign*min(distance,.14)
                delta = max(-.12,min(.12,error))
                if abs(error) >= .85 or distance >= .48:
                    reason = '当前位姿无法衔接这段车迹'
                elif self.now-start >= 10.:
                    reason = '本段挪车没有及时到位'
                elif (sign > 0 and self.forward_blocked) or not self.local_translation_clear(hop):
                    reason = '实时局部障碍进入移动包络'
                elif not (self.view.straight_clear(self.here,yaw,hop)
                          and self.view.straight_clear(self.here,yaw+delta,hop)):
                    reason = '全局移动包络含有占据、未知或越界栅格'
                else:
                    reason = None
                if reason:
                    self.command()
                    self.note(f'分段车迹恢复中止：{reason}，位姿 {self.pose}，'
                              f'目标 {target}，距离 {distance:.2f} m，航向差 {math.degrees(error):.0f}°')
                    return False
                self.command(sign*min(.07,self.speed_limit), max(-.18,min(.18,.9*error)))
                yield
            cell = self.cell()
            reached = (math.dist(self.here,goal) <= .12 if goal is not None else
                       inside(self.walk,cell) and self.walk[cell] and self.view.clearance_map[cell] >= margin)
            if reached:
                self.command()
                self.note(f'沿车迹到达换向点 {self.here}，继续原任务')
                return True
        self.command()
        return False

    def local_translation_clear(self, distance):
        hits = getattr(self, 'local_hits', None)
        if hits is None:
            return not getattr(self, 'turn_blocked', False)
        x, y, resolution = hits
        pad = .045+.71*resolution
        return not ((x >= min(0., distance)-.34-pad) & (x <= max(0., distance)+.34+pad)
                    & (abs(y) <= .275+pad)).any()

    def aim(self, yaw):
        self.clear_path()
        start = self.now
        revision = getattr(self, 'motion_revision', 0)
        slow = False
        while abs(wrap(yaw-self.pose[2])) > .065:
            if revision != getattr(self, 'motion_revision', 0):
                revision, start = self.motion_revision, self.now
            if self.now-start > 22.:
                raise Unmet('转向未达到要求的朝向')
            error = wrap(yaw-self.pose[2])
            tight = self.tight_turn(error)
            if tight and self.slow_turn_clear(error):
                if not slow:
                    self.note('保守转向余量不足，已核实转动包络空闲，低速转向并逐帧复核')
                    start = self.now
                slow = True
                self.command(angular=max(-.20,min(.20,1.45*error)))
                yield
                continue
            if tight:
                if not (yield from self.retreat_for_turn()):
                    raise Unmet(f'转向空间不足，已检查直线换位和分段车迹恢复，当前位姿 {self.pose}')
                start = self.now
                continue
            slow = False
            self.command(angular=max(-.46, min(.46, 1.45*error)))
            yield
        self.command()

    def look(self, yaw):
        yield from self.aim(yaw)
        settled = self.stamps['global']
        while self.stamps['global'] <= settled:
            self.command()
            yield

    def drive_path(self, path, stop=None, *, tolerance=.12):
        """Track one observed path; return false if new evidence invalidates it."""
        if not path:
            return False
        self.set_path(path)
        goal = self.xy(path[-1])
        index, last_progress = 0, self.now
        revision = getattr(self, 'motion_revision', 0)
        tight_sites = []
        best = math.inf
        while math.dist(self.here, goal) > tolerance:
            if stop and stop():
                self.command()
                return True
            start = self.cell()
            if revision != getattr(self, 'motion_revision', 0):
                path = self.view.replan_after_turn(self.here, goal, tight_sites+getattr(self, 'recovery_sites', []))
                if not path:
                    break
                self.set_path(path)
                index, last_progress, best = 0, self.now, math.inf
                revision = self.motion_revision
            if self.now-last_progress > 16.:
                break
            exit_search = self.vision_key is not None and self.vision_key[0] == 'exit'
            self.remember_position()
            targets = []
            for i in range(max(0, index-4), min(len(path), index+24)):
                p = self.xy(path[i])
                if .045 < math.dist(self.here, p) <= .34 and clear_segment(self.walk, start, path[i]):
                    targets.append((i, p))
            if not targets:
                break
            if exit_search and self.view.turning_room[start]:
                roomy = [t for t in targets if clear_segment(self.view.turning_room, start, path[t[0]])]
                targets = roomy or targets[:1]
            i, point = targets[-1]
            dist = math.dist(self.here, goal)
            if i > index or dist < best-.05:
                last_progress, best = self.now, dist
            index = max(i, index)
            error = wrap(bearing(self.here, point)-self.pose[2])
            if self.tight_turn(error):
                if self.slow_turn_clear(error):
                    yield from self.aim(bearing(self.here, point))
                    continue
                if sum(math.dist(self.here, p)<.3 for p in tight_sites) >= 2:
                    break
                tight_sites.append(self.here)
                if not (yield from self.retreat_for_turn()):
                    break
                yield from self.aim(bearing(self.here, point))
                self.motion_revision = getattr(self, 'motion_revision', 0)+1
                continue
            angular = max(-.46, min(.46, 1.55*error))
            if abs(error) > .15:
                self.command(angular=angular)
            elif self.forward_blocked:
                break
            else:
                cap = self.speed_limit
                if exit_search and self.view.clearance_map[start] < .85:
                    cap = min(cap, .12)
                self.command(min(cap, .62*dist), angular)
            yield
        self.command()
        self.clear_path()
        return math.dist(self.here, goal) <= tolerance

    def recover_at_node(self, history, nodes, goal):
        for arc, node in nodes:
            self.command()
            self.clear_path()
            self.note(f'前往 {goal} 受阻，尝试退回实测节点 {node}')
            try:
                arrived = yield from self.drive_path(self.view.path_to(self.here, node))
            except Unmet:
                self.command()
                arrived = False
            if not arrived:
                trail = [p for distance,p in history.track if distance >= arc]
                if not trail or trail[0] != node or not (
                        yield from self.retrace_to_room(trail, 0., goal=node)):
                    continue
            self.command()
            settled = self.stamps['global']
            while self.stamps['global'] <= settled:
                yield
            self.note(f'已回到节点 {node}，按最新全局 OCC 重规划至 {goal}')
            return True
        return False

    def travel(self, point, *, straight=False, explore=False, precise=False):
        """Replan to a measured point. Extend a straight segment as its corridor becomes visible."""
        retries = 0
        nodes = None
        # Stop within the final grid cell's quantization error plus tracking tolerance.
        arrival = .06 + self.resolution / math.sqrt(2) if precise else .20
        while math.dist(self.here, point) > arrival:
            dist = math.dist(self.here, point)
            yaw = bearing(self.here, point)
            try:
                if straight:
                    yield from self.look(yaw)
                    path = self.view.advance(self.here, yaw, min(1.1, dist))
                else:
                    if retries:
                        yield from self.look(yaw)
                    path = self.view.path_to(self.here, point)
                    if not path and explore:
                        frontier = self.view.frontier_path(self.here, target=point)
                        if frontier and math.dist(self.xy(frontier[-1]), point) < dist-.18:
                            path = frontier[:max(4, round(1.1/self.resolution))]
                final_leg = path and self.view.cell(point) == path[-1]
                progress = yield from self.drive_path(path, tolerance=.06 if precise and final_leg else .12)
            except Unmet as exc:
                self.command()
                self.note(f'当前路径无法跟踪：{exc}')
                retries, progress = 2, False
            if progress and math.dist(self.here, point) < dist-.04:
                retries = 0
            else:
                retries += 1
                if retries >= 3:
                    history = getattr(self, 'route_history', None)
                    if history is not None and nodes is None:
                        history.checkpoint()
                        nodes = iter(history.candidates(self.here))
                    if not (yield from self.recover_at_node(history, nodes or (), point)):
                        raise Unmet('当前目标无安全路线，可用回退节点已检查')
                    retries = 0

    def probe(self, travel_yaw, tracker, *, force=False):
        """Merge candidate observations across fresh maps and short, targeted side looks."""
        def collect():
            observations = self.view.junctions(self.here, travel_yaw)
            confirmed = tracker.update(observations, self.stamps['global'])
            events = getattr(self, 'stage_crossings', None)
            if events is not None:
                events.extend((j, travel_yaw) for j in confirmed)
            return confirmed

        seen = list(collect())
        sides = self.view.probe_sides(self.here, travel_yaw)
        if force:
            sides = {'left', 'right'}
        if sides or seen:
            for name, sign in [('left', 1), ('right', -1)]:
                if name in sides:
                    yield from self.look(travel_yaw+sign*math.radians(75))
                    seen.extend(collect())
            yield from self.look(travel_yaw)
            seen.extend(collect())
        return sorted(seen, key=lambda j: components(self.here, j.point, travel_yaw)[0])

    def branches(self, j, yaw, side):
        for name, sign in [('left', 1), ('right', -1)]:
            if name in j.branches:
                angle = wrap(yaw+sign*math.pi/2)
                if not side or abs(wrap(angle-self.direction(side))) < math.pi/4:
                    yield angle

    def vision_start(self, kind, label=''):
        self.vision.reset()
        self.vision_key = kind, label
        self.images = deque(maxlen=16)
        self.panels = deque(maxlen=12)
        self.remember_position()
        self.snapshot = None
        self.vision_errors = 0

    def match(self, accepts, images=None, outward=None):
        # Two frames of the same target must agree in world coordinates.
        images = self.images if images is None else images
        for i in reversed(range(1, len(images))):
            e = images[i]
            if (self.now-e.stamp > 12. or not accepts(e.point)
                    or outward is not None and (e.heading is None or math.cos(e.heading-outward) < .5)):
                continue
            for old in list(images)[:i]:
                if (old.stamp < e.stamp and e.stamp-old.stamp < 12. and accepts(old.point)
                        and math.dist(old.point, e.point) < .55
                        and (e.heading is None or old.heading is not None
                             and abs(wrap(e.heading-old.heading)) < .30)):
                    return e
        return None

    def row_bank_match(self, gate, yaw):
        # The referent is a planted bank. The clearest individual plant can change
        # between frames, especially when looking obliquely along the same bank.
        recent = []
        for e in self.images:
            if e.point is None or not 0 <= self.now-e.stamp < 12.:
                continue
            along, lateral = components(gate, e.point, yaw)
            if not (.05 < along < 4.8 and .35 < abs(lateral) < 2.1):
                continue
            for stamp, x, y in recent:
                if (stamp < e.stamp and y*lateral > 0 and
                        abs(y-lateral) <= .35 and abs(x-along) <= 1.25):
                    return e
            recent.append((e.stamp, along, lateral))
        return None

    def confirm_row(self, gate, yaw, label):
        yield from self.look(yaw)
        self.vision_start('crop', label)
        def nearby(p):
            along, lateral = components(gate, p, yaw)
            return .05 < along < 4.8 and .35 < abs(lateral) < 2.1

        # Look at both shoulders; foliage beside the near edge can be outside the central frame.
        for angle in (0., .34, -.34):
            yield from self.look(yaw+angle)
            start, initial = self.now, self.responses
            while self.now-start < 15. and self.responses-initial < 4:
                match = self.match(nearby)
                if match is None and not label:
                    match = self.row_bank_match(gate, yaw)
                if match:
                    self.vision_key = None
                    return match.point
                yield
        self.vision_key = None
        return None

    def seek_junction(self, step, seen=None, row_exit=None, branch=None):
        target = step['target']
        order = step.get('order')
        rank_all = order in ('last', 'middle')
        reference = None
        if order == 'other':
            reference = getattr(self, 'last_confirmed_row', None) or getattr(self, 'last_row', None)
        if order == 'other' and reference is None:
            raise Unmet('“另一条”需要一条已确认的作物行作为参照')
        side = step.get('side')
        region = step.get('region', [])
        area = side and (target == 'crossroad' or step.get('area'))
        search_axis = step.get('search_axis') if branch is None else None
        yaw = (self.direction(search_axis) if search_axis else
               self.direction(region[0]) if region else self.direction(side) if area else self.course)
        if seen is not None:
            yaw=self.course
        if branch is not None:
            yaw = branch.heading
        elif reference and reference.approach_yaw is not None and not (search_axis or side or region):
            yaw = reference.approach_yaw
        self.course = yaw
        origin = branch.point if branch is not None else self.here
        ordinal = step.get('ordinal', 1)
        counted, excluded, visited = [], [], []
        ordered_rows = []
        failed_bypasses = set()
        reversed_search = False
        detours, tried = [], []
        tracker = JunctionTracker()
        while True:
            yield from self.aim(yaw)
            forced = self.view.clearance(self.here, yaw) < .65
            at_exit = row_exit is not None and math.dist(self.here, row_exit) <= .45
            candidates = yield from self.probe(yaw, tracker, force=forced or at_exit)
            for j in candidates:
                current_exit = at_exit and math.dist(j.point, row_exit) <= .45
                if not current_exit and components(origin, j.point, yaw)[0] < (.55 if seen is None else -.35):
                    continue
                if (search_axis and counted and
                        components(counted[-1], j.point, self.direction(search_axis))[0] < -.75):
                    continue
                if any(math.dist(j.point, p) < .9 for p in counted+excluded):
                    continue
                if branch is not None or order:
                    along, across = components(origin, j.point, yaw)
                    if abs(across) > .75 or along < (1.1 if branch is not None else .55) or (
                            counted and components(counted[-1], j.point, yaw)[0] < .9):
                        continue
                if area and components(self.entrance, j.point, yaw)[0] < .4:
                    continue
                crop = None
                angle = yaw
                if target == 'crossroad':
                    if not {'forward', 'back', 'left', 'right'} <= j.branches or not self.in_region(j.point, region):
                        continue
                else:
                    directions = list(self.branches(j, yaw, None if area else side))
                    if reference is not None and math.dist(reference.point, j.point) < .95:
                        continue
                    if reference is not None:
                        # Exclude re-entry at the opposite mouth of the same observed aisle.
                        distance, lateral = components(reference.point, j.point, reference.heading)
                        if (distance > 0 and abs(lateral) < .65 and
                                self.view.clearance(reference.point, reference.heading, distance) >= distance-.15):
                            directions = [a for a in directions if math.cos(a-reference.heading) > -.8]
                    if branch is not None and step.get('search_axis'):
                        target_yaw = self.direction(step['search_axis'])
                        if abs(math.sin(target_yaw-yaw)) > .7:
                            directions = [a for a in directions if math.cos(a-target_yaw) > .7]
                        elif not self.in_region(j.point, [step['search_axis']]):
                            continue
                    if seen is not None:
                        directions = [a for a in directions if not any(
                            math.dist(j.point,f.point)<.95 and math.cos(a-f.heading)>.7 for f in seen)]
                    directions = [angle for angle in directions if self.in_region(offset(j.point, angle, 1.3), region)]
                    if 'parallel_to' in step:
                        directions = [a for a in directions if math.cos(a-step['parallel_to']) > .8]
                    if search_axis and len(directions) > 1:
                        def entry_clearance(angle):
                            cell = self.view.cell(offset(j.point, angle, .65))
                            return self.view.clearance_map[cell] if inside(self.walk, cell) and self.walk[cell] else 0.
                        directions.sort(key=entry_clearance, reverse=True)
                    if search_axis and abs(wrap(yaw-self.direction(search_axis))) < .1:
                        for angle in directions:
                            if not any(math.dist(j.point,p) < .9 and abs(wrap(angle-a)) < .3
                                       for p,a in detours+tried):
                                detours.append((j.point, angle))
                    if not directions:
                        continue
                    angle = directions[0]
                    if target == 'row':
                        yield from self.travel(j.point)
                        verified_lane = False
                        for angle in directions:
                            yield from self.look(angle)
                            side_name = 'left' if wrap(angle-yaw) > 0 else 'right'
                            refreshed = self.view.junctions(self.here, yaw)
                            if not any(math.dist(k.point, j.point) < .85 and side_name in k.crop_sides
                                       for k in refreshed):
                                # A terminal side lane can lose its junction signature once
                                # the camera faces into it. Recheck the measured branch itself;
                                # still require a clear lane and two agreeing crop observations.
                                if not (rank_all and self.view.clearance(j.point, angle, 2.) >= 1.8):
                                    continue
                            verified_lane = True
                            crop = yield from self.confirm_row(j.point, angle, step.get('label', ''))
                            if not crop:
                                # Move only on an observed, traversable path and restore
                                # the gate pose before testing another branch or counting.
                                vantage = offset(j.point, angle, .65)
                                if (self.view.clearance(j.point, angle, 1.2) >= 1.05
                                        and self.view.path_to(self.here, vantage)):
                                    self.note('行口作物证据不足，进入已测通道换位复核')
                                    yield from self.travel(vantage)
                                    crop = yield from self.confirm_row(j.point, angle, step.get('label', ''))
                                    yield from self.travel(j.point)
                            if crop:
                                break
                        if crop is None:
                            if not verified_lane:
                                continue
                            if step.get('label'):
                                excluded.append(j.point)
                                self.note(f'此行口未核实指定作物 {step["label"]}，继续寻找')
                                continue
                            raise Unmet('发现疑似行口，但两帧 RGB-D 尚未核实通道旁的作物')
                counted.append(j.point)
                self.note(f'现场确认 {target} 第 {len(counted)} 处，位于 {j.point}')
                found = Feature(target, j.point, angle, crop, approach_yaw=yaw)
                if rank_all:
                    ordered_rows.append(found)
                elif len(counted) == ordinal:
                    yield from self.travel(j.point)
                    self.feature = found
                    if target == 'row':
                        self.last_confirmed_row = found
                    return
            yield from self.aim(yaw)
            path = self.view.advance(self.here, yaw, .72)
            if (yield from self.drive_path(path)):
                continue
            if target == 'row':
                bypass = self.view.aisle_bypass(self.here, yaw, origin, failed_bypasses)
                if bypass:
                    self.note('沿行口搜索方向受阻，从已知空闲侧绕行，再继续数行')
                    if not (yield from self.drive_path(bypass)):
                        failed_bypasses.add(bypass[-1])
                    continue
            if rank_all:
                yield from self.look(yaw)
                if not self.view.observed_dead_end(self.here, yaw, banks=False):
                    raise Unmet('当前无法继续沿通道前进，尚未观测到尽头，无法判断最后或中间的行口')
                if not ordered_rows:
                    raise SearchExhausted('直到通道尽头仍未确认符合条件的行口')
                # In an even set use the first of the two central entrances.
                index = len(ordered_rows)-1 if order == 'last' else (len(ordered_rows)-1)//2
                selected = ordered_rows[index]
                self.note(f'实测通道结束，共确认 {len(ordered_rows)} 个行口；返回第 {index+1} 处 {selected.point}')
                yield from self.travel(selected.point)
                self.feature = self.last_confirmed_row = selected
                return
            if order == 'other' and not branch and not reversed_search and not (search_axis or region or side):
                yield from self.travel(origin)
                yaw = self.course = wrap(yaw+math.pi)
                tracker = JunctionTracker()
                reversed_search = True
                continue
            if order:
                raise SearchExhausted('这条通道上未能确认符合位置约束的作物行')
            if search_axis and detours:
                # Explore observed side aisles from the far end of the selected
                # direction when the main road has yielded too few crop gates.
                gate, angle = max(detours, key=lambda d: components(
                    self.entrance, d[0], self.direction(search_axis))[0])
                detours.remove((gate, angle))
                tried.append((gate, angle))
                if math.dist(self.here, gate) < .2 or self.view.path_to(self.here, gate):
                    yield from self.travel(gate)
                    yaw, origin, self.course = angle, gate, angle
                    tracker = JunctionTracker()
                    self.note(f'主通道尚未数够行口，回到已观测岔口 {gate} 检查侧道')
                continue
            if seen is None and branch is None and (area or region or search_axis):
                path = self.view.frontier_path(self.here, direction=yaw, visited=visited)
                if path and len(path) > 3 and (not search_axis or
                        components(self.here, self.xy(path[-1]), yaw)[0] > .15):
                    end = self.xy(path[-1])
                    moved = yield from self.drive_path(path[:max(4, round(1.1/self.resolution))])
                    if not moved or math.dist(self.here, end) < .45:
                        visited.append(end)
                    continue
            raise SearchExhausted(f'沿途已确认 {len(counted)}/{ordinal} 处 {target}，没有再观测到可达的指定路口')

    def patrol_rows(self, step):
        seen, finished, visited = [], [], []
        anchor, course = self.here, self.course
        indices = step.get('indices', [])
        search = dict(step['filter'])
        axis = search.pop('search_axis',None)
        if search.pop('area',False):
            side=search.pop('side')
            search['region']=list(dict.fromkeys([*search.get('region',[]),side]))
            axis=axis or side
        if axis:
            course=self.direction(axis)
        def inspect(row):
            self.feature = row
            yield from self.enter()
            yield from self.run_skill({'do':'patrol','count':step['count']},None)
            finished.append(row)
            self.note(f'已完成 {len(finished)} 个现场核实的行口，最近行口 {row.point}')
            self.deadline = self.now + 720.*step['count']
        if step.get('start'):
            yield from self.seek_junction(step['start'])
            yield from inspect(self.feature)
            if not step.get('all'):
                seen.append(self.feature)
            else:
                yield from self.travel(anchor)
        self.course = course
        directions = iter([course+math.pi/2,course-math.pi/2,course+math.pi])
        remaining = list(indices)
        while True:
            while remaining and remaining[0]<=len(seen):
                row=seen[remaining.pop(0)-1]
                if row not in finished:
                    yield from inspect(row)
                    if remaining:
                        yield from self.travel(seen[-1].point)
                        self.course=seen[-1].approach_yaw
            if indices and not remaining:
                return
            try:
                yield from self.seek_junction(search, seen=seen)
            except SearchExhausted:
                yaw = next(directions,None)
                if yaw is not None:
                    yield from self.travel(anchor)
                    self.course = yaw
                    continue
                path = self.view.frontier_path(self.here,visited=visited)
                if path and len(path)>1:
                    end = self.xy(path[-1])
                    visited.append(end)
                    if not (yield from self.drive_path(path)):
                        raise Unmet('剩余探索边界暂时无法安全抵达')
                    self.course=bearing(self.xy(path[0]),end)
                    anchor=self.here
                    directions=iter([self.course+math.pi/2,self.course-math.pi/2,self.course+math.pi])
                    continue
                if remaining or not finished:
                    raise Unmet(f'已观测 {len(seen)} 个不同行口，未找到要求的其余行')
                self.note(f'当前可达探索边界已耗尽，完成 {len(finished)} 个已核实行口')
                return
            row=self.feature
            seen.append(row)
            course=row.approach_yaw
            if step.get('all') and not any(math.dist(row.point,f.point)<.95 and
                                           math.cos(row.heading-f.heading)>.7 for f in finished):
                yield from inspect(row)
            self.course=course

    def inspect_panel(self, panel, verified):
        """Look beside a measured door leaf from reachable, front-facing viewpoints."""
        normal = panel.heading
        if self.view.turning_room[self.cell()]:
            yield from self.look(bearing(self.here, panel.point))
            start, responses = self.now, self.responses
            while self.now-start < 8. and self.responses-responses < 2 and not verified():
                yield
            if verified():
                return
        views = []
        for gap in (-1.25, 0., 1.25):
            aim = offset(panel.point, normal+math.pi/2, gap)
            for standoff in (2.6, 2.1, 3.1):
                point = offset(aim, normal, -standoff)
                cell = self.view.cell(point)
                if not inside(self.walk, cell) or not self.view.turning_room[cell]:
                    continue
                path = astar(self.walk, self.cell(), cell, roomy=self.view.turning_room)
                if path:
                    views.append((len(path), point, aim))
                    break
        for _, point, aim in sorted(views):
            path = astar(self.walk, self.cell(), self.view.cell(point), roomy=self.view.turning_room)
            if not (yield from self.drive_path(path, stop=verified)):
                continue
            if verified():
                return
            yield from self.look(bearing(self.here, aim))
            start, responses = self.now, self.responses
            while self.now-start < 12. and self.responses-responses < 4 and not verified():
                self.command()
                yield
            if verified():
                return

    def seek_visual(self, step, kind):
        self.vision_start(kind, step.get('label', ''))
        side = step.get('side')
        direction = self.direction(side) if side else None
        region = step.get('region', [])
        if region:
            axial = self.direction(region[0])
            direction = wrap(axial + .5*wrap(self.direction(region[1])-axial))
        counted = []
        visited = []
        failed_looks = 0
        inspected = []
        settled = []
        looked_from = []
        surveyed = []

        def accepts(p):
            in_sector = direction is None or (math.dist(self.entrance[:2], p) > .45 and
                abs(wrap(bearing(self.entrance, p)-direction)) < math.radians(55))
            return in_sector and self.in_region(p, region) and all(math.dist(p, q) > .9 for q in counted)

        def verified():
            return self.match(accepts, outward=direction if kind == 'exit' else None)

        def pending_gate():
            if kind != 'exit' or not self.images:
                return None
            cue = self.images[-1]
            if (self.now-cue.stamp > 3. or not accepts(cue.point)
                    or direction is not None and math.cos(cue.heading-direction) < .5):
                return None
            if any(self.now-t < 10. and math.dist(cue.point, p) < .8 for t, p in settled):
                return None
            return cue

        def new_panel(p):
            return accepts(p) and all(math.dist(p, q) > 1.1 or math.dist(self.here, observer) > 2.5
                                      for q, observer in inspected)

        def panel_hint():
            if kind == 'exit':
                return self.match(new_panel, self.panels, outward=direction)
            return None

        while True:
            m = verified()
            if m:
                counted.append(m.point)
                if len(counted) >= step.get('ordinal', 1):
                    self.feature = Feature(kind, m.point, m.heading if m.heading is not None
                                           else bearing(m.pose, m.point))
                    self.note(f'两帧 RGB-D 核实 {kind}，位置 {m.point}')
                    return
            cue = pending_gate()
            if cue:
                self.command()
                self.clear_path()
                settled.append((self.now, cue.point))
                self.note(f'门洞首帧通过，停车等待第二帧确认 {cue.point}')
                start, responses = self.now, self.responses
                while self.now-start < 8. and self.responses-responses < 4 and not verified():
                    yield
                continue
            panel = panel_hint()
            if panel:
                inspected.append((panel.point, self.here))
                self.note(f'发现候选门板 {panel.point}，寻找能看到两侧门框的安全观察点')
                yield from self.inspect_panel(panel, lambda: verified() or pending_gate())
                continue
            if (kind in ('exit', 'door') and direction is not None and self.view.turning_room[self.cell()]
                    and all(math.dist(self.here, p) > 2.2 for p in looked_from)):
                looked_from.append(self.here)
                yield from self.look(direction)
                start, responses = self.now, self.responses
                while self.now-start < 6. and self.responses == responses and not pending_gate():
                    yield
                continue
            path = self.view.frontier_path(self.here, direction=direction, visited=visited,
                                           room_to_turn=kind in ('exit', 'door'))
            if path and len(path) > 3:
                end = self.xy(path[-1])
                if kind == 'exit':
                    path = observation_path(path, self.grid, self.resolution)
                prefix = path[:max(4, round(1.15/self.resolution))]
                progress = yield from self.drive_path(prefix, stop=lambda: verified() or pending_gate() or panel_hint())
                if math.dist(self.here, end) < .45 or not progress:
                    visited.append(end)
                if progress:
                    failed_looks = 0
                    continue
            # Observe other directions only when the current observed frontier does not advance.
            yaw = direction if direction is not None else self.course
            if failed_looks >= 4:
                if kind not in ('door', 'exit'):
                    raise Unmet('可观测范围内尚未找到符合约束的目标')
                surveyed.append(self.here)
                alternative = self.view.survey_path(self.here, surveyed, direction)
                if not alternative:
                    raise Unmet(f'未核实目标门；当前可达地图已无新的边界或间隔足够的安全观察位，已检查 {len(surveyed)} 个位置记录')
                station = self.xy(alternative[-1])
                self.note(f'当前视角未确认门，沿已知通路换到观察位 {station}，允许折返')
                moved = yield from self.drive_path(alternative, stop=lambda: verified() or pending_gate() or panel_hint())
                if verified() or pending_gate() or panel_hint():
                    continue
                if not moved:
                    surveyed.append(station)
                    continue
                surveyed.append(self.here)
                for i in range(8):
                    yield from self.look(yaw+i*math.pi/4)
                    start, n = self.now, self.responses
                    while self.now-start < 8. and self.responses-n < 2 and not verified() and not panel_hint():
                        yield
                    if verified() or pending_gate() or panel_hint():
                        break
                # Retry frontiers on the refreshed map before selecting another station.
                continue
            yield from self.look(yaw+(0., math.pi/2, -math.pi/2, math.pi)[failed_looks])
            start, n = self.now, self.responses
            while self.now-start < 8. and self.responses-n < 2 and not verified() and not panel_hint():
                yield
            failed_looks += 1

    def need(self, kinds):
        if self.feature is None or self.feature.kind not in kinds:
            raise Unmet('缺少上一阶段现场确认的目标')
        return self.feature

    def enter(self):
        f = self.need(('row', 'opening'))
        yield from self.travel(f.point)
        self.course = f.heading
        start = self.here
        self.entered = None
        if f.kind == 'row':
            self.loop_trace = LoopTrace(start, f.crop)
        yield from self.travel(offset(f.point, f.heading, .9), straight=True)
        if f.kind == 'row':
            self.entered = f
            self.last_confirmed_row = f
            if getattr(self, 'last_row', None) is not f:
                self.last_row = f
                self.row_laps = 0

    def shoulder_seen(self, yaw):
        # Unknown or out-of-map cells cannot establish that the junction ended.
        for distance in (.55, .70, .85, 1., 1.15, 1.30, 1.45, 1.60, 1.75, 1.90):
            cell = self.view.cell(offset(self.here, yaw, distance))
            if inside(self.grid, cell) and self.grid[cell] >= 50:
                return True
        return False

    def loop(self, patrol=False):
        f = self.need(('row',))
        if self.entered is not f or self.loop_trace is None:
            raise Unmet('需要先进入已确认的作物行，再开始绕行')
        trace = self.loop_trace
        hand = 1 if components(f.point, f.crop, f.heading)[1] > 0 else -1
        yaw = f.heading
        corner = f.point
        tracker = JunctionTracker()
        turns = 0
        checked_at = None
        try:
            while not trace.closed():
                if turns >= 3:
                    forward, lateral = components(self.here, trace.gate, yaw)
                    if -.1 < forward < 1.5 and abs(lateral) < .50:
                        yield from self.travel(trace.gate, straight=True)
                        if trace.closed():
                            break
                        raise Unmet('回到附近时，车迹未环绕作物一周')
                limited = self.view.clearance(self.here, yaw) < .65
                candidates = yield from self.probe(yaw, tracker, force=limited)
                side = 'left' if hand == 1 else 'right'
                j = next((j for j in candidates if side in j.branches
                          and components(corner, j.point, yaw)[0] > 1.0), None)
                if j is not None:
                    if turns >= 3 and math.dist(j.point, trace.gate) < .8:
                        yield from self.travel(trace.gate)
                        if trace.closed():
                            break
                    yield from self.travel(j.point)
                    yaw = wrap(yaw+hand*math.pi/2)
                    self.course, corner = yaw, j.point
                    tracker = JunctionTracker()
                    turns += 1
                    self.note(f'绕行现场转角 {turns}，位置 {j.point}')
                    if turns > 6:
                        raise Unmet('已走过多个转角，仍未核实作物闭环')
                    yield from self.look(yaw)
                    continue
                if not (yield from self.drive_path(self.view.advance(self.here, yaw, .72))):
                    if patrol:
                        refreshed = yield from self.probe(yaw, tracker, force=True)
                        side_open = any(j.branches & {'left', 'right'} and
                                        -.35 <= components(self.here, j.point, yaw)[0] <= 1.35
                                        for j in refreshed)
                        if not side_open and self.view.observed_dead_end(self.here, yaw):
                            self.loop_trace = None
                            self.note('补看前方和两侧后确认行末封堵，沿实测车迹退回本行入口')
                            for point in reversed(trace.trail):
                                yield from self.travel(point)
                            self.entered = None
                            self.note(f'已回到本行入口 {trace.gate}，完成死胡同巡检')
                            return
                        if checked_at is None or math.dist(self.here, checked_at) > .35:
                            checked_at = self.here
                            continue
                    raise Unmet('行间前方受阻，未形成闭环')
        except Unmet as exc:
            self.loop_trace = None
            self.note(f'{exc}，尝试沿已观测通路退回行口')
            yield from self.travel(trace.gate)
            raise Unmet(f'{exc}，已回到行口') from exc
        self.loop_trace = self.entered = None
        self.note('车迹已环绕观测到的植株一周，并回到同一行口')

    def traverse_row(self, step):
        row = self.need(('row',))
        if self.entered is not row or self.loop_trace is None:
            raise Unmet('需要先驶入已实测的行口')
        trace = self.loop_trace
        tracker = JunctionTracker()
        yaw = row.heading
        while True:
            near_end = self.view.clearance(self.here, yaw) < .65
            junctions = yield from self.probe(yaw, tracker, force=near_end)
            for j in junctions:
                along, lateral = components(row.point, j.point, yaw)
                if along < 1.5 or abs(lateral) > .65:
                    continue
                # A transverse aisle beyond the banks, or a measured T at their end.
                arms = j.branches & {'left', 'right'}
                if len(arms) == 2 or (arms and self.view.observed_dead_end(j.point, yaw, banks=False)):
                    yield from self.travel(j.point)
                    self.entered = self.loop_trace = None
                    row.row_exit = j.point
                    self.note(f'到达作物行另一端实测横道 {j.point}，完成穿行')
                    return
            if (yield from self.drive_path(self.view.advance(self.here, yaw, .72))):
                continue
            if not near_end or not self.view.observed_dead_end(self.here, yaw):
                raise Unmet('行内暂时不能前进，尚无足够观测确认行末封堵')
            if not step.get('if_blocked'):
                raise Unmet('实测行末封堵，无法穿过该行')
            self.entered = self.loop_trace = None
            self.note('确认行末封堵，沿入行车迹退回后执行条件分支')
            for point in reversed(trace.trail):
                yield from self.travel(point)
            for action in step['if_blocked']:
                self.vision_key = None
                self.vision.reset()
                yield from self.run_skill(action, None)
                self.command()
            return

    def exit_door(self):
        f = self.need(('exit',))
        outer = offset(f.point, f.heading, .85)
        yield from self.travel(offset(f.point, f.heading, -.82), explore=True)
        start = self.here
        inside, _ = components(f.point, start, f.heading)
        if inside > -.35:
            raise Unmet('未确认当前在门洞内侧')
        self.portal = f
        self.crossed = False
        yield from self.travel(outer, straight=True)
        self.portal = None
        if not self.crossed:
            raise Unmet('缺少连续位姿穿越观测门槛的证据')

    def watch_sector(self, point, side):
        if not side:
            return True
        return (math.dist(self.entrance[:2], point) > .45 and
                abs(wrap(bearing(self.entrance, point)-self.direction(side))) < math.radians(55))

    def run_watched(self, step, next_step):
        host = dict(step)
        rule = host.pop('watch')
        if host['do'] not in ('advance', 'traverse_row', 'loop', 'patrol') or 'if_blocked' in host:
            raise Unmet('当前运动阶段不支持该监测范围')
        target, effect = rule['target'], rule['effect']
        motion = self.run_skill(host, next_step)
        tracker = JunctionTracker()
        counted = []
        axis = self.course if host['do'] != 'advance' or not host.get('side') else self.direction(host['side'])
        stamp = -math.inf
        last_peek = None
        fired = False
        finished = None
        if target != 'crossroad':
            self.vision_start(target, rule.get('label', ''))
        else:
            self.stage_crossings = deque(maxlen=32)
        self.note(f'开始途中监测 {rule}')
        try:
            while True:
                features = []
                if not fired and target == 'crossroad' and self.stamps['global'] >= stamp+.15:
                    stamp = self.stamps['global']
                    # Keep the approach axis stable during side looks. A four-way crossing
                    # has the same topology when viewed along either aisle.
                    observed = self.view.junctions(self.here, axis, horizon=1.5)
                    confirmed = tracker.update(observed, stamp)
                    if (host['do'] == 'advance' and finished is None and abs(wrap(self.pose[2]-axis)) < .25
                            and (last_peek is None or math.dist(last_peek, self.here) > .55)
                            and self.view.probe_sides(self.here, axis)):
                        last_peek = self.here
                        self.command()
                        confirmed = yield from self.probe(axis, tracker)
                    self.stage_crossings.extend((j, axis) for j in confirmed)
                if not fired and target == 'crossroad':
                    while self.stage_crossings:
                        j, approach = self.stage_crossings.popleft()
                        if ({'left','right','forward','back'} <= j.branches
                                and math.dist(self.here, j.point) <= 1.5):
                            features.append(Feature(target, j.point, approach))
                elif not fired and target != 'crossroad':
                    e = self.match(lambda p: self.watch_sector(p, rule.get('side')) and
                                   all(math.dist(p, old) > .9 for old in counted))
                    if e is not None:
                        features.append(Feature(target, e.point, e.heading if e.heading is not None
                                                else bearing(e.pose, e.point)))
                for feature in features:
                    if (not self.watch_sector(feature.point, rule.get('side'))
                            or any(math.dist(feature.point, p) < .9 for p in counted)):
                        continue
                    counted.append(feature.point)
                    if len(counted) != rule.get('ordinal', 1):
                        continue
                    fired = True
                    self.note(f'途中确认 {target} 第 {len(counted)} 处 {feature.point}，响应 {effect}')
                    if target != 'crossroad':
                        self.vision_key = None
                        self.vision.reset()
                    if effect == 'record':
                        self.observed_targets.append((feature, rule.get('label', '')))
                        break
                    self.command()
                    self.clear_path()
                    motion.close()
                    self.loop_trace = self.entered = None
                    self.feature = feature
                    if effect == 'turn':
                        yield from self.travel(feature.point)
                        yield from self.run_skill({'do':'turn', 'side':rule['turn_side']}, next_step)
                    return
                if finished is not None:
                    # Give the last in-scope exposure time to finish; the car is stationary.
                    if (not fired and target != 'crossroad' and self.now < finished+5.
                            and getattr(self.vision, 'pending', None) is not None):
                        yield
                        continue
                    if not fired and effect != 'record':
                        raise Unmet(f'运动阶段结束，途中确认 {target} {len(counted)}/{rule.get("ordinal",1)} 处')
                    if not fired:
                        self.note(f'运动阶段结束，未记录到指定的 {target}')
                    return
                try:
                    next(motion)
                except StopIteration:
                    self.command()
                    finished = self.now
                    if target != 'crossroad' and not fired:
                        self.vision.capture_until = finished
                    continue
                yield
        finally:
            motion.close()
            if target != 'crossroad':
                self.vision_key = None
                self.vision.reset()
            else:
                self.stage_crossings = None

    def run_skill(self, step, next_step):
        if 'watch' in step:
            yield from self.run_watched(step, next_step)
            return
        op = step['do']
        row_exit = self.feature.row_exit if self.feature is not None else None
        if row_exit is not None:
            self.feature.row_exit = None
        if op == 'speed':
            self.command()
        elif op == 'patrol_rows':
            yield from self.patrol_rows(step)
        elif op == 'seek':
            branch = self.feature if self.feature and self.feature.kind == 'branch' else None
            self.feature = self.entered = self.loop_trace = None
            kind = step['target']
            if kind == 'crossroad' and row_exit is not None:
                self.note(f'先复核刚到达的行末横道 {row_exit} 是否四向连通')
                yield from self.seek_junction(step, row_exit=row_exit)
            elif kind in ('crossroad', 'opening', 'row'):
                if branch is not None and kind == 'row':
                    self.note(f'沿刚驶入的支路按经过顺序计数，路口 {branch.point}，入口方位约束 {step.get("search_axis")}')
                    yield from self.seek_junction(step, branch=branch)
                else:
                    yield from self.seek_junction(step)
            else:
                if kind == 'door' and next_step == 'exit':
                    kind = 'exit'
                # A saved observation identifies a particular target in this instruction.
                # The approach skill still rechecks it from its requested stopping pose.
                if (kind in ('door', 'crop') and next_step == 'approach'
                        and step.get('ordinal', 1) == 1 and not step.get('region')):
                    for feature, label in reversed(getattr(self, 'observed_targets', [])):
                        if (feature.kind == kind and label == step.get('label', '')
                                and self.watch_sector(feature.point, step.get('side'))):
                            self.feature = feature
                            self.note(f'采用途中实测的 {kind} 位置 {feature.point}')
                            return
                yield from self.seek_visual(step, kind)
        elif op in ('resume_row', 'neighbor'):
            previous = getattr(self, 'last_row', None)
            if previous is None:
                raise Unmet('尚无已进入的作物行可作为参照，请先指定一条作物行')
            if op == 'resume_row':
                self.feature = previous
            else:
                yaw = self.direction(step['side'])
                if abs(math.cos(yaw-previous.heading)) > .5:
                    raise Unmet('指定的入口方位沿着原行，无法在这一侧确认平行的相邻行')
                yield from self.travel(previous.point)
                self.course = yaw
                yield from self.seek_junction({'target': 'row', 'parallel_to': previous.heading})
        elif op == 'turn':
            self.course = self.direction(step['side'])
            yield from self.aim(self.course)
            if self.feature and self.feature.kind == 'crossroad' and next_step == 'seek':
                center = self.feature.point
                yield from self.travel(offset(center, self.course, .95), straight=True)
                self.feature = Feature('branch', center, self.course)
                self.note(f'已从十字路口驶入支路，实测位置 {self.here}')
        elif op == 'advance':
            if step.get('side'):
                self.course = self.direction(step['side'])
            yield from self.travel(offset(self.here, self.course, step.get('distance', 1.)), straight=True)
        elif op == 'pass':
            f = self.need(('crossroad', 'opening'))
            yield from self.travel(offset(f.point, self.course, .95), straight=True)
            sides = [f.heading] if f.kind == 'opening' else [self.course+math.pi/2, self.course-math.pi/2]
            # Re-observe the banks beyond the mouth before declaring the junction passed.
            for _ in range(5):
                for angle in sides:
                    if not self.shoulder_seen(angle):
                        yield from self.look(angle)
                if all(self.shoulder_seen(a) for a in sides):
                    break
                yield from self.travel(offset(self.here, self.course, .4), straight=True)
            else:
                raise Unmet('尚未观测到路口的远端')
        elif op == 'enter':
            yield from self.enter()
        elif op == 'traverse_row':
            yield from self.traverse_row(step)
        elif op == 'loop':
            yield from self.loop()
        elif op == 'patrol':
            row = self.need(('row',))
            count = step.get('count', 1)
            if step.get('total'):
                count = max(0, count-getattr(self, 'row_laps', 0))
            for _ in range(count):
                if self.entered is not row:
                    yield from self.enter()
                yield from self.loop(patrol=True)
                self.row_laps = getattr(self, 'row_laps', 0)+1
                self.note(f'本行已完成 {self.row_laps} 趟巡检')
        elif op == 'exit':
            if self.feature is not None and self.feature.kind == 'door':
                door = self.feature
                yield from self.travel(offset(door.point, door.heading, -1.5))
                yield from self.look(bearing(self.here, door.point))
                self.vision_start('exit')
                start = self.now
                while self.now-start < 22.:
                    e = self.match(lambda p: math.dist(p, door.point) < 2.0)
                    if e is not None:
                        self.feature = Feature('exit', e.point, e.heading)
                        break
                    yield
                self.vision_key = None
            yield from self.exit_door()
        elif op == 'landmark_stop':
            f = self.need(('row', 'opening', 'crossroad'))
            yaw = f.approach_yaw if f.approach_yaw is not None else self.course
            point = offset(f.point, yaw, -step['distance'])
            if not inside(self.grid, self.view.cell(point)):
                raise Unmet('指定的停车点在当前地图边界外')
            yield from self.travel(point, explore=True)
            yield from self.aim(yaw)
            gap, lateral = components(self.here, f.point, yaw)
            if abs(gap-step['distance']) > .22 or abs(lateral) > .25:
                raise Unmet('与实测路口的停车距离仍有偏差')
            self.note(f'相对已观测路口停车，纵向 {gap:.2f} m，横向偏差 {lateral:.2f} m')
        elif op == 'approach':
            f = self.need(('door', 'exit', 'crop', 'object'))
            distance = step.get('distance', 1.)
            near_door = f.kind == 'door' and distance < 4.5
            for _ in range(2):
                point = offset(f.point, f.heading, -distance)
                if not inside(self.grid, self.view.cell(point)):
                    raise Unmet('指定的停车点在当前地图边界外')
                yield from self.travel(point, explore=True, precise=near_door)
                yield from self.look(bearing(self.here, f.point))
                if distance >= 4.5:
                    gap, lateral = components(self.here, f.point, f.heading)
                    if abs(gap-distance) > .22 or abs(lateral) > .25:
                        raise Unmet('相对先前实测目标的位置仍未满足停车距离')
                    self.note(f'远距停车，依据已确认目标与里程计复核 {gap:.2f} m，横向 {lateral:.2f} m')
                    return
                self.vision_start(f.kind, step.get('label', ''))
                start = self.now
                while self.now-start < 22.:
                    e = self.match(lambda p: math.dist(p, f.point) < .65)
                    if e is not None:
                        break
                    yield
                else:
                    raise Unmet('到达估计停车点后，新的 RGB-D 帧尚未复核同一目标')
                yaw = e.heading if e.heading is not None else bearing(self.here, e.point)
                self.feature = f = Feature(f.kind, e.point, yaw)
                gap, lateral = components(self.here, f.point, yaw)
                if abs(gap-distance) <= (.10 if near_door else .22) and abs(lateral) <= .25:
                    self.note(f'停车复核通过，实测距离 {gap:.2f} m，横向偏差 {lateral:.2f} m')
                    return
            raise Unmet('现场复核仍未满足指令中的停车距离')
        elif op in ('inspect', 'photo'):
            self.command()
            if step.get('face_target') and self.feature and self.feature.kind in ('door', 'exit', 'crop', 'object'):
                yield from self.look(bearing(self.here, self.feature.point))
            elif step.get('face_target') and self.feature and self.feature.crop is not None:
                yield from self.look(bearing(self.here, self.feature.crop))
            self.vision_start(op, step.get('label', ''))
            start = self.now
            while self.snapshot is None:
                if self.now-start > 45.:
                    raise Unmet('未能取得本次观察的新画面')
                yield
            if self.journal is None and op == 'photo':
                raise Unmet('本次日志不可写，照片无法保存')
            stored = self.note(self.snapshot.detail, jpeg=self.snapshot.jpeg)
            if op == 'photo' and not stored:
                raise Unmet('记录画面时写入失败')
        elif op == 'home':
            yield from self.travel(self.entrance[:2])
            yield from self.aim(self.entrance[2])
            self.course = self.entrance[2]
        elif op == 'wait':
            until = self.now+step['seconds']
            self.command()
            while self.now < until:
                yield
        else:
            raise Unmet(step.get('reason', '当前无法执行该动作'))
