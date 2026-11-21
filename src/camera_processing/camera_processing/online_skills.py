"""Observation-driven driving skills. All distances and bearings are measured in odom."""

from collections import deque
from dataclasses import dataclass
import math

from .frontier_planner import clear_segment, inside
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


class Unmet(RuntimeError):
    """A requested action could not be verified from the live observations."""


@dataclass
class Feature:
    kind: str
    point: tuple
    heading: float
    crop: tuple | None = None


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
        self.gate, self.plant, self.last = gate, plant, gate
        self.angle = self.distance = 0.

    def observe(self, point):
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

    def aim(self, yaw):
        self.clear_path()
        start = self.now
        while abs(wrap(yaw-self.pose[2])) > .065:
            if self.now-start > 22.:
                raise Unmet('转向未达到要求的朝向')
            error = wrap(yaw-self.pose[2])
            self.command(angular=max(-.46, min(.46, 1.45*error)))
            yield
        self.command()

    def look(self, yaw):
        yield from self.aim(yaw)
        settled = self.stamps['global']
        while self.stamps['global'] <= settled:
            self.command()
            yield

    def drive_path(self, path, stop=None):
        """Track one observed path; return false if new evidence invalidates it."""
        if not path:
            return False
        self.set_path(path)
        goal = self.xy(path[-1])
        index, last_progress = 0, self.now
        best = math.inf
        while math.dist(self.here, goal) > .12:
            if stop and stop():
                self.command()
                return True
            if self.now-last_progress > 16.:
                break
            start = self.cell()
            targets = []
            for i in range(max(0, index-4), min(len(path), index+24)):
                p = self.xy(path[i])
                if .045 < math.dist(self.here, p) <= .34 and clear_segment(self.walk, start, path[i]):
                    targets.append((i, p))
            if not targets:
                break
            i, point = targets[-1]
            dist = math.dist(self.here, goal)
            if i > index or dist < best-.05:
                last_progress, best = self.now, dist
            index = max(i, index)
            error = wrap(bearing(self.here, point)-self.pose[2])
            angular = max(-.46, min(.46, 1.55*error))
            if abs(error) > .15:
                self.command(angular=angular)
            elif self.forward_blocked:
                break
            else:
                self.command(min(self.speed_limit, .62*dist), angular)
            yield
        self.command()
        self.clear_path()
        return math.dist(self.here, goal) <= .12

    def travel(self, point, *, straight=False, explore=False):
        """Replan to a measured point. Extend a straight segment as its corridor becomes visible."""
        retries = 0
        while math.dist(self.here, point) > .20:
            dist = math.dist(self.here, point)
            yaw = bearing(self.here, point)
            if straight:
                yield from self.look(yaw)
                path = self.view.advance(self.here, yaw, min(1.1, dist))
            else:
                path = self.view.path_to(self.here, point)
                if not path and explore:
                    frontier = self.view.frontier_path(self.here, target=point)
                    if frontier and math.dist(self.xy(frontier[-1]), point) < dist-.18:
                        path = frontier[:max(4, round(1.1/self.resolution))]
            progress = yield from self.drive_path(path)
            if progress and math.dist(self.here, point) < dist-.04:
                retries = 0
            else:
                retries += 1
                if retries >= 3:
                    raise Unmet('已观测区域中没有通往当前目标的安全路线')
                yield from self.look(yaw)

    def probe(self, travel_yaw, tracker, *, force=False):
        """Merge candidate observations across fresh maps and short, targeted side looks."""
        def collect():
            observations = self.view.junctions(self.here, travel_yaw)
            return tracker.update(observations, self.stamps['global'])

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
        self.snapshot = None
        self.vision_errors = 0

    def match(self, accepts):
        # Two frames of the same target must agree in world coordinates.
        for i in reversed(range(1, len(self.images))):
            e = self.images[i]
            if not accepts(e.point):
                continue
            for old in list(self.images)[:i]:
                if (old.stamp < e.stamp and e.stamp-old.stamp < 12. and accepts(old.point)
                        and math.dist(old.point, e.point) < .55
                        and (e.heading is None or old.heading is not None
                             and abs(wrap(e.heading-old.heading)) < .30)):
                    return e
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
                if match:
                    self.vision_key = None
                    return match.point
                yield
        self.vision_key = None
        return None

    def seek_junction(self, step):
        target = step['target']
        side = step.get('side')
        region = step.get('region', [])
        area = side and (target == 'crossroad' or step.get('area'))
        yaw = self.direction(region[0]) if region else (self.direction(side) if area else self.course)
        self.course = yaw
        origin = self.here
        ordinal = step.get('ordinal', 1)
        counted, excluded, visited = [], [], []
        tracker = JunctionTracker()
        while True:
            yield from self.aim(yaw)
            forced = self.view.clearance(self.here, yaw) < .65
            candidates = yield from self.probe(yaw, tracker, force=forced)
            for j in candidates:
                if components(origin, j.point, yaw)[0] < .55:
                    continue
                if any(math.dist(j.point, p) < .9 for p in counted+excluded):
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
                    directions = [angle for angle in directions if self.in_region(offset(j.point, angle, 1.3), region)]
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
                                continue
                            verified_lane = True
                            crop = yield from self.confirm_row(j.point, angle, step.get('label', ''))
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
                if len(counted) == ordinal:
                    yield from self.travel(j.point)
                    self.feature = Feature(target, j.point, angle, crop)
                    return
            yield from self.aim(yaw)
            path = self.view.advance(self.here, yaw, .72)
            if (yield from self.drive_path(path)):
                continue
            if area or region:
                path = self.view.frontier_path(self.here, direction=yaw, visited=visited)
                if path and len(path) > 3:
                    end = self.xy(path[-1])
                    moved = yield from self.drive_path(path[:max(4, round(1.1/self.resolution))])
                    if not moved or math.dist(self.here, end) < .45:
                        visited.append(end)
                    continue
            raise Unmet('沿途没有再观测到可达的指定路口')

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

        def accepts(p):
            in_sector = direction is None or (math.dist(self.entrance[:2], p) > .45 and
                abs(wrap(bearing(self.entrance, p)-direction)) < math.radians(55))
            return in_sector and self.in_region(p, region) and all(math.dist(p, q) > .9 for q in counted)

        while True:
            m = self.match(accepts)
            if m:
                counted.append(m.point)
                if len(counted) >= step.get('ordinal', 1):
                    self.feature = Feature(kind, m.point, m.heading if m.heading is not None
                                           else bearing(m.pose, m.point))
                    self.note(f'两帧 RGB-D 核实 {kind}，位置 {m.point}')
                    return
            path = self.view.frontier_path(self.here, direction=direction, visited=visited)
            if path and len(path) > 3:
                end = self.xy(path[-1])
                prefix = path[:max(4, round(1.15/self.resolution))]
                progress = yield from self.drive_path(prefix, stop=lambda: self.match(accepts) is not None)
                if math.dist(self.here, end) < .45 or not progress:
                    visited.append(end)
                if progress:
                    failed_looks = 0
                    continue
            # Observe other directions only when the current observed frontier does not advance.
            yaw = direction if direction is not None else self.course
            if failed_looks >= 4:
                raise Unmet('可观测范围内尚未找到符合约束的目标')
            yield from self.look(yaw+(0., math.pi/2, -math.pi/2, math.pi)[failed_looks])
            start, n = self.now, self.responses
            while self.now-start < 8. and self.responses-n < 2 and not self.match(accepts):
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

    def shoulder_seen(self, yaw):
        # Unknown or out-of-map cells cannot establish that the junction ended.
        for distance in (.55, .70, .85, 1., 1.15, 1.30, 1.45, 1.60, 1.75, 1.90):
            cell = self.view.cell(offset(self.here, yaw, distance))
            if inside(self.grid, cell) and self.grid[cell] >= 50:
                return True
        return False

    def loop(self):
        f = self.need(('row',))
        if self.entered is not f or self.loop_trace is None:
            raise Unmet('需要先进入已确认的作物行，再开始绕行')
        trace = self.loop_trace
        hand = 1 if components(f.point, f.crop, f.heading)[1] > 0 else -1
        yaw = f.heading
        corner = f.point
        tracker = JunctionTracker()
        turns = 0
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
                    raise Unmet('行间前方受阻，未形成闭环')
        except Unmet as exc:
            self.loop_trace = None
            self.note(f'{exc}，尝试沿已观测通路退回行口')
            yield from self.travel(trace.gate)
            raise Unmet(f'{exc}，已回到行口') from exc
        self.loop_trace = self.entered = None
        self.note('车迹已环绕观测到的植株一周，并回到同一行口')

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

    def run_skill(self, step, next_step):
        op = step['do']
        if op == 'seek':
            self.feature = self.entered = self.loop_trace = None
            kind = step['target']
            if kind in ('crossroad', 'opening', 'row'):
                yield from self.seek_junction(step)
            else:
                if kind == 'door' and next_step == 'exit':
                    kind = 'exit'
                yield from self.seek_visual(step, kind)
        elif op == 'turn':
            self.course = self.direction(step['side'])
            yield from self.aim(self.course)
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
        elif op == 'loop':
            yield from self.loop()
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
        elif op == 'approach':
            f = self.need(('door', 'exit', 'crop', 'object'))
            distance = step.get('distance', 1.)
            for _ in range(2):
                yield from self.travel(offset(f.point, f.heading, -distance), explore=True)
                yield from self.look(bearing(self.here, f.point))
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
                if abs(gap-distance) <= .22 and abs(lateral) <= .25:
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
