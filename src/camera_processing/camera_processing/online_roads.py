"""Road observations on a growing occupancy grid. All positions are in odom metres."""

from dataclasses import dataclass
from functools import cached_property
import math

import cv2
import numpy as np

from .aisle_tracking import observed_path
from .frontier_planner import astar, frontiers, inside, reachable_cells


@dataclass(frozen=True)
class Junction:
    point: tuple
    branches: frozenset
    crop_sides: frozenset  # Parallel boundaries only; crop identity needs visual evidence.


class JunctionTracker:
    """Confirm each branch on two successive map stamps. Use one tracker per approach axis."""

    def __init__(self):
        self.previous = []
        self.confirmed = []
        self.stamp = None

    def update(self, observations, stamp):
        if stamp == self.stamp:
            return self.confirmed
        self.confirmed = []
        previous = self.previous if self.stamp is None or stamp > self.stamp else []
        for here in observations:
            old = min(previous, key=lambda j: math.dist(j.point, here.point), default=None)
            if old is None or math.dist(old.point, here.point) > .65:
                continue
            common = old.branches & here.branches
            if common & {'left', 'right'}:
                middle = tuple((a+b)/2 for a, b in zip(old.point, here.point))
                self.confirmed.append(Junction(middle, common, old.crop_sides & here.crop_sides))
        self.previous = list(observations)
        self.stamp = stamp
        return self.confirmed


def offset(point, yaw, along, across=0.):
    c, s = math.cos(yaw), math.sin(yaw)
    return point[0]+along*c-across*s, point[1]+along*s+across*c


def runs(mask):
    """Contiguous true intervals, inclusive at both ends."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)-1)


class RoadView:
    def __init__(self, grid, walk, origin, resolution):
        self.grid = grid
        self.walk = walk
        self.origin = origin
        self.resolution = resolution

    def cell(self, point):
        return (math.floor((point[1]-self.origin[1])/self.resolution),
                math.floor((point[0]-self.origin[0])/self.resolution))

    def xy(self, cell):
        return (self.origin[0]+(cell[1]+.5)*self.resolution,
                self.origin[1]+(cell[0]+.5)*self.resolution)

    def _sample(self, point, yaw, along, across):
        """Project samples into the map. Outside the grid has exactly the semantics of unknown."""
        x, y = offset(point, yaw, along, across)
        rows = np.floor((y-self.origin[1])/self.resolution).astype(int)
        cols = np.floor((x-self.origin[0])/self.resolution).astype(int)
        h, w = self.grid.shape
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        rows, cols = rows.clip(0, h-1), cols.clip(0, w-1)
        return np.where(valid, self.grid[rows, cols], -1), valid & self.walk[rows, cols]

    def clearance(self, point, yaw, limit=2.):
        samples = np.linspace(0., limit, max(2, math.ceil(limit/(self.resolution*.5))+1))
        _, safe = self._sample(point, yaw, samples, 0.)
        stops = np.flatnonzero(~safe)
        return float(samples[max(0, stops[0]-1)]) if stops.size else limit

    def footprint_clear(self, point, yaw, *, known):
        # SDF wheel envelope: 0.34 x 0.275 m half-extents. Pad for cell
        # quantization, sampling between poses and small tracking errors.
        pad = self.resolution*.71 + .045
        length, width = .34+pad, .275+pad
        r, c = self.cell(point)
        radius = math.ceil(math.hypot(length, width)/self.resolution)+1
        rows = np.arange(r-radius, r+radius+1)
        cols = np.arange(c-radius, c+radius+1)
        x = self.origin[0]+(cols[None, :]+.5)*self.resolution-point[0]
        y = self.origin[1]+(rows[:, None]+.5)*self.resolution-point[1]
        co, si = math.cos(yaw), math.sin(yaw)
        body = (abs(co*x+si*y) <= length) & (abs(co*y-si*x) <= width)
        rr, cc = np.nonzero(body)
        rr, cc = rows[rr], cols[cc]
        if rr.min()<0 or cc.min()<0 or rr.max()>=self.grid.shape[0] or cc.max()>=self.grid.shape[1]:
            return False
        cells = self.grid[rr, cc]
        return bool(np.all(cells == 0 if known else cells < 50))

    def straight_clear(self, point, yaw, distance):
        samples = max(1, math.ceil(abs(distance)/self.resolution))
        return all(self.footprint_clear(offset(point, yaw, distance*i/samples), yaw, known=True)
                   for i in range(samples+1))

    def observed_dead_end(self, point, yaw, *, banks=True):
        """Require a measured cap across the chassis and banks on both sides."""
        def bank(angle, limit, offsets):
            distances = np.arange(.10, limit, self.resolution/2)
            raw, _ = self._sample(point, angle, distances[:, None], np.array(offsets)[None, :])
            return (raw >= 50).any(axis=0).all()
        return (bank(yaw, 1.35, [-.25, 0., .25])
                and (not banks or all(bank(yaw+sign*math.pi/2, 1.95, [-.25, 0., .25]) for sign in (-1, 1))))

    def _profile(self, point, yaw, horizon, sign):
        # Examine both road shoulders, including a short stretch behind the robot.
        step = max(.07, self.resolution)
        along = np.arange(-2.8, horizon+2.8+step/2, step)
        across = np.arange(0., 3.3, self.resolution*.65)
        raw, safe = self._sample(point, yaw, along[:, None], sign*across[None, :])
        shoulder = (raw >= 50) & (across >= .28)
        wall_distance = np.where(shoulder, across, np.inf).min(axis=1)
        reach = np.where(~safe, across, np.inf).min(axis=1)
        return along, wall_distance, reach

    @staticmethod
    def _opening_mask(along, wall_distance, reach):
        # An open room or an unseen shoulder alone does not establish a branch.
        # Require an observed transition from a bounded aisle to a deeper side opening.
        reference = np.full(along.shape, np.inf)
        for i, at in enumerate(along):
            separation = abs(along-at)
            nearby = wall_distance[(separation > .35) & (separation <= 2.7)]
            if nearby.size:
                reference[i] = nearby.min()
        depth = np.maximum(1.30, reference+.55)
        return reference, np.isfinite(reference) & (reach >= depth)

    def _paired_edges(self, point, yaw):
        # Search for two observed banks bounding the *outgoing* lane. Two samples
        # spaced along its length distinguish a flanked passage from a single jamb.
        ranges = np.arange(.28, 1.90, max(.045, self.resolution))
        depths = np.arange(1.3, 2.71, .25)
        edges = []
        for sign in (1, -1):
            raw, _ = self._sample(point, yaw, depths[:, None], sign*ranges[None, :])
            edges.append(np.where(raw >= 50, ranges, np.inf).min(axis=1))
        a, b = edges
        paired = np.flatnonzero(np.isfinite(a+b) & (a+b >= 1.12) & (a+b <= 3.3))
        for i in paired:
            later = paired[depths[paired]-depths[i] >= .48]
            if np.any((abs(a[later]-a[i]) < .35) & (abs(b[later]-b[i]) < .35)):
                return True
        return False

    def junctions(self, point, yaw, horizon=3.):
        """Detect branches along an observed aisle, sorted along the approach direction."""
        sides = []
        for name, sign in [('left', 1), ('right', -1)]:
            along, walls, reach = self._profile(point, yaw, horizon, sign)
            _, openings = self._opening_mask(along, walls, reach)
            for lo, hi in runs(openings):
                if along[hi]-along[lo] < .12:
                    continue
                middle = (along[hi]+along[lo])/2
                if -.40 <= middle <= horizon:
                    sides.append((middle, name, sign))
        grouped = []
        for at, name, sign in sorted(sides):
            if not grouped or at-grouped[-1][0][0] > .6:
                grouped.append([])
            grouped[-1].append((at, name, sign))
        found = []
        for group in grouped:
            at = sum(d for d, _, _ in group)/len(group)
            center = offset(point, yaw, at)
            # A disconnected corridor beyond a wall is not the next junction on this road.
            direction = yaw if at >= 0 else yaw+math.pi
            if self.clearance(point, direction, abs(at)) < abs(at)-self.resolution:
                continue
            branches = {name for _, name, _ in group}
            for name, angle in [('forward', yaw), ('back', yaw+math.pi)]:
                if self.clearance(center, angle, 1.25) >= 1.14:
                    branches.add(name)
            crops = {name for _, name, sign in group if self._paired_edges(center, yaw+sign*math.pi/2)}
            found.append(Junction(center, frozenset(branches), frozenset(crops)))
        return found

    def probe_sides(self, point, yaw):
        """Suggest a side view when a previously observed near shoulder disappears nearby."""
        result = set()
        for name, sign in [('left', 1), ('right', -1)]:
            along, walls, reach = self._profile(point, yaw, .8, sign)
            reference, open_mask = self._opening_mask(along, walls, reach)
            near = (along >= -.12) & (along <= .48)
            missing = (reference <= 1.8) & (walls > reference+.45)
            if np.count_nonzero(missing & near) >= 2 and not np.any(near & open_mask):
                result.add(name)
        return frozenset(result)

    def advance(self, point, yaw, distance=1.1):
        """Select a reachable prefix which makes forward progress within the local road."""
        if distance <= .01:
            return []
        distance = min(1.15, distance)
        target = offset(point, yaw, distance)
        return observed_path(self.walk, self.origin, self.resolution, point, point, target)

    def path_to(self, point, goal):
        return astar(self.walk, self.cell(point), self.cell(goal))

    def replan_after_turn(self, point, goal, failed):
        start, end = self.cell(point), self.cell(goal)
        walk = self.walk.copy()
        radius = math.ceil(.25/self.resolution)
        for site in failed:
            r, c = self.cell(site)
            if math.dist((r,c), start) <= radius or math.dist((r,c), end) <= radius:
                continue
            cv2.circle(walk.view(np.uint8), (c,r), radius, 0, -1)
        return (astar(walk, start, end, roomy=self.turning_room)
                or astar(self.walk, start, end, roomy=self.turning_room))

    @cached_property
    def clearance_map(self):
        free = np.pad(self.grid < 50, 1).astype(np.uint8)
        return cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]*self.resolution

    @cached_property
    def turning_room(self):
        return self.walk & (self.clearance_map >= .78)

    def frontier_path(self, point, direction=None, visited=(), target=None, *, room_to_turn=False):
        """Rank reachable frontier groups by proximity, direction and optional observed target."""
        start = self.cell(point)
        if not inside(self.walk, start) or not self.walk[start]:
            return []
        frontier = frontiers(self.grid) & reachable_cells(self.walk, start)
        for pose in visited:
            r, c = self.cell(pose)
            radius = math.ceil(.65/self.resolution)
            top, left = max(0, r-radius), max(0, c-radius)
            chunk = frontier[top:r+radius+1, left:c+radius+1]
            rr, cc = np.ogrid[:chunk.shape[0], :chunk.shape[1]]
            chunk[(rr+top-r)**2+(cc+left-c)**2 <= radius*radius] = False
        remaining = set(zip(*np.nonzero(frontier)))
        candidates = []
        while remaining:
            group = [remaining.pop()]
            for r, c in group:
                for neighbour in ((r+1,c), (r-1,c), (r,c+1), (r,c-1)):
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        group.append(neighbour)
            if len(group)*self.resolution < .18:
                continue
            # Long FOV frontiers can span many metres. Rank actual accessible
            # cells rather than discarding the requested side of a long group.
            for cell in group:
                x, y = self.xy(cell)
                dx, dy = x-point[0], y-point[1]
                distance = math.hypot(dx, dy)
                if distance < .22:
                    continue
                score = distance
                if direction is not None:
                    progress = dx*math.cos(direction)+dy*math.sin(direction)
                    alignment = progress/distance
                    score += 3.5*(1-alignment)
                    if room_to_turn:
                        # A nearby frontier behind the robot should not repeatedly
                        # outrank a reachable frontier farther into the requested side.
                        score -= 1.35*progress
                if target is not None:
                    score += 1.6*math.dist((x, y), target)
                if room_to_turn:
                    score += 12.*max(0., .85-self.clearance_map[cell])
                candidates.append((score, cell))
        if not candidates:
            return []
        return astar(self.walk, start, min(candidates)[1], roomy=self.turning_room if room_to_turn else None)
