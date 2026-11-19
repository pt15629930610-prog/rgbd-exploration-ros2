"""Frontier selection and collision-checked paths on an odom-aligned grid.

Cells are (row, column), with row increasing along odom +Y.
"""

from collections import deque
from heapq import heappop, heappush
import math

import numpy as np


# Room for the four wheels, tracking error and half-cell quantization.
CLEARANCE_M = 0.55
CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def traversable(grid, resolution, start=None, *, clearance=CLEARANCE_M):
    radius = math.ceil(clearance / resolution)
    occupied = np.pad(grid >= 50, radius, constant_values=True)
    height, width = grid.shape
    blocked = np.zeros_like(grid, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                y, x = radius + dy, radius + dx
                blocked |= occupied[y:y + height, x:x + width]
    walk = (grid == 0) & ~blocked
    # Pose quantization or new observations can place the robot one cell
    # inside the planning margin. Connect only its measured free cell to a
    # cardinal free neighbour; keep the footprint clearance even at startup.
    if start is not None and inside(walk, start) and not walk[start] and grid[start] == 0:
        y, x = start
        if CLEARANCE_M - resolution >= 0.45:
            walk[start] = any(inside(walk, (y+dy, x+dx)) and walk[y+dy, x+dx]
                              for dy, dx in CARDINAL)
    return walk


def frontiers(grid):
    # Ignore isolated unknown pinholes left by depth sampling.
    unknown = np.pad(grid < 0, 1)
    h, w = grid.shape
    density = np.zeros((h, w), dtype=np.uint8)
    for y in range(3):
        for x in range(3):
            density += unknown[y:y + h, x:x + w]
    substantial = np.pad((grid < 0) & (density >= 5), 1)
    return (grid == 0) & (
        substantial[:-2, 1:-1] | substantial[2:, 1:-1]
        | substantial[1:-1, :-2] | substantial[1:-1, 2:]
    )


def inside(mask, cell):
    y, x = cell
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]


def reachable_cells(walk, start):
    seen = np.zeros_like(walk)
    if not inside(walk, start) or not walk[start]:
        return seen
    seen[start] = True
    todo = deque([start])
    while todo:
        y, x = todo.popleft()
        for dy, dx in CARDINAL:
            cell = (y + dy, x + dx)
            if inside(walk, cell) and walk[cell] and not seen[cell]:
                seen[cell] = True
                todo.append(cell)
    return seen


def astar(walk, start, goal, *, roomy=None):
    if not all(inside(walk, c) and walk[c] for c in (start, goal)):
        return []
    cost = {start: 0.0}
    parent = {}
    queue = [(math.dist(start, goal), 0.0, start)]
    while queue:
        _, distance, cell = heappop(queue)
        if distance > cost[cell]:
            continue
        if cell == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1]
        y, x = cell
        for dy, dx in CARDINAL + DIAGONAL:
            nxt = y + dy, x + dx
            if not inside(walk, nxt) or not walk[nxt]:
                continue
            if dy and dx and not (walk[y + dy, x] and walk[y, x + dx]):
                continue
            new_cost = distance + (math.sqrt(2) if dy and dx else 1)
            if roomy is not None and not roomy[nxt]:
                new_cost += 3
            if new_cost < cost.get(nxt, math.inf):
                cost[nxt] = new_cost
                parent[nxt] = cell
                heappush(queue, (new_cost + math.dist(nxt, goal), new_cost, nxt))
    return []


def choose_path(walk, frontier, start, resolution, scanned, failed):
    """Compare reachable frontier groups by A* length; return path in cells."""
    eligible = frontier & reachable_cells(walk, start)
    rows, cols = np.ogrid[:walk.shape[0], :walk.shape[1]]
    for y, x in [*scanned, *failed]:
        eligible &= (rows - y)**2 + (cols - x)**2 > (0.55 / resolution)**2
    pending = set(zip(*np.nonzero(eligible)))
    candidates = []
    while pending:
        group = [pending.pop()]
        for y, x in group:
            for dy, dx in CARDINAL + DIAGONAL:
                nxt = y + dy, x + dx
                if nxt in pending:
                    pending.remove(nxt)
                    group.append(nxt)
        if len(group) < max(3, math.ceil(0.20 / resolution)):
            continue
        center = np.mean(group, axis=0)
        goal = min(group, key=lambda c: (math.dist(c, center), c))
        candidates.append((math.dist(start, goal), -len(group), goal))

    best, best_cost = [], math.inf
    for lower_bound, _, goal in sorted(candidates):
        if lower_bound >= best_cost:
            break
        path = astar(walk, start, goal)
        if path:
            cost = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
            if cost < best_cost:
                best, best_cost = path, cost
    return best


def obstacle_distance(grid, cell, resolution, limit):
    """Nearest occupied cell or map edge, capped at the requested distance."""
    y, x = cell
    h, w = grid.shape
    radius = math.ceil(limit / resolution)
    top, left = max(0, y-radius), max(0, x-radius)
    window = grid[top:y+radius+1, left:x+radius+1]
    rows, cols = np.nonzero(window >= 50)
    edge = min(y+0.5, x+0.5, h-y-0.5, w-x-0.5) * resolution
    nearest = np.hypot(rows + top - y, cols + left - x).min() * resolution if rows.size else limit
    return min(limit, edge, nearest)


def observation_path(path, grid, resolution):
    """Prefer room to turn within the last 30 cm of the verified approach."""
    desired = CLEARANCE_M + 0.20
    best = len(path) - 1
    clearance = obstacle_distance(grid, path[best], resolution, desired)
    distance = 0.0
    for i in range(best - 1, 0, -1):
        distance += math.dist(path[i], path[i+1]) * resolution
        if distance > 0.30 or clearance >= desired:
            break
        candidate = obstacle_distance(grid, path[i], resolution, desired)
        if candidate > clearance:
            best, clearance = i, candidate
    return path[:best+1]


def clear_segment(walk, start, goal):
    """Supercover of a line between cell centers, including corner contacts."""
    if not inside(walk, start) or not inside(walk, goal) or not walk[start]:
        return False
    y, x = start
    dy, dx = goal[0] - y, goal[1] - x
    sy, sx = int(np.sign(dy)), int(np.sign(dx))
    ny, nx = abs(dy), abs(dx)
    iy = ix = 0
    while iy < ny or ix < nx:
        delta = (2 * ix + 1) * ny - (2 * iy + 1) * nx
        if delta == 0 and not (walk[y, x + sx] and walk[y + sy, x]):
            return False
        if delta <= 0:
            x += sx
            ix += 1
        if delta >= 0:
            y += sy
            iy += 1
        if not walk[y, x]:
            return False
    return True
