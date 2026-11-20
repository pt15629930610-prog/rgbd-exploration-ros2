"""Surveyed aisle topology. Coordinates are metres forward and left of the entrance spawn."""

from dataclasses import dataclass
from heapq import heappop, heappush
import math


ENTRANCE = (0., 0.)
ENTRANCE_POSE = (-6., 0., 0.)  # Spawn in the world's odom frame, including after a node restart.
CROSS = (6., 0.)
FRONT = 11.94
ROWS = (3.11, 6.21)
OUTER = 9.66
RIGHT_SILL = (6., -11.)
OUTSIDE = (6., -12.35)
SIDES = {'left': '左', 'right': '右'}


def to_world(point, entrance):
    u, v = point
    x, y, yaw = entrance
    c, s = math.cos(yaw), math.sin(yaw)
    return x + u*c - v*s, y + u*s + v*c


def from_world(point, entrance):
    dx, dy = point[0]-entrance[0], point[1]-entrance[1]
    c, s = math.cos(entrance[2]), math.sin(entrance[2])
    return dx*c + dy*s, -dx*s + dy*c


def project(point, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    t = ((point[0]-a[0])*dx + (point[1]-a[1])*dy) / (dx*dx + dy*dy)
    t = min(1., max(0., t))
    return a[0] + dx*t, a[1] + dy*t


def aisle_graph():
    graph = {}

    def link(a, b):
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)

    lanes = (-OUTER, -ROWS[1], -ROWS[0], 0., *ROWS, OUTER)
    for x in (0., CROSS[0], FRONT):
        for a, b in zip(lanes, lanes[1:]):
            link((x, a), (x, b))
    for y in lanes:
        # The two irrigation tanks close different ends of two aisles.
        link((0., y), (4.05 if y == ROWS[0] else CROSS[0], y))
        link((CROSS[0], y), (10.02 if y == -ROWS[1] else FRONT, y))
    link((CROSS[0], -OUTER), RIGHT_SILL)
    link(RIGHT_SILL, OUTSIDE)
    return graph


def shortest(graph, start, end):
    queue, costs, parent = [(0., start)], {start: 0.}, {}
    while queue:
        cost, node = heappop(queue)
        if node == end:
            path = []
            while node != start:
                path.append(node)
                node = parent[node]
            return path[::-1]
        if cost > costs[node]:
            continue
        for other in graph[node]:
            total = cost + math.dist(other, node)
            if total < costs.get(other, math.inf):
                parent[other], costs[other] = node, total
                heappush(queue, (total, other))
    raise ValueError('道路简图中没有连通的路线')


@dataclass(frozen=True)
class Checkpoint:
    point: tuple
    label: str
    wait: int = 0
    outside: bool = False


def compile_route(steps, position=ENTRANCE):
    """Bind the whole request before moving. A loop must traverse four distinct road edges."""
    graph = aisle_graph()
    edges = ((a, b) for a in graph for b in graph[a] if a < b)
    a, b = min(edges, key=lambda e: math.dist(position, project(position, *e)))
    joint = project(position, a, b)
    if math.dist(position, joint) > .60:
        raise ValueError('当前位置偏离道路简图，请从入口或通道内开始')
    # Attach a replacement instruction to the actual aisle occupied by the robot.
    current = min((a, b), key=lambda n: math.dist(joint, n))
    path = []
    if math.dist(joint, current) > .05:
        current = joint
        graph[current] = {a, b}
        graph[a].add(current)
        graph[b].add(current)
    join_aisle = math.dist(position, current) > .12
    side = None

    def go(point, label, *, outside=False):
        nonlocal current
        for waypoint in shortest(graph, current, point):
            path.append(Checkpoint(waypoint, label, outside=outside and waypoint == point))
        current = point

    for step in steps:
        op = step['do']
        if op == 'wait':
            stopped_at = path[-1].point if path else position
            path.append(Checkpoint(stopped_at, f'停留 {step["seconds"]} 秒', wait=step['seconds']))
            continue
        if join_aisle:
            path.append(Checkpoint(current, '回到当前通道中心'))
            join_aisle = False
        if op == 'turn':
            if step['at'] != 1:
                raise ValueError('沿入口主路只有一个十字路口，请改用第一个')
            side = 1 if step['side'] == 'left' else -1
            go(CROSS, '到达入口正前方第一个十字路口')
            go((CROSS[0], side*ROWS[0]), '向入口' + SIDES[step['side']] + '侧拐入横道')
        elif op == 'loop':
            if side is None:
                raise ValueError('请说明先从十字路口向入口左侧还是右侧转，再进入作物行')
            row = step['row']
            if row > len(ROWS):
                raise ValueError('这一侧有两条行间通道，请指定第一条或第二条')
            y = side*ROWS[row-1]
            inner = 0. if row == 1 else side*ROWS[row-2]
            gate = (CROSS[0], y)
            choices = {}
            for name, far in [('front', FRONT), ('back', 0.)]:
                ring = [gate, (far, y), (far, inner), (CROSS[0], inner), gate]
                if all(v in graph[u] for u, v in zip(ring, ring[1:])):
                    choices[name] = ring
            toward = step['toward']
            if toward == 'auto':
                if not choices:
                    raise ValueError('该行的两端都无法闭环，不能绕一圈回到行口')
                if len(choices) > 1:
                    raise ValueError('该行的进入方向不唯一，请明确向入口前方还是后方绕行')
                toward = next(iter(choices))
            if toward not in choices:
                raise ValueError('这一方向有封堵，无法绕一圈回到行口')
            go(gate, f'到第{row}条作物行入口')
            for i, vertex in enumerate(choices[toward][1:], 1):
                path.append(Checkpoint(vertex, f'第{row}条作物行绕行 {i}/4' if i < 4
                                       else f'第{row}条作物行闭环，回到行口'))
            current = gate
        elif op == 'exit':
            if step['side'] != 'right':
                raise ValueError('当前只有入口右侧的门是开的，请更换出口')
            if current[1] < RIGHT_SILL[1]:
                raise ValueError('小车已在棚外，不能从棚内再次穿门')
            go((CROSS[0], -OUTER), '前往右门内侧')
            go(RIGHT_SILL, '穿过右门门槛')
            go(OUTSIDE, '从右门抵达棚外', outside=True)
            side = None
        elif op == 'home':
            go(ENTRANCE, '返回原入口')
            side = None
        else:
            raise ValueError('未知路线动作')
    return path
