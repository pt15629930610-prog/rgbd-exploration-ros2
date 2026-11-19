"""Language templates and RGB-D geometry for the three-exit simulation."""

import json
import math
import re

import numpy as np

from .frontier_planner import astar, reachable_cells


def parse_task(text):
    match = re.fullmatch(
        r'(?:请)?(?:前往|抵达|去|到达|到)?(对面|前方|左边|左侧|右边|右侧)(?:的)?(?:门|出口)'
        r'(?:[，,]?(?:在)?(?:它的?|他的?|其)?(?:门)?(?:正)?前(?:方)?(?:大约|大概|约)?'
        r'([0-9]+(?:\.[0-9]+)?)(米|厘米|m|cm)(?:处)?(?:停下|停车|停止)?)?[。！!]?',
        ''.join(text.split()))
    if not match:
        raise ValueError('示例：去左边的门，在门前1.2米停下')
    side, number, unit = match.groups()
    distance = float(number) if number else 1.
    if unit in ('厘米', 'cm'):
        distance /= 100
    if not 0.7 <= distance <= 3.0:
        raise ValueError('当前场景支持车体中心距门面 0.7 至 3.0 米')
    angle = math.pi/2 if side.startswith('左') else -math.pi/2 if side.startswith('右') else 0.
    return angle, distance


def door_box(answer):
    raw = answer.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = raw.partition('\n')[2][:-3].strip()
    box = json.loads(raw)
    if box == []:
        return None
    if isinstance(box, list) and box and isinstance(box[0], dict):
        box = box[0]
    if isinstance(box, dict):
        box = box.get('bbox_2d')
    if not isinstance(box, list) or len(box) != 4 or not all(
            type(v) in (float, int) and 0 <= v <= 1000 for v in box):
        raise ValueError('invalid door box')
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('empty door box')
    return box


def locate_panel(box, depth, fx, cx, pose, direction):
    """Check depth across the closed door against the requested wall normal."""
    h, w = depth.shape
    left, top, right, bottom = np.array(box) * (w, h, w, h) / 1000
    u = np.linspace(left*.8+right*.2, left*.2+right*.8, 9).astype(int)
    v = np.linspace(top*.75+bottom*.25, top*.25+bottom*.75, 5).astype(int)
    z = depth[np.ix_(v, u)]
    valid = np.isfinite(z) & (z > .1) & (z < 4.99)
    if valid.sum() < z.size * .7:
        return None
    z_left = (cx-u) * z / fx
    yaw = pose[2]
    x = pose[0] + z * math.cos(yaw) - z_left * math.sin(yaw)
    y = pose[1] + z * math.sin(yaw) + z_left * math.cos(yaw)
    nx, ny = math.cos(direction), math.sin(direction)
    normal = (x*nx + y*ny)[valid]
    along = (-x*ny + y*nx)[valid]
    # A different wall, an oblique occluder or a frame spanning mixed depths
    # cannot reliably determine the requested door's parking point.
    if np.ptp(np.percentile(normal, (10, 90))) > .14 or np.ptp(along) < .16:
        return None
    return np.array([np.median(x[valid]), np.median(y[valid])])


def search_path(walk, frontier, start, resolution, direction, visited, hint=None):
    rows, cols = np.where(frontier & reachable_cells(walk, start))
    if not rows.size:
        return []
    dy, dx = (rows-start[0])*resolution, (cols-start[1])*resolution
    score = dx*math.cos(direction) + dy*math.sin(direction) - .15*np.hypot(dx, dy)
    if hint is not None:
        score = -np.hypot(rows-hint[0], cols-hint[1])*resolution
    eligible = np.hypot(dx, dy) > .55
    for y, x in visited:
        eligible &= np.hypot(rows-y, cols-x)*resolution > .65
    indices = np.flatnonzero(eligible)
    if not indices.size:
        return []
    i = indices[np.argmax(score[indices])]
    path = astar(walk, start, (int(rows[i]), int(cols[i])))
    # Observe again after a short approach even on a long open aisle.
    for i, cell in enumerate(path):
        if math.dist(start, cell)*resolution >= 1.6:
            return path[:i+1]
    return path
