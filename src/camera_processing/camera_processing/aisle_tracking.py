"""Short A* paths on the observed occupancy grid, constrained to the active road."""

import math

import numpy as np

from .frontier_planner import astar, inside, reachable_cells


ARRIVAL_M = .22
ROAD_BAND_M = .55


def observed_path(walk, origin, resolution, position, road_start, target):
    """Return a reachable prefix toward the next checkpoint, never through an unknown cell."""
    row = math.floor((position[1]-origin[1])/resolution)
    col = math.floor((position[0]-origin[0])/resolution)
    if not inside(walk, (row, col)) or not walk[row, col]:
        return []
    dx, dy = target[0]-road_start[0], target[1]-road_start[1]
    length = math.hypot(dx, dy)
    if length < .01:
        return []
    dx, dy = dx/length, dy/length
    radius = math.ceil(1.65/resolution)
    top, left = max(0, row-radius), max(0, col-radius)
    local = walk[top:row+radius+1, left:col+radius+1]
    yy, xx = np.ogrid[:local.shape[0], :local.shape[1]]
    x = origin[0]+(left+xx+.5)*resolution-road_start[0]
    y = origin[1]+(top+yy+.5)*resolution-road_start[1]
    along, across = x*dx+y*dy, -x*dy+y*dx
    corridor = local & (abs(across) <= ROAD_BAND_M) & (along >= -.35) & (along <= length+.12)
    start = (row-top, col-left)
    p = (position[0]-road_start[0])*dx+(position[1]-road_start[1])*dy
    ahead = min(length, max(0., p)+1.15)
    connected = reachable_cells(corridor, start)
    rows, cols = np.nonzero(connected)
    if not rows.size:
        return []
    cost = (along[rows, cols]-ahead)**2 + 2.5*across[rows, cols]**2
    i = int(np.argmin(cost))
    end = (int(rows[i]), int(cols[i]))
    dest = (origin[0]+(left+end[1]+.5)*resolution, origin[1]+(top+end[0]+.5)*resolution)
    if math.dist(dest, target) > ARRIVAL_M and math.dist(position, target)-math.dist(dest, target) < .16:
        return []
    path = astar(corridor, start, end, roomy=abs(across) < .23)
    return [(top+r, left+c) for r, c in path]


def crossed_right_door(previous, current, sill, clearance=.42):
    """Interpolate a measured outward crossing, clear of both jambs and with continuous odometry."""
    if previous is None or not previous[1] > sill[1] >= current[1] or math.dist(previous, current) > .5:
        return False
    fraction = (sill[1]-previous[1])/(current[1]-previous[1])
    crossing_x = previous[0] + fraction*(current[0]-previous[0])
    return abs(crossing_x-sill[0]) <= clearance
