"""Build the local, untextured COLLADA assets for the four-door greenhouse.

Run with Python and NumPy when editing the models. Gazebo loads the exported
meshes directly; each complete planting bed is one instanced mesh.
"""

from collections import defaultdict
from math import cos, pi, sin
from pathlib import Path
from random import Random
from xml.etree import ElementTree as ET

import numpy as np


OUTPUT = Path(__file__).resolve().parents[1] / 'src/camera_processing/worlds/meshes'
# Eaves and ridge are above the RGB-D camera's vertical field within its 5 m range.
HALF_DEPTH, HALF_WIDTH, EAVE, RIDGE = 7., 11., 3.40, 5.10
COLORS = {
    'steel': (.55, .63, .62), 'dark_steel': (.27, .34, .34),
    'trough': (.51, .57, .49), 'slate': (.23, .29, .26), 'soil': (.16, .115, .085),
    'hose': (.12, .16, .15), 'clay': (.60, .32, .21), 'clay_rim': (.72, .43, .29),
    'nursery_pot': (.24, .32, .29), 'stem': (.36, .47, .16),
    'leaf': (.16, .40, .16), 'leaf_light': (.28, .52, .19),
    'basil': (.21, .46, .25), 'basil_light': (.34, .56, .28),
    'ripe': (.82, .085, .075), 'blush': (.96, .31, .17),
    'seed': (.93, .72, .39), 'petal': (.96, .95, .84), 'pollen': (.99, .73, .19),
    'film': (.79, .89, .90),
}


class Mesh:
    def __init__(self):
        self.vertices = {}
        self.normals = {}
        self.faces = defaultdict(list)

    @staticmethod
    def index(table, vector):
        key = tuple(round(float(component), 5) for component in vector)
        return table.setdefault(key, len(table))

    def face(self, material, points, *, two_sided=False):
        points = np.asarray(points)
        first = points[0]
        for a, b in zip(points[1:-1], points[2:]):
            normal = np.cross(a - first, b - first)
            length = np.linalg.norm(normal)
            if length < 1e-10:
                continue
            ids = [self.index(self.vertices, p) for p in (first, a, b)]
            normal_id = self.index(self.normals, normal / length)
            self.faces[material].append(tuple((v, normal_id) for v in ids))
            if two_sided:
                back = self.index(self.normals, -normal / length)
                self.faces[material].append(tuple((v, back) for v in reversed(ids)))

    def join_rings(self, material, rings, caps=True):
        for lower, upper in zip(rings, rings[1:]):
            for i in range(len(lower)):
                j = (i + 1) % len(lower)
                self.face(material, [lower[i], lower[j], upper[j], upper[i]])
        if caps:
            self.face(material, rings[0][::-1])
            self.face(material, rings[-1])

    def beam(self, material, start, end, width, thickness=None):
        start, end = np.asarray(start), np.asarray(end)
        axis = (end - start) / np.linalg.norm(end - start)
        guide = (0, 0, 1) if abs(axis[2]) < .95 else (0, 1, 0)
        side = np.cross(guide, axis)
        side /= np.linalg.norm(side)
        up = np.cross(axis, side) * (thickness or width) / 2
        side *= width / 2
        corners = [-side-up, side-up, side+up, -side+up]
        self.join_rings(material, [[p + c for c in corners] for p in (start, end)])

    def box(self, material, center, size):
        x, y, z = center
        length, width, height = size
        self.beam(material, (x-length/2, y, z), (x+length/2, y, z), width, height)

    def lathe(self, material, x, y, profile, sides=12):
        rings = [[(x + r*cos(2*pi*i/sides), y + r*sin(2*pi*i/sides), z)
                  for i in range(sides)] for z, r in profile]
        self.join_rings(material, rings)

    def write(self, name):
        def rows(items, per_line):
            data = list(items)
            return '\n' + '\n'.join(' '.join(data[i:i+per_line]) for i in range(0, len(data), per_line))

        sub = ET.SubElement
        root = ET.Element('COLLADA', xmlns='http://www.collada.org/2005/11/COLLADASchema',
                          version='1.4.1')
        asset = sub(root, 'asset')
        for tag in ('created', 'modified'):
            sub(asset, tag).text = '2026-11-20T00:00:00Z'
        sub(asset, 'unit', name='meter', meter='1')
        sub(asset, 'up_axis').text = 'Z_UP'
        effects, materials = sub(root, 'library_effects'), sub(root, 'library_materials')
        geom = sub(sub(root, 'library_geometries'), 'geometry', id=name, name=name)
        mesh = sub(geom, 'mesh')
        scene = sub(sub(root, 'library_visual_scenes'), 'visual_scene', id='main')
        inst = sub(sub(scene, 'node', id='model'), 'instance_geometry', url='#' + name)
        bindings = sub(sub(inst, 'bind_material'), 'technique_common')
        for material in self.faces:
            effect = sub(effects, 'effect', id=material + '-fx')
            technique = sub(sub(effect, 'profile_COMMON'), 'technique', sid='common')
            surface = sub(technique, 'lambert')
            rgb = COLORS[material]
            for tag, gain in [('ambient', .70), ('diffuse', 1.)]:
                sub(sub(surface, tag), 'color').text = ' '.join(f'{v*gain:.3f}' for v in rgb) + ' 1'
            mat = sub(materials, 'material', id=material, name=material)
            sub(mat, 'instance_effect', url='#' + material + '-fx')
            sub(bindings, 'instance_material', symbol=material, target='#' + material)
        for source_name, vectors in [('positions', self.vertices), ('normals', self.normals)]:
            source = sub(mesh, 'source', id=source_name)
            array = sub(source, 'float_array', id=source_name+'-data', count=str(3*len(vectors)))
            array.text = rows((' '.join(f'{v:g}' for v in vector) for vector in vectors), 8)
            accessor = sub(sub(source, 'technique_common'), 'accessor',
                           source='#'+source_name+'-data', count=str(len(vectors)), stride='3')
            for axis in 'XYZ':
                sub(accessor, 'param', name=axis, type='float')
        vertices = sub(mesh, 'vertices', id='vertices')
        sub(vertices, 'input', semantic='POSITION', source='#positions')
        for material, faces in self.faces.items():
            triangles = sub(mesh, 'triangles', material=material, count=str(len(faces)))
            sub(triangles, 'input', semantic='VERTEX', source='#vertices', offset='0')
            sub(triangles, 'input', semantic='NORMAL', source='#normals', offset='1')
            sub(triangles, 'p').text = rows((' '.join(f'{v} {n}' for v, n in face) for face in faces), 8)
        sub(sub(root, 'scene'), 'instance_visual_scene', url='#main')
        ET.indent(root)
        ET.ElementTree(root).write(OUTPUT / (name+'.dae'), encoding='utf-8', xml_declaration=True)


def leaf(mesh, start, angle, length, width, rise, materials, serrated=False):
    """Two curved blade halves meeting at a raised midrib."""
    along = np.array([cos(angle), sin(angle), 0.])
    across = np.array([-sin(angle), cos(angle), 0.])
    levels = 8 if serrated else 5
    previous = None
    for i in range(levels + 1):
        t = i / levels
        arch = sin(pi * t)
        middle = start + along*(length*t) + [0, 0, rise*t + .035*arch]
        half = width * arch**.70 * (.78 if serrated and i % 2 else 1.)
        edge = middle - [0, 0, .25*half]
        row = (edge - across*half, middle, edge + across*half)
        if previous is not None:
            for side, material in enumerate(materials):
                mesh.face(material, [previous[side], row[side], row[side+1], previous[side+1]],
                          two_sided=True)
        previous = row


def berry(mesh, x, y, top, scale, color):
    profile = [(-.113, .004), (-.085, .027), (-.044, .048), (-.016, .052), (0, .036)]
    mesh.lathe(color, x, y, [(top+z*scale, r*scale) for z, r in profile], 9)
    for i in range(6):
        a = 2*pi*i / 6
        along = np.array([cos(a), sin(a), 0.])
        across = np.array([-sin(a), cos(a), 0.])
        crown = np.array([x, y, top+.002])
        mesh.face('leaf', [crown-.014*across, crown+.014*across,
                           crown+.052*scale*along+[0, 0, -.014]], two_sided=True)
        for z, r in [(-.030, .051), (-.068, .037)]:
            seed = np.array([x, y, top+z*scale]) + r*scale*along
            mesh.face('seed', [seed-.003*across, seed+.003*across, seed+[0, 0, -.007]],
                      two_sided=True)


def strawberry_bed():
    mesh, rng = Mesh(), Random(17)
    mesh.box('slate', (0, 0, .22), (3.86, 1.12, .44))
    mesh.box('trough', (0, 0, .59), (4, 1.24, .56))
    mesh.box('soil', (0, 0, .87), (3.88, 1.16, .035))
    mesh.beam('hose', (-1.94, 0, .895), (1.94, 0, .895), .030)
    for sign in (-1, 1):
        mesh.box('steel', (0, sign*.617, .86), (4.01, .043, .064))
        for i in range(6):
            x = -1.62 + i*.648
            base = np.array([x+rng.uniform(-.025, .025), sign*.31, .883])
            for j in range(5):
                a = 2*pi*j/5 + rng.uniform(-.32, .32)
                joint = base + [.115*cos(a), .115*sin(a), rng.uniform(.12, .19)]
                mesh.beam('stem', base, joint, .012)
                for offset, length in [(-.86, .13), (0, .17), (.86, .13)]:
                    leaf(mesh, joint, a+offset, length, .057, rng.uniform(-.04, .05),
                         ('leaf', 'leaf_light'), serrated=True)
            cluster = base + [0, sign*.23, .13]
            mesh.beam('stem', base, cluster, .012)
            for k in range(3):
                tip = np.array([x+(k-1)*.09, sign*(.646-.016*abs(k-1)),
                                .935+rng.uniform(-.055, .035)])
                mesh.beam('stem', cluster, tip, .009)
                berry(mesh, *tip, rng.uniform(.76, 1.08), 'blush' if (i+k)%5 == 0 else 'ripe')
            if i % 2 == 0:
                flower = base + [.04, sign*.08, .23]
                mesh.beam('stem', base, flower, .009)
                for j in range(5):
                    a = 2*pi*j/5
                    along = np.array([cos(a), sin(a), .18])
                    across = np.array([-sin(a), cos(a), 0.])
                    mesh.face('petal', [flower, flower+.022*along-.013*across, flower+.043*along,
                                        flower+.022*along+.013*across], two_sided=True)
                mesh.lathe('pollen', *flower[:2], [(flower[2]+.003, .011), (flower[2]+.012, .009)], 7)
    return mesh


def potted_bench():
    mesh, rng = Mesh(), Random(31)
    mesh.box('slate', (0, 0, .12), (3.94, 1.20, .24))
    for x in (-1.76, 0, 1.76):
        for y in (-.51, .51):
            mesh.box('dark_steel', (x, y, .39), (.065, .065, .42))
    mesh.box('steel', (0, 0, .59), (4., 1.26, .09))
    for sign in (-1, 1):
        mesh.box('dark_steel', (0, sign*.605, .51), (3.95, .045, .12))
        for i in range(5):
            x, y = (i-2)*.74, sign*.30
            clay = (i+sign) % 3 != 0
            mesh.lathe('clay' if clay else 'nursery_pot', x, y, [(.635, .135), (.932, .20)])
            mesh.lathe('clay_rim' if clay else 'slate', x, y, [(.901, .209), (.95, .213)])
            mesh.lathe('soil', x, y, [(.951, .181), (.957, .18)])
            for stem in range(3):
                angle = 2*pi*stem/3 + rng.uniform(-.3, .3)
                base = np.array([x+.048*cos(angle), y+.048*sin(angle), .956])
                top = base + [.065*cos(angle), .065*sin(angle), rng.uniform(.31, .47)]
                mesh.beam('stem', base, top, .013)
                for n, t in enumerate((.27, .56, .83)):
                    for spin in (0, pi):
                        leaf(mesh, base+(top-base)*t, angle+n*1.45+spin,
                             .205-.052*t, .062-.02*t, .095-.07*t, ('basil', 'basil_light'))
    return mesh


def greenhouse_frame():
    mesh = Mesh()
    eave, ridge = EAVE, RIDGE

    def roof_z(x):
        return ridge - abs(x)/HALF_DEPTH * (ridge-eave)

    for y in np.linspace(-HALF_WIDTH, HALF_WIDTH, 9):
        for x in (-HALF_DEPTH, HALF_DEPTH):
            if y != 0:  # Keep the entrance and opposite door unobstructed.
                mesh.beam('steel', (x, y, .02), (x, y, eave), .085)
            mesh.beam('steel', (x, y, eave), (0, y, ridge), .08, .115)
        mesh.beam('dark_steel', (-7, y, eave), (7, y, eave), .050, .065)
        for i in range(8):
            x0, x1 = -7+i*1.75, -7+(i+1)*1.75
            mesh.beam('steel', (x0, y, eave if i%2 == 0 else roof_z(x0)),
                      (x1, y, roof_z(x1) if i%2 == 0 else eave), .038)
    for x in np.linspace(-7, 7, 9):
        mesh.beam('steel', (x, -11, roof_z(x)), (x, 11, roof_z(x)), .065, .08)
    for side in (-1, 1):
        for x in (-5.25, -3.5, -1.75, -.73, .73, 1.75, 3.5, 5.25):
            mesh.beam('steel', (x, side*11, 0), (x, side*11, roof_z(x)), .065)
        for y in (-.73, .73):
            mesh.beam('steel', (side*7, y, 0), (side*7, y, eave), .065)
        for z in (.82, 1.70, 2.32):
            for half in (-1, 1):
                stop = .70 if z < 2.1 else 0.
                mesh.beam('steel', (side*7, half*stop, z), (side*7, half*11, z), .065)
                mesh.beam('steel', (half*stop, side*11, z), (half*7, side*11, z), .065)
    return mesh


def greenhouse_cover():
    mesh = Mesh()
    for side in (-1, 1):
        x = side * HALF_DEPTH
        mesh.face('film', [(x, -HALF_WIDTH, EAVE), (x, HALF_WIDTH, EAVE),
                           (0, HALF_WIDTH, RIDGE), (0, -HALF_WIDTH, RIDGE)], two_sided=True)
        y = side * HALF_WIDTH
        mesh.face('film', [(-HALF_DEPTH, y, EAVE), (HALF_DEPTH, y, EAVE), (0, y, RIDGE)],
                  two_sided=True)
    return mesh


if __name__ == '__main__':
    OUTPUT.mkdir(parents=True, exist_ok=True)
    strawberry_bed().write('strawberry_bed')
    potted_bench().write('potted_bench')
    greenhouse_frame().write('greenhouse_frame')
    greenhouse_cover().write('greenhouse_cover')
