"""Ground visual targets in synchronized RGB-D observations, independent of greenhouse layout."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math

import cv2
import numpy as np

from .door_task import door_box


@dataclass
class Evidence:
    stamp: float
    pose: tuple
    kind: str
    point: tuple | None
    heading: float | None
    detail: str
    jpeg: bytes
    error: bool = False
    panel: tuple | None = None  # Observation hint only; never a verified exit.


def world_point(u, z, fx, cx, pose):
    left = (cx-u)*z/fx
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + c*z - s*left, y + s*z + c*left)


def locate_visible(box, depth, fx, cx, pose):
    """Estimate a foreground surface from coherent measurements inside the visual box."""
    l, t, r, b = box
    columns = np.linspace(l+.22*(r-l), l+.78*(r-l), 11).astype(int)
    rows = np.linspace(t+.27*(b-t), t+.72*(b-t), 7).astype(int)
    samples = depth[np.ix_(rows, columns)]
    valid = np.isfinite(samples) & (samples > .10) & (samples < 4.99)
    if valid.sum() < 20:
        return None
    z0 = np.percentile(samples[valid], 35)
    select = valid & (np.abs(samples-z0) < .25)
    if select.sum() < 20 or select.any(axis=0).sum() < 4:
        return None
    return world_point(float(np.median(columns[np.nonzero(select)[1]])),
                       float(np.median(samples[select])), fx, cx, pose)


def locate_door_face(box, depth, fx, cx, pose):
    """Estimate the face normal from depth across a panel, including an oblique view."""
    l, t, r, b = box
    cols = np.unique(np.linspace(l+.20*(r-l), l+.80*(r-l), 15).astype(int))
    rows = np.linspace(t+.25*(b-t), t+.70*(b-t), 7).astype(int)
    points = []
    for u in cols:
        samples = depth[rows, u]
        valid = samples[np.isfinite(samples) & (samples > .10) & (samples < 4.99)]
        if len(valid) >= 4 and np.ptp(np.percentile(valid, [20, 80])) < .16:
            points.append(world_point(u, float(np.median(valid)), fx, cx, pose))
    if len(points) < 7:
        return None
    center = np.median(points, axis=0)
    offsets = np.asarray(points) - center
    _, _, axes = np.linalg.svd(offsets, full_matrices=False)
    tangent, normal = axes
    if np.ptp(offsets @ tangent) < .18 or np.percentile(abs(offsets @ normal), 85) > .09:
        return None
    if (center-pose[:2]) @ normal < 0:
        normal = -normal
    return tuple(center), math.atan2(normal[1], normal[0])


def locate_opening(box, depth, fx, cx, pose):
    """Fit a threshold between two observed near-to-far edges inside the proposed portal."""
    l, t, r, b = box
    if r-l < 14 or b-t < 12:
        return None
    w = depth.shape[1]
    rows = np.linspace(t+.26*(b-t), t+.62*(b-t), 9).astype(int)
    profile = depth[rows]
    # Gazebo uses +inf for a return beyond the far clip. Zero/NaN denotes missing data.
    near = np.isfinite(profile) & (profile > .10) & (profile < 4.99)
    edges = []
    step = max(2, int(.035*(r-l)))
    radius = max(5, int(.23*(r-l)))
    for edge, inward in ((l, 1), (r, -1)):
        best = None
        for u in range(max(step*2, edge-radius), min(w-2*step, edge+radius+1)):
            outer = u - inward*step
            patch = profile[:, outer-1:outer+2]
            valid = near[:, outer-1:outer+2]
            if valid.sum() < 19:
                continue
            z = float(np.median(patch[valid]))
            if (valid & (abs(patch-z) < .18)).sum() < 18:
                continue
            inner = u + inward*step
            background = profile[:, inner-1:inner+2]
            far = (background > z+.65).sum()
            if far >= 20:
                score = abs(u-edge) + .1*(background.size-far)
                if best is None or score < best[0]:
                    best = score, u, z
        if best is None:
            return None
        edges.append(best[1:])
    (ul, zl), (ur, zr) = edges
    if ur <= ul:
        return None
    columns = np.linspace(.24*ul+.76*ur, .76*ul+.24*ur, 15).astype(int)
    weight = (columns-ul)/(ur-ul)
    gate_depth = 1 / ((1-weight)/zl + weight/zr)
    if np.any((profile[:, columns] > gate_depth+.4).sum(axis=0) < 6):
        return None
    a, b = [world_point(u, z, fx, cx, pose) for u, z in edges]
    width = math.dist(a, b)
    if not 1.12 <= width <= 3.5:
        return None
    center = ((a[0]+b[0])/2, (a[1]+b[1])/2)
    normal = math.atan2(b[1]-a[1], b[0]-a[0]) + math.pi/2
    toward = (math.cos(normal), math.sin(normal))
    if (center[0]-pose[0])*toward[0] + (center[1]-pose[1])*toward[1] < 0:
        normal += math.pi
    # An oblique glimpse of one jamb or a crate edge is poor crossing evidence.
    bearing = math.atan2(center[1]-pose[1], center[0]-pose[0])
    if math.cos(normal-bearing) < .55:
        return None
    return center, math.atan2(math.sin(normal), math.cos(normal))


def depth_openings(depth, fx, cx, pose):
    """Propose complete near/far/near openings around camera height, before querying RGB."""
    h, w = depth.shape
    rows = np.linspace(.39*h, .59*h, 9).astype(int)
    lag = max(2, round(w*.012))
    a, b = depth[rows, :-2*lag], depth[rows, 2*lag:]
    def edges(near, far):
        mask = ((near > .1) & (near < 4.99) & (far > near+.65)).sum(axis=0) >= 6
        changes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        return [lag+(lo+hi-1)//2 for lo, hi in zip(changes[::2], changes[1::2])]
    lefts, rights = edges(a, b), edges(b, a)
    boxes = [(l, int(.20*h), r, int(.84*h)) for l in lefts for r in rights if r-l >= 14
             and not any(l < edge < r for edge in lefts+rights)]
    for box in sorted(boxes, key=lambda box: abs((box[0]+box[2])/2-cx)):
        gate = locate_opening(box, depth, fx, cx, pose)
        if gate is not None:
            yield box, gate


def analyze_frame(server, kind, label, stamp, pose, rgb, depth, fx, cx):
    from .door_navigator import candidate_crop
    from .door_snapshot import query_image

    names = {'door': '一扇房门的门板',
             'exit': '已经打开并能穿过的建筑门洞，框住两侧门框之间的开口',
             'crop': '一株植物（盆栽或种植槽上的植株）', 'object': '指定物体'}
    result = Evidence(stamp, pose, kind, None, None, '', b'')
    ok, jpeg = cv2.imencode('.jpg', rgb)
    if not ok:
        raise ValueError('RGB 图像编码失败')
    result.jpeg = jpeg.tobytes()
    if kind == 'photo':
        result.detail = '已保存当前 RGB 画面到本次日志'
        return result
    if kind == 'inspect':
        prompt = '根据这一帧，用一至两句话描述能看见的物体、位置和外观。模糊的细节请说明看不清。'
        if label:
            prompt += '观察对象是' + json.dumps(label, ensure_ascii=False)
        result.detail = '现场观察：' + query_image(server, jpeg, 'jpeg', prompt).strip()
        return result
    h, w = depth.shape
    rejected_gaps = []
    if kind == 'exit':
        # A panel box at the edge of the RGB image can miss the free gap completely.
        # Verify the actual gap in context, including the jambs and nearby door leaves.
        for index, (box, gate) in enumerate(depth_openings(depth, fx, cx, pose)):
            if index >= 2:
                break
            l, t, r, b = box
            margin = (r-l)*.42
            context = (max(0,l-margin)*1000/w, 0, min(w,r+margin)*1000/w, 1000)
            crop = candidate_crop(rgb, context)
            if crop is None:
                continue
            subject = ('门扇为'+label+'的') if label else ''
            prompt = ('判断画面中间通路的用途。若能看到建筑门框或移开的门扇，并有向外的开口，回答'
                      '「'+subject+'打开的门洞」。作物通道、两件家具之间的空隙、实墙或关着的门回答「其他」。')
            answer = query_image(server, crop, 'jpeg', prompt).strip(' \r\n。！“”「」')
            result.detail = f'深度门洞候选像素 {l}～{r}，RGB 复核 {answer[:80]}'
            if answer == subject+'打开的门洞':
                result.point, result.heading = gate
                return result
            rejected_gaps.append(gate[0])
    target = names[kind]
    if label:
        target += '，指定外观或名称为' + json.dumps(label, ensure_ascii=False)
    query = f'找出画面中最清楚的{target}。只输出一个 bbox_2d，坐标为 0 至 1000 的 JSON；看不见就输出 []。'
    raw = query_image(server, jpeg, 'jpeg', query)
    result.detail += f' {kind} 定位回答 {raw[:240]}'
    try:
        box = door_box(raw)
    except (TypeError, ValueError):
        box = None
    if not box:
        return result
    crop = candidate_crop(rgb, box)
    if crop is None:
        return result
    classes = {'exit': ('打开的门洞', '普通走道、关闭的门、其他'),
               'door': ('房门', '柜子、箱子、墙或屏风、其他'),
               'crop': ('植物', '空种植槽、地面、墙、其他'), 'object': (label, '其他')}
    wanted, alternatives = classes[kind]
    if label and kind in ('door', 'exit'):
        wanted = (label+'的房门') if kind == 'door' else ('门扇为'+label+'的打开门洞')
        alternatives = '其他颜色的门、' + alternatives
    if kind == 'crop' and label:
        wanted, alternatives = label, '其他植物、空种植槽、其他'
    prompt = f'图像中心的主体更像什么？在「{wanted}、{alternatives}」中选一个。只输出类别名称。'
    classification = query_image(server, crop, 'jpeg', prompt).strip(' \r\n。！“”「」')
    result.detail += f'；局部复核 {classification[:100]}'
    px = tuple(min(limit-1, round(value*limit/1000)) for value, limit in zip(box, (w, h, w, h)))
    if kind == 'exit' and classification in (wanted, '关闭的门', '房门', '门板'):
        result.panel = locate_door_face(px, depth, fx, cx, pose)
    if classification != wanted:
        return result
    if kind == 'exit':
        gate = locate_opening(px, depth, fx, cx, pose)
        if gate is not None and all(math.dist(gate[0], p) > .55 for p in rejected_gaps):
            result.point, result.heading = gate
    elif kind == 'door':
        face = locate_door_face(px, depth, fx, cx, pose)
        if face is not None:
            result.point, result.heading = face
    else:
        result.point = locate_visible(px, depth, fx, cx, pose)
    result.detail += '；深度几何未确认' if result.point is None else f'；实测位置 {result.point}'
    return result


class LiveVision:
    """One latest pair and one in-flight request; ignore results predating a changed task."""

    def __init__(self, node, server):
        import message_filters
        from cv_bridge import CvBridge
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.server = server
        self.bridge = CvBridge()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pair = self.intrinsics = self.pending = self.key = None
        self.generation = 0
        self.capture_until = math.inf
        self.last = -math.inf
        self.poses = deque(maxlen=100)
        node.create_subscription(CameraInfo, '/rgbd_camera/camera_info', self.info, qos_profile_sensor_data)
        self.subs = [message_filters.Subscriber(node, Image, '/rgbd_camera/'+topic,
                                               qos_profile=qos_profile_sensor_data)
                     for topic in ('image', 'depth_image')]
        self.sync = message_filters.ApproximateTimeSynchronizer(self.subs, 5, .035)
        self.sync.registerCallback(self.images)

    def images(self, rgb, depth):
        stamp = rgb.header.stamp
        self.pair = stamp.sec + stamp.nanosec*1e-9, rgb, depth

    def info(self, msg):
        if msg.k[0] > 0:
            self.intrinsics = (msg.k[0], msg.k[2])

    def pose(self, stamp, pose):
        if self.poses and stamp < self.poses[-1][0]-.5:
            self.poses.clear()
            self.reset()
            self.last = -math.inf
            self.pair = None
        self.poses.append((stamp, pose))

    def reset(self):
        self.generation += 1
        self.capture_until = math.inf
        self.last = self.pair[0] if self.pair else -math.inf
        self.key = None

    def poll(self, kind, now, label=''):
        key = (kind, label)
        if self.key != key:
            self.generation += 1
            self.key = key
        if self.pending is not None:
            if not self.pending.done():
                return None
            future, self.pending = self.pending, None
            if self.ticket == self.generation:
                try:
                    return future.result()
                except Exception as exc:
                    return Evidence(self.last, self.capture_pose, kind, None, None, f'视觉请求失败 {exc}', b'', True)
        if not self.pair or not self.intrinsics or not self.poses:
            return None
        stamp, rgb, depth = self.pair
        if stamp <= self.last or stamp > self.capture_until or not -.1 <= now-stamp <= .7:
            return None
        ts, pose = min(self.poses, key=lambda sample: abs(sample[0]-stamp))
        if abs(ts-stamp) > .12:
            return None
        self.last = stamp
        try:
            color = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')
            metres = self.bridge.imgmsg_to_cv2(depth, 'passthrough')
            if depth.encoding == '16UC1':
                metres = metres.astype(np.float32)*.001
            elif depth.encoding != '32FC1':
                raise ValueError('无法读取深度单位 '+depth.encoding)
            if color.shape[:2] != metres.shape:
                raise ValueError('RGB 和深度图尺寸不一致')
        except Exception as exc:
            return Evidence(stamp, pose, kind, None, None, f'相机同步帧无效 {exc}', b'', True)
        self.ticket, self.capture_pose = self.generation, pose
        self.pending = self.pool.submit(analyze_frame, self.server, kind, label, stamp, pose,
                                        color, metres, *self.intrinsics)
        return None

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
