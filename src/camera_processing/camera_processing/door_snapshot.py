"""Save one simulated RGB frame and optionally query a local Qwen server."""

import argparse
import base64
import json
from pathlib import Path
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New PNG filename')
    parser.add_argument('--server', help='Model server URL, e.g. http://host:18080')
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    rclpy.init()
    node = rclpy.create_node('door_snapshot')
    frames = []
    try:
        node.create_subscription(Image, '/rgbd_camera/image', frames.append, qos_profile_sensor_data)
        end = time.monotonic() + 15
        while rclpy.ok() and not frames and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.25)
        if not frames:
            raise SystemExit('No RGB frame in 15 s. Check the four_doors world and RGB bridge.')
        rgb = CvBridge().imgmsg_to_cv2(frames[0], desired_encoding='bgr8')
        ok, png = cv2.imencode('.png', rgb)
        if not ok:
            raise SystemExit('Could not encode RGB frame.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
    try:
        with args.output.open('xb') as output:
            output.write(png.tobytes())
    except FileExistsError:
        raise SystemExit('File already exists. Choose a new output filename.')
    print(f'Saved {args.output.resolve()}', flush=True)
    if args.server:
        content = [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                                               base64.b64encode(png).decode('ascii')}},
            {'type': 'text', 'text': '只根据当前画面判断：能否看到房门？若可见，描述其位置和可见依据。'
                                    '若遮挡或不确定请说明。'},
        ]
        body = {'model': 'Qwen3-VL-2B-Instruct', 'messages': [{'role': 'user', 'content': content}],
                'temperature': 0, 'max_tokens': 160}
        request = Request(args.server.rstrip('/') + '/v1/chat/completions',
                          json.dumps(body).encode(), {'Content-Type': 'application/json'})
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=90) as response:
                print(json.load(response)['choices'][0]['message']['content'])
        except OSError as exc:
            raise SystemExit(f'Frame saved, but the model server was unreachable: {exc}') from exc
