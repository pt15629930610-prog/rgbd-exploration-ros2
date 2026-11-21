"""Greenhouse route terminal, with a model-only preview mode."""

import argparse
import json
from threading import Thread

from .online_language import interpret_online


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', metavar='指令', help='调用本地模型预演任务分解')
    parser.add_argument('--server', default='http://192.168.19.1:18080')
    args, ros_args = parser.parse_known_args()
    if args.preview:
        try:
            trace = []
            steps = interpret_online(args.server, args.preview, trace)
            print('\n'.join(trace))
            print('在线执行步骤（地标位置由运行时观测确定）')
            for index, step in enumerate(steps, 1):
                print(f'{index}. {json.dumps(step, ensure_ascii=False)}')
                if step['do'] == 'unsupported':
                    if step.get('dependent', True):
                        print('执行到这里时停车并报告。')
                        break
                    print('跳过这项后继续执行后续独立动作。')
        except (ValueError, OSError, KeyError, IndexError, TypeError) as exc:
            parser.exit(1, f'未执行：{exc}\n')
        return

    import rclpy
    from rclpy.executors import ExternalShutdownException
    from std_msgs.msg import String

    rclpy.init(args=ros_args)
    node = rclpy.create_node('route_console')
    pub = node.create_publisher(String, '/greenhouse/instruction', 5)
    node.create_subscription(String, '/greenhouse/feedback', lambda msg: print(msg.data, flush=True), 10)
    thread = None

    def display():
        try:
            rclpy.spin(node)
        except ExternalShutdownException:
            pass

    try:
        print('等待大棚路线节点…', flush=True)
        while rclpy.ok() and not pub.get_subscription_count():
            rclpy.spin_once(node, timeout_sec=.2)
        thread = Thread(target=display, daemon=True)
        thread.start()
        print('输入自然语言路线，途中输入“停车”可取消。', flush=True)
        while rclpy.ok():
            text = input('指令 > ').strip()
            if text:
                pub.publish(String(data=text))
    except (KeyboardInterrupt, EOFError, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            pub.publish(String(data='停车'))
            rclpy.shutdown()
        if thread is not None:
            thread.join(timeout=1.)
        node.destroy_node()
