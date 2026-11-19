"""Interactive terminal for the entrance-relative door task."""

import rclpy
from std_msgs.msg import String

from .door_task import parse_command


def main():
    rclpy.init()
    node = rclpy.create_node('door_console')
    pub = node.create_publisher(String, '/door/task', 5)
    try:
        print('正在等待四门导航节点…', flush=True)
        while rclpy.ok() and not pub.get_subscription_count():
            rclpy.spin_once(node, timeout_sec=.2)
        print('例如：以0.12米/秒去对面的门，在门前1米停下。途中可输入“速度改为0.08米/秒”或“停车”。')
        while rclpy.ok():
            text = input('指令 > ').strip()
            if text not in ('停止', '停车', '取消'):
                try:
                    parse_command(text)
                except ValueError as exc:
                    print(exc)
                    continue
            pub.publish(String(data=text))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if rclpy.ok():
            pub.publish(String(data='停车'))
            rclpy.spin_once(node, timeout_sec=.1)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
