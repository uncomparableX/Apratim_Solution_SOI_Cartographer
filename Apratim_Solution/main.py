"""
Entry point of the package.
"""

import rclpy

from .solution_node import SolutionNode


def main(args=None):

    rclpy.init(args=args)

    node = SolutionNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()