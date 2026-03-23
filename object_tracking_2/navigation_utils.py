import math
from dataclasses import dataclass

from geometry_msgs.msg import Point, PoseStamped


@dataclass
class GoalComputation:
    goal_pose: PoseStamped | None
    distance_to_target: float
    reached: bool


def build_standoff_goal(
    robot_x: float,
    robot_y: float,
    target_point: Point,
    stop_distance: float,
    stop_distance_tolerance: float,
    stamp,
    yaw_to_quaternion,
    frame_id: str = 'map',
) -> GoalComputation:
    dx = target_point.x - robot_x
    dy = target_point.y - robot_y
    distance = math.hypot(dx, dy)

    if distance <= stop_distance + stop_distance_tolerance or distance <= 1e-6:
        return GoalComputation(
            goal_pose=None,
            distance_to_target=distance,
            reached=True,
        )

    travel_distance = distance - stop_distance
    scale = travel_distance / distance

    goal_x = robot_x + dx * scale
    goal_y = robot_y + dy * scale
    yaw = math.atan2(dy, dx)

    goal = PoseStamped()
    goal.header.frame_id = frame_id
    goal.header.stamp = stamp
    goal.pose.position.x = goal_x
    goal.pose.position.y = goal_y
    goal.pose.orientation = yaw_to_quaternion(yaw)

    return GoalComputation(
        goal_pose=goal,
        distance_to_target=distance,
        reached=False,
    )