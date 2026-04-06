import argparse
import math
import json

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PoseStamped, Twist, Quaternion, PointStamped

from cv_bridge import CvBridge, CvBridgeError

from message_filters import ApproximateTimeSynchronizer, Subscriber

import tf2_ros
from tf2_geometry_msgs import do_transform_point

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from object_tracking_2.navigation_utils import build_standoff_goal
from object_tracking_2.segment_factory import create_segmentator
from object_tracking_2.tracking_performance import (
    SegmentatorPerformanceMonitor,
    TrackingMode,
)


class TrackerNode(Node):
    def __init__(self, segmentator_name: str):
        super().__init__('tracker_node')

        self.segmentator = create_segmentator(segmentator_name)
        self.get_logger().info(
            f'Node initialized with segmentator: {self.segmentator.name}'
        )

        self.bridge = CvBridge()

        self.camera_info_received = False
        self.current_prompt = None
        self.target_found = False
        self.target_reached = False

        self.total_seg_time = 0.0
        self.segmentations = 0
        self.last_goal_update_time = self.get_clock().now()
        self.last_announced_mode = None

        self.search_rotation_active = False
        self.search_rotation_target_yaw = None

        self.nav2_goal_handle = None
        self.nav2_goal_active = False
        self.pending_goal_pose = None

        self.declare_parameter('search_angular_speed', 0.5)
        self.declare_parameter('goal_update_period', 2.0)
        self.declare_parameter('stop_distance', 0.5)
        self.declare_parameter('stop_distance_tolerance', 0.1)
        self.declare_parameter('realtime_threshold_sec', 0.35)
        self.declare_parameter('segmentation_time_ema_alpha', 0.3)
        self.declare_parameter('slow_search_step_angle_deg', 15.0)
        self.declare_parameter('slow_search_angle_tolerance_deg', 2.0)

        self.search_angular_speed = (
            self.get_parameter('search_angular_speed')
            .get_parameter_value()
            .double_value
        )
        self.goal_update_period = (
            self.get_parameter('goal_update_period')
            .get_parameter_value()
            .double_value
        )
        self.stop_distance = (
            self.get_parameter('stop_distance')
            .get_parameter_value()
            .double_value
        )
        self.stop_distance_tolerance = (
            self.get_parameter('stop_distance_tolerance')
            .get_parameter_value()
            .double_value
        )
        self.realtime_threshold_sec = (
            self.get_parameter('realtime_threshold_sec')
            .get_parameter_value()
            .double_value
        )
        self.segmentation_time_ema_alpha = (
            self.get_parameter('segmentation_time_ema_alpha')
            .get_parameter_value()
            .double_value
        )
        self.slow_search_step_angle_deg = (
        self.get_parameter('slow_search_step_angle_deg')
        .get_parameter_value()
        .double_value
        )
        self.slow_search_angle_tolerance_deg = (
            self.get_parameter('slow_search_angle_tolerance_deg')
            .get_parameter_value()
            .double_value
        )
        self.slow_search_step_angle = math.radians(self.slow_search_step_angle_deg)
        self.slow_search_angle_tolerance = math.radians(
            self.slow_search_angle_tolerance_deg
        )
        self.performance_monitor = SegmentatorPerformanceMonitor(
            realtime_threshold_sec=self.realtime_threshold_sec,
            ema_alpha=self.segmentation_time_ema_alpha,
        )

        self.rgb_sub = Subscriber(
            self,
            Image,
            '/camera/image_raw',
            qos_profile=rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value,
        )
        self.depth_sub = Subscriber(
            self,
            Image,
            '/depth_camera/depth/image_raw',
            qos_profile=rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value,
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1,
        )
        self.ts.registerCallback(self.synced_image_depth_callback)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            1,
        )

        self.prompt_sub = self.create_subscription(
            String,
            '/target_prompt',
            self.prompt_callback,
            1,
        )

        self.search_cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_pub = self.create_publisher(Image, '/image_out', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 1)
        self.target_reached_pub = self.create_publisher(Bool, '/target_reached', 10)

        self.episode_start_info_pub = self.create_publisher(String, '/episode_start_info', 10)
        self.episode_start_image_pub = self.create_publisher(
            Image,
            '/episode_start_image',
            rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value,
        )
        self.episode_start_artifacts_published = False

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.timer = self.create_timer(0.1, self.timer_callback)

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Quaternion:
        q = Quaternion()
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q
    
    @staticmethod
    def quaternion_to_yaw(q: Quaternion) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _get_robot_yaw_in_map(self) -> float:
        transform_base = self._lookup_transform(
            'map',
            'base_link',
            timeout_sec=0.5,
        )
        return self.quaternion_to_yaw(transform_base.transform.rotation)
    
    def _publish_target_reached(self, value: bool):
        msg = Bool()
        msg.data = value
        self.target_reached_pub.publish(msg)

    def _lookup_transform(self, target_frame: str, source_frame: str, timeout_sec: float = 0.5):
        return self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=timeout_sec),
        )

    def _publish_search_rotation(self):
        msg = Twist()
        msg.angular.z = self.search_angular_speed
        self.search_cmd_pub.publish(msg)

    def _publish_stop(self):
        self.search_cmd_pub.publish(Twist())

    def _start_search_rotation(self):
        try:
            current_yaw = self._get_robot_yaw_in_map()
        except Exception as e:
            self.get_logger().error(f'Не удалось получить yaw робота: {e}')
            self.search_rotation_active = False
            self.search_rotation_target_yaw = None
            return

        self.search_rotation_target_yaw = self.normalize_angle(
            current_yaw + self.slow_search_step_angle
        )
        self.search_rotation_active = True

        self.get_logger().info(
            f'Запуск дискретного поворота на '
            f'{self.slow_search_step_angle_deg:.1f} deg'
        )

    def _stop_search_rotation(self):
        if self.search_rotation_active:
            self._publish_stop()

        self.search_rotation_active = False
        self.search_rotation_target_yaw = None

    def _update_discrete_search_rotation(self):
        if not self.search_rotation_active or self.search_rotation_target_yaw is None:
            return

        try:
            current_yaw = self._get_robot_yaw_in_map()
        except Exception as e:
            self.get_logger().error(f'Не удалось обновить поворот поиска: {e}')
            self._stop_search_rotation()
            return

        yaw_error = self.normalize_angle(
            self.search_rotation_target_yaw - current_yaw
        )

        if abs(yaw_error) <= self.slow_search_angle_tolerance:
            self.get_logger().info('Дискретный поворот завершён')
            self._stop_search_rotation()
            return

        msg = Twist()
        msg.angular.z = math.copysign(self.search_angular_speed, yaw_error)
        self.search_cmd_pub.publish(msg)

    def _add_segmentation_time(self, segmentation_time: float):
        self.total_seg_time += segmentation_time
        self.segmentations += 1

    def _log_avg_seg_time(self, prefix: str):
        if self.segmentations == 0:
            return

        avg_time = self.total_seg_time / self.segmentations
        self.get_logger().info(f'{prefix} {avg_time:.4f} sec')
        self.total_seg_time = 0.0
        self.segmentations = 0

    def _announce_mode_if_needed(self, mode: TrackingMode):
        if mode == self.last_announced_mode:
            return

        if mode == TrackingMode.REALTIME:
            self.get_logger().info(
                f'Switching to REALTIME tracking mode '
                f'(avg segmentation time = {self.performance_monitor.ema_segmentation_time:.4f} sec)'
            )
        else:
            self.get_logger().info(
                f'Switching to SINGLE_GOAL tracking mode '
                f'(avg segmentation time = {self.performance_monitor.ema_segmentation_time:.4f} sec)'
            )

        self.last_announced_mode = mode

    def _pixel_to_camera_point(self, x_px: int, y_px: int, depth_image) -> Point | None:
        depth_value = depth_image[y_px, x_px]

        if np.isnan(depth_value) or depth_value <= 0.0:
            return None

        point_camera = Point()
        point_camera.x = (x_px - self.cx) * depth_value / self.fx
        point_camera.y = (y_px - self.cy) * depth_value / self.fy
        point_camera.z = float(depth_value)

        return point_camera

    def _camera_point_to_world(
        self,
        point_camera: Point,
        camera_frame: str = 'depth_camera_link_optical',
        world_frame: str = 'map',
        timeout_sec: float = 0.5,
    ) -> Point:
        point_stamped = PointStamped()
        point_stamped.header.frame_id = camera_frame
        point_stamped.header.stamp = self.get_clock().now().to_msg()
        point_stamped.point = point_camera

        transform = self._lookup_transform(
            world_frame,
            camera_frame,
            timeout_sec=timeout_sec,
        )

        point_world_stamped = do_transform_point(point_stamped, transform)
        return point_world_stamped.point

    def _get_robot_position_in_map(self) -> tuple[float, float]:
        transform_base = self._lookup_transform(
            'map',
            'base_link',
            timeout_sec=0.5,
        )

        robot_x = transform_base.transform.translation.x
        robot_y = transform_base.transform.translation.y

        return robot_x, robot_y

    def _send_goal_to_nav2(self, goal_pose: PoseStamped):
        self.pose_pub.publish(goal_pose)

        if not self.nav2_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn('Nav2 action server not available')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        send_goal_future = self.nav2_client.send_goal_async(
            goal_msg,
            feedback_callback=self._nav2_feedback_callback,
        )
        send_goal_future.add_done_callback(self._nav2_goal_response_callback)

        self.nav2_goal_active = True
        self.target_reached = False

    def _cancel_active_goal(self):
        self.pending_goal_pose = None

        if self.nav2_goal_handle is not None:
            self.get_logger().info('Canceling active Nav2 goal')
            self.nav2_goal_handle.cancel_goal_async()

    def _nav2_feedback_callback(self, feedback_msg):
        pass

    def _nav2_goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 goal rejected')
            self.nav2_goal_active = False
            self.nav2_goal_handle = None
            return

        self.get_logger().info('Nav2 goal accepted')
        self.nav2_goal_handle = goal_handle

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self._nav2_result_callback)

    def _update_nav2_goal(self, new_goal_pose: PoseStamped):
        if not self.nav2_goal_active or self.nav2_goal_handle is None:
            self.pending_goal_pose = None
            self._send_goal_to_nav2(new_goal_pose)
            return

        self.pending_goal_pose = new_goal_pose
        self.get_logger().info('Requesting Nav2 goal cancel for update...')
        self.nav2_goal_handle.cancel_goal_async()

    def _nav2_result_callback(self, future):
        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Nav2: goal reached')
            self.target_reached = True
            self._publish_target_reached(True)
            self.nav2_goal_active = False
            self.nav2_goal_handle = None
            self._stop_search_rotation()
            self._log_avg_seg_time('Average segmentation time:')
            return

        if status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('Nav2: goal canceled')
            self.nav2_goal_active = False
            self.nav2_goal_handle = None

            if self.pending_goal_pose is not None:
                next_goal = self.pending_goal_pose
                self.pending_goal_pose = None
                self._send_goal_to_nav2(next_goal)

            return

        self.get_logger().warn(f'Nav2: goal ended with status {status}')
        self.nav2_goal_active = False
        self.nav2_goal_handle = None

    def _reset_tracking_state(self):
        self.episode_start_artifacts_published = False
        self.target_found = False
        self.target_reached = False
        self.pending_goal_pose = None
        self.last_goal_update_time = self.get_clock().now()
        self._stop_search_rotation()
        self._cancel_active_goal()

    def _publish_episode_start_artifacts(self, point_world: Point, segmented_image, image_header):
        if self.episode_start_artifacts_published:
            return

        info = {
            'prompt': self.current_prompt,
            'target_x': float(point_world.x),
            'target_y': float(point_world.y),
            'target_z': float(point_world.z),
        }

        info_msg = String()
        info_msg.data = json.dumps(info, ensure_ascii=False)
        self.episode_start_info_pub.publish(info_msg)

        image_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding='bgr8')
        image_msg.header = image_header
        self.episode_start_image_pub.publish(image_msg)

        self.episode_start_artifacts_published = True

    def timer_callback(self):
        if self.current_prompt is None or self.current_prompt.strip() == '':
            self._stop_search_rotation()
            return

        if self.target_found or self.target_reached:
            self._stop_search_rotation()
            return

        mode = self.performance_monitor.current_mode()

        if mode == TrackingMode.REALTIME:
            self._publish_search_rotation()
            return

        if self.search_rotation_active:
            self._update_discrete_search_rotation()

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_info_received:
            return

        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        self.camera_info_received = True

        self.get_logger().info(
            f'Camera intrinsics updated: '
            f'fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}'
        )

    def prompt_callback(self, msg: String):
        prompt = msg.data.strip()

        if self.current_prompt == prompt:
            return

        self.current_prompt = prompt
        self.get_logger().info(f'Новый промпт получен: "{self.current_prompt}"')
        self._reset_tracking_state()
        self._publish_target_reached(False)

    def _should_skip_frame(self, mode: TrackingMode, now) -> bool:
        if self.current_prompt is None or self.current_prompt.strip() == '':
            return True

        if self.target_reached:
            return True

        if mode == TrackingMode.SINGLE_GOAL:
            if self.search_rotation_active:
                return True

            if self.nav2_goal_active:
                return True

            return False

        if self.pending_goal_pose is not None:
            return True

        if self.target_found:
            if (now - self.last_goal_update_time) < Duration(
                seconds=self.goal_update_period
            ):
                return True

        return False

    def _handle_missing_detection(self, mode: TrackingMode):
        if mode == TrackingMode.SINGLE_GOAL:
            self.get_logger().warn('Объект не найден, выполняется поворот для следующей попытки')
            self._start_search_rotation()

    def _handle_detected_target(self, center_coords, depth, now, mode: TrackingMode, segmented_image, image_header):
        self._stop_search_rotation()

        if not self.target_found:
            self.get_logger().info(
                f'Координаты центра: ({center_coords[0]}, {center_coords[1]})'
            )
            self.target_found = True

        x_px = int(center_coords[0])
        y_px = int(center_coords[1])

        point_camera = self._pixel_to_camera_point(x_px, y_px, depth)
        if point_camera is None:
            self.get_logger().warn('Некорректная глубина в центре объекта')
            return

        try:
            point_world = self._camera_point_to_world(point_camera)
            robot_x, robot_y = self._get_robot_position_in_map()

            self._publish_episode_start_artifacts(
            point_world=point_world,
            segmented_image=segmented_image,
            image_header=image_header,
        )

            goal_data = build_standoff_goal(
                robot_x=robot_x,
                robot_y=robot_y,
                target_point=point_world,
                stop_distance=self.stop_distance,
                stop_distance_tolerance=self.stop_distance_tolerance,
                stamp=now.to_msg(),
                yaw_to_quaternion=self.yaw_to_quaternion,
                frame_id='map',
            )

            self.get_logger().info(
                f'Объект в map frame: '
                f'X={point_world.x:.2f}, '
                f'Y={point_world.y:.2f}, '
                f'Z={point_world.z:.2f}'
            )
            self.get_logger().info(
                f'Расстояние до объекта = {goal_data.distance_to_target:.2f}, '
                f'требуемая дистанция остановки = {self.stop_distance:.2f}'
            )

            if goal_data.reached:
                self.target_reached = True

                self._publish_target_reached(True)

                self.get_logger().info(
                    f'Цель достигнута: робот находится на расстоянии '
                    f'{self.stop_distance:.2f} м от объекта'
                )
                self._cancel_active_goal()
                self._log_avg_seg_time('Average segmentation time:')
                return

            if mode == TrackingMode.REALTIME:
                self.last_goal_update_time = now
                self._update_nav2_goal(goal_data.goal_pose)
            else:
                self._send_goal_to_nav2(goal_data.goal_pose)

        except Exception as e:
            self.get_logger().error(f'Ошибка трансформации в map: {e}')

    def synced_image_depth_callback(self, rgb_msg: Image, depth_msg: Image):

        if not self.camera_info_received:
            return

        now = self.get_clock().now()
        mode_before_inference = self.performance_monitor.current_mode()

        if self._should_skip_frame(mode_before_inference, now):
            return

        try:
            image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'Ошибка конвертации изображения: {e}')
            return

        try:
            segmentation_result = self.segmentator.segment(
                image=image,
                prompt=self.current_prompt,
                depth=depth,
            )
        except Exception as e:
            self.get_logger().error(f'Ошибка сегментации: {e}')
            return

        self.get_logger().info(
            f'segmentation done, center={segmentation_result.center_coords}, '
            f'time={segmentation_result.segmentation_time:.3f}'
        )

        self.image_pub.publish(
            self.bridge.cv2_to_imgmsg(segmentation_result.vis_image, encoding='bgr8')
        )

        self._add_segmentation_time(segmentation_result.segmentation_time)

        mode_after_inference = self.performance_monitor.update(
            segmentation_result.segmentation_time
        )
        self._announce_mode_if_needed(mode_after_inference)

        if segmentation_result.center_coords is None:
            self._handle_missing_detection(mode_after_inference)
            return

        self._handle_detected_target(
            center_coords=segmentation_result.center_coords,
            depth=depth,
            now=now,
            mode=mode_after_inference,
            segmented_image=segmentation_result.vis_image,
            image_header=rgb_msg.header,
        )


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='ROS2 tracker node')
    parser.add_argument(
        '--segmentator',
        required=True,
        choices=['CLIP', 'DinoSAM', 'SEEM', 'OpenSeeD'],
        help='Имя используемого сегментатора',
    )
    return parser.parse_args(args)


def main(args=None):
    non_ros_args = rclpy.utilities.remove_ros_args(args=args)[1:]
    parsed_args = parse_args(non_ros_args)

    rclpy.init(args=args)

    node = TrackerNode(segmentator_name=parsed_args.segmentator)

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()