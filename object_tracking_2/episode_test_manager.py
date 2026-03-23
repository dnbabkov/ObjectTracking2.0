import csv
import json
import math
import re
from enum import Enum, auto
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node

import tf2_ros

from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Quaternion, Twist

from cv_bridge import CvBridge, CvBridgeError

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class TestPhase(Enum):
    IDLE = auto()
    RUNNING_EPISODE = auto()
    RETURNING_HOME = auto()
    ALIGNING_HOME = auto()


class EpisodeTestManager(Node):
    def __init__(self):
        super().__init__('episode_test_manager')

        # ------------------------------------------------------------------
        # НАСТРАИВАЕМЫЕ ЧАСТИ В КОДЕ
        # ------------------------------------------------------------------
        self.TEST_PROMPTS = [
            'traffic cone',
            'red fire hydrant',
            'car wheel',
            'cardboard box',
            'dark green trash can',
        ]
        self.EPISODES_PER_PROMPT = 20

        # ------------------------------------------------------------------
        # PARAMETERS
        # ------------------------------------------------------------------
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('prompt_topic', '/target_prompt')
        self.declare_parameter('target_reached_topic', '/target_reached')
        self.declare_parameter('episode_start_info_topic', '/episode_start_info')
        self.declare_parameter('episode_start_image_topic', '/episode_start_image')
        self.declare_parameter('navigate_action_name', 'navigate_to_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('return_timeout_sec', 120.0)
        self.declare_parameter('episode_timeout_sec', 60.0)
        self.declare_parameter('restart_delay_sec', 1.0)
        self.declare_parameter('home_align_angular_speed', 0.15)
        self.declare_parameter('home_yaw_tolerance_deg', 1.5)
        self.declare_parameter('home_align_timeout_sec', 20.0)

        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.robot_frame = self.get_parameter('robot_frame').get_parameter_value().string_value
        self.prompt_topic = self.get_parameter('prompt_topic').get_parameter_value().string_value
        self.target_reached_topic = self.get_parameter('target_reached_topic').get_parameter_value().string_value
        self.episode_start_info_topic = self.get_parameter('episode_start_info_topic').get_parameter_value().string_value
        self.episode_start_image_topic = self.get_parameter('episode_start_image_topic').get_parameter_value().string_value
        self.navigate_action_name = self.get_parameter('navigate_action_name').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.return_timeout_sec = self.get_parameter('return_timeout_sec').get_parameter_value().double_value
        self.restart_delay_sec = self.get_parameter('restart_delay_sec').get_parameter_value().double_value
        self.home_align_angular_speed = self.get_parameter('home_align_angular_speed').get_parameter_value().double_value
        self.home_yaw_tolerance_deg = self.get_parameter('home_yaw_tolerance_deg').get_parameter_value().double_value
        self.home_align_timeout_sec = self.get_parameter('home_align_timeout_sec').get_parameter_value().double_value

        self.home_yaw_tolerance = math.radians(self.home_yaw_tolerance_deg)

        # ------------------------------------------------------------------
        # PUB / SUB
        # ------------------------------------------------------------------
        self.prompt_pub = self.create_publisher(String, self.prompt_topic, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.target_reached_sub = self.create_subscription(
            Bool,
            self.target_reached_topic,
            self.target_reached_callback,
            10,
        )

        self.episode_start_info_sub = self.create_subscription(
            String,
            self.episode_start_info_topic,
            self.episode_start_info_callback,
            10,
        )

        self.episode_start_image_sub = self.create_subscription(
            Image,
            self.episode_start_image_topic,
            self.episode_start_image_callback,
            rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value,
        )

        # ------------------------------------------------------------------
        # SERVICES
        # ------------------------------------------------------------------
        self.save_pose_srv = self.create_service(
            Trigger,
            'save_start_pose',
            self.save_start_pose_callback,
        )
        self.start_test_srv = self.create_service(
            Trigger,
            'start_test_mode',
            self.start_test_mode_callback,
        )
        self.cancel_episode_srv = self.create_service(
            Trigger,
            'cancel_current_episode',
            self.cancel_current_episode_callback,
        )
        self.stop_test_srv = self.create_service(
            Trigger,
            'stop_test_mode',
            self.stop_test_mode_callback,
        )

        # ------------------------------------------------------------------
        # TF / ACTIONS / BRIDGE
        # ------------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.nav2_client = ActionClient(self, NavigateToPose, self.navigate_action_name)
        self.bridge = CvBridge()

        # ------------------------------------------------------------------
        # TEST STATE
        # ------------------------------------------------------------------
        self.saved_start_pose = None

        self.phase = TestPhase.IDLE
        self.test_active = False

        self.prompt_index = 0
        self.completed_episodes_for_current_prompt = 0
        self.current_prompt = ''
        self.current_episode_number_for_prompt = 0

        self.last_target_reached = False

        self.return_goal_handle = None
        self.return_deadline = None
        self.align_deadline = None
        self.restart_timer = None

        # Текущее состояние эпизода для логирования
        self.current_episode_start_time = None
        self.current_episode_target_coords = None
        self.current_episode_image_saved = False

        # Артефакты теста
        self.results_root = Path('test_results')
        self.current_test_number = None
        self.current_results_file = None
        self.current_images_dir = None

        self.watchdog_timer = self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info('EpisodeTestManager started')

        self.episode_timeout_sec = (
            self.get_parameter('episode_timeout_sec')
            .get_parameter_value()
            .double_value
        )

        self.episode_deadline = None

    # ----------------------------------------------------------------------
    # GEOMETRY HELPERS
    # ----------------------------------------------------------------------
    @staticmethod
    def quaternion_to_yaw(q: Quaternion) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def sanitize_prompt_for_filename(prompt: str) -> str:
        sanitized = re.sub(r'[^\w\-.]+', '_', prompt, flags=re.UNICODE)
        sanitized = sanitized.strip('_')
        return sanitized if sanitized else 'prompt'

    # ----------------------------------------------------------------------
    # BASIC PUB HELPERS
    # ----------------------------------------------------------------------
    def _publish_prompt(self, text: str):
        msg = String()
        msg.data = text
        self.prompt_pub.publish(msg)

    def _clear_prompt(self):
        self._publish_prompt('')

    def _publish_stop(self):
        self.cmd_vel_pub.publish(Twist())

    # ----------------------------------------------------------------------
    # TF HELPERS
    # ----------------------------------------------------------------------
    def _lookup_current_robot_pose(self) -> PoseStamped:
        transform = self.tf_buffer.lookup_transform(
            self.map_frame,
            self.robot_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=0.5),
        )

        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return pose

    def _get_robot_yaw_in_map(self) -> float:
        pose = self._lookup_current_robot_pose()
        return self.quaternion_to_yaw(pose.pose.orientation)

    def _get_saved_start_yaw(self) -> float:
        return self.quaternion_to_yaw(self.saved_start_pose.pose.orientation)

    # ----------------------------------------------------------------------
    # OUTPUT FILES
    # ----------------------------------------------------------------------
    def _get_next_test_number(self) -> int:
        self.results_root.mkdir(parents=True, exist_ok=True)

        max_number = 0
        pattern = re.compile(r'^test_(\d+)_(results|images)$')

        for path in self.results_root.iterdir():
            match = pattern.match(path.name)
            if match:
                max_number = max(max_number, int(match.group(1)))

        return max_number + 1

    def _prepare_output_artifacts(self):
        self.current_test_number = self._get_next_test_number()
        self.current_results_file = self.results_root / f'test_{self.current_test_number}_results'
        self.current_images_dir = self.results_root / f'test_{self.current_test_number}_images'

        self.current_images_dir.mkdir(parents=True, exist_ok=True)

        with self.current_results_file.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([
                'prompt',
                'episode_number_for_prompt',
                'result',
                'target_x',
                'target_y',
                'target_z',
                'time_to_target_sec',
            ])

        self.get_logger().info(
            f'Results will be saved to: {self.current_results_file}'
        )
        self.get_logger().info(
            f'Images will be saved to: {self.current_images_dir}'
        )

    def _append_result_row(self, result: str):
        if self.current_results_file is None:
            return

        if self.current_episode_target_coords is None:
            tx, ty, tz = '', '', ''
        else:
            tx, ty, tz = self.current_episode_target_coords

        if self.current_episode_start_time is None:
            time_to_target_sec = ''
        else:
            elapsed = (
                self.get_clock().now() - self.current_episode_start_time
            ).nanoseconds / 1e9
            time_to_target_sec = f'{elapsed:.3f}'

        with self.current_results_file.open('a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([
                self.current_prompt,
                self.current_episode_number_for_prompt,
                result,
                tx,
                ty,
                tz,
                time_to_target_sec,
            ])

    # ----------------------------------------------------------------------
    # SERVICES
    # ----------------------------------------------------------------------
    def save_start_pose_callback(self, request, response):
        try:
            self.saved_start_pose = self._lookup_current_robot_pose()
            response.success = True
            response.message = (
                f'Start pose saved: '
                f'x={self.saved_start_pose.pose.position.x:.3f}, '
                f'y={self.saved_start_pose.pose.position.y:.3f}'
            )
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f'Failed to save start pose: {e}'
            self.get_logger().error(response.message)

        return response

    def start_test_mode_callback(self, request, response):
        if self.saved_start_pose is None:
            response.success = False
            response.message = 'Start pose is not saved'
            return response

        if self.test_active:
            response.success = False
            response.message = 'Test mode is already active'
            return response

        if len(self.TEST_PROMPTS) == 0:
            response.success = False
            response.message = 'TEST_PROMPTS list is empty'
            return response

        self._prepare_output_artifacts()

        self.prompt_index = 0
        self.completed_episodes_for_current_prompt = 0
        self.current_prompt = ''
        self.current_episode_number_for_prompt = 0

        self.test_active = True
        self.phase = TestPhase.IDLE
        self.last_target_reached = False

        self.get_logger().info(
            f'Starting test mode: {len(self.TEST_PROMPTS)} prompts, '
            f'{self.EPISODES_PER_PROMPT} episodes per prompt'
        )

        self._start_next_episode()

        response.success = True
        response.message = 'Test mode started'
        return response

    def cancel_current_episode_callback(self, request, response):
        if not self.test_active:
            response.success = False
            response.message = 'Test mode is not active'
            return response

        if self.phase != TestPhase.RUNNING_EPISODE:
            response.success = False
            response.message = f'Cannot cancel in phase {self.phase.name}'
            return response

        self.get_logger().warn('Current episode canceled by service call')
        self._finish_episode_and_return_home('canceled')

        response.success = True
        response.message = 'Current episode canceled'
        return response

    def stop_test_mode_callback(self, request, response):
        if not self.test_active:
            response.success = False
            response.message = 'Test mode is not active'
            return response

        self.get_logger().warn('Stopping test mode')
        self._abort_test_mode('Stopped by service call')

        response.success = True
        response.message = 'Test mode stopped'
        return response

    # ----------------------------------------------------------------------
    # SUBSCRIBERS
    # ----------------------------------------------------------------------
    def target_reached_callback(self, msg: Bool):
        current = bool(msg.data)
        rising_edge = current and not self.last_target_reached
        self.last_target_reached = current

        if not rising_edge:
            return

        if not self.test_active:
            return

        if self.phase != TestPhase.RUNNING_EPISODE:
            return

        self.get_logger().info('Received target reached signal')
        self._finish_episode_and_return_home('success')

    def episode_start_info_callback(self, msg: String):
        if not self.test_active or self.phase != TestPhase.RUNNING_EPISODE:
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Failed to parse /episode_start_info JSON: {e}')
            return

        msg_prompt = str(payload.get('prompt', ''))
        if msg_prompt != self.current_prompt:
            return

        tx = payload.get('target_x')
        ty = payload.get('target_y')
        tz = payload.get('target_z')

        if tx is None or ty is None or tz is None:
            return

        self.current_episode_target_coords = (
            float(tx),
            float(ty),
            float(tz),
        )

    def episode_start_image_callback(self, msg: Image):
        if not self.test_active or self.phase != TestPhase.RUNNING_EPISODE:
            return

        if self.current_images_dir is None:
            return

        if self.current_episode_image_saved:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().warn(f'Failed to decode episode start image: {e}')
            return

        filename = (
            f'{self.sanitize_prompt_for_filename(self.current_prompt)}'
            f'_{self.current_episode_number_for_prompt}.png'
        )
        filepath = self.current_images_dir / filename

        try:
            import cv2
            cv2.imwrite(str(filepath), image)
            self.current_episode_image_saved = True
            self.get_logger().info(f'Saved start segmented image: {filepath}')
        except Exception as e:
            self.get_logger().warn(f'Failed to save segmented image: {e}')

    # ----------------------------------------------------------------------
    # EPISODE FLOW
    # ----------------------------------------------------------------------
    def _start_next_episode(self):
        if not self.test_active:
            return

        while (
            self.prompt_index < len(self.TEST_PROMPTS)
            and self.completed_episodes_for_current_prompt >= self.EPISODES_PER_PROMPT
        ):
            self.prompt_index += 1
            self.completed_episodes_for_current_prompt = 0

        if self.prompt_index >= len(self.TEST_PROMPTS):
            self.get_logger().info('All prompts completed')
            self._finish_test_mode()
            return

        self.current_prompt = self.TEST_PROMPTS[self.prompt_index]
        self.current_episode_number_for_prompt = self.completed_episodes_for_current_prompt + 1

        self.current_episode_start_time = self.get_clock().now()
        self.episode_deadline = self.current_episode_start_time + Duration(
            seconds=self.episode_timeout_sec
        )
        self.current_episode_target_coords = None
        self.current_episode_image_saved = False
        self.last_target_reached = False
        self.phase = TestPhase.RUNNING_EPISODE

        self.get_logger().info(
            f'Starting prompt {self.prompt_index + 1}/{len(self.TEST_PROMPTS)}: '
            f'"{self.current_prompt}", '
            f'episode {self.current_episode_number_for_prompt}/{self.EPISODES_PER_PROMPT}'
        )

        self._publish_prompt(self.current_prompt)

    def _finish_episode_and_return_home(self, result: str):
        if not self.test_active:
            return

        if self.phase != TestPhase.RUNNING_EPISODE:
            return

        self._append_result_row(result)

        self.get_logger().info(
            f'Finished prompt "{self.current_prompt}", '
            f'episode {self.current_episode_number_for_prompt}/{self.EPISODES_PER_PROMPT}, '
            f'result={result}'
        )

        self.completed_episodes_for_current_prompt += 1

        self.episode_deadline = None

        self._clear_prompt()
        self.phase = TestPhase.RETURNING_HOME
        self._send_return_home_goal()

    # ----------------------------------------------------------------------
    # RETURN HOME
    # ----------------------------------------------------------------------
    def _send_return_home_goal(self):
        if self.saved_start_pose is None:
            self.get_logger().error('No saved start pose')
            self._abort_test_mode('No saved start pose')
            return

        if not self.nav2_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Nav2 action server is not available')
            self._abort_test_mode('Nav2 action server unavailable')
            return

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.saved_start_pose.header.frame_id
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose = self.saved_start_pose.pose

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info(
            f'Returning to saved pose: '
            f'x={goal_pose.pose.position.x:.3f}, '
            f'y={goal_pose.pose.position.y:.3f}'
        )

        send_goal_future = self.nav2_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._return_goal_response_callback)

        self.return_deadline = self.get_clock().now() + Duration(
            seconds=self.return_timeout_sec
        )

    def _return_goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Return-home goal rejected')
            self._abort_test_mode('Return-home goal rejected')
            return

        self.return_goal_handle = goal_handle
        self.get_logger().info('Return-home goal accepted')

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self._return_goal_result_callback)

    def _return_goal_result_callback(self, future):
        result = future.result()
        status = result.status

        self.return_goal_handle = None
        self.return_deadline = None

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Robot returned to saved position')
            self.phase = TestPhase.ALIGNING_HOME
            self.align_deadline = self.get_clock().now() + Duration(
                seconds=self.home_align_timeout_sec
            )
            return

        self.get_logger().error(f'Return-home goal failed with status {status}')
        self._abort_test_mode(f'Return-home goal failed with status {status}')

    def _align_home_orientation_step(self):
        if self.saved_start_pose is None:
            self._abort_test_mode('No saved start pose for alignment')
            return

        try:
            current_yaw = self._get_robot_yaw_in_map()
            target_yaw = self._get_saved_start_yaw()
        except Exception as e:
            self.get_logger().error(f'Failed to align home orientation: {e}')
            self._abort_test_mode(f'Failed to align home orientation: {e}')
            return

        yaw_error = self.normalize_angle(target_yaw - current_yaw)

        if abs(yaw_error) <= self.home_yaw_tolerance:
            self._publish_stop()
            self.phase = TestPhase.IDLE
            self.align_deadline = None
            self.get_logger().info('Home orientation aligned')
            self._schedule_next_episode()
            return

        cmd = Twist()
        cmd.angular.z = math.copysign(self.home_align_angular_speed, yaw_error)
        self.cmd_vel_pub.publish(cmd)

    # ----------------------------------------------------------------------
    # TIMERS / FINISH / ABORT
    # ----------------------------------------------------------------------
    def _schedule_next_episode(self):
        if self.restart_timer is not None:
            self.restart_timer.cancel()
            self.destroy_timer(self.restart_timer)
            self.restart_timer = None

        def timer_cb():
            if self.restart_timer is not None:
                self.restart_timer.cancel()
                self.destroy_timer(self.restart_timer)
                self.restart_timer = None
            self._start_next_episode()

        self.restart_timer = self.create_timer(self.restart_delay_sec, timer_cb)

    def _finish_test_mode(self):
        self._clear_prompt()
        self._publish_stop()

        self.phase = TestPhase.IDLE
        self.test_active = False

        self.prompt_index = 0
        self.completed_episodes_for_current_prompt = 0
        self.current_prompt = ''
        self.current_episode_number_for_prompt = 0

        self.current_episode_start_time = None
        self.current_episode_target_coords = None
        self.current_episode_image_saved = False

        self.last_target_reached = False
        self.return_deadline = None
        self.align_deadline = None

        if self.restart_timer is not None:
            self.restart_timer.cancel()
            self.destroy_timer(self.restart_timer)
            self.restart_timer = None

        self.get_logger().info(
            f'Test mode finished. Results: {self.current_results_file}, images: {self.current_images_dir}'
        )

        self.episode_deadline = None

    def _abort_test_mode(self, reason: str):
        self._clear_prompt()
        self._publish_stop()

        if self.return_goal_handle is not None:
            self.return_goal_handle.cancel_goal_async()
            self.return_goal_handle = None

        if self.restart_timer is not None:
            self.restart_timer.cancel()
            self.destroy_timer(self.restart_timer)
            self.restart_timer = None

        self.phase = TestPhase.IDLE
        self.test_active = False

        self.prompt_index = 0
        self.completed_episodes_for_current_prompt = 0
        self.current_prompt = ''
        self.current_episode_number_for_prompt = 0

        self.current_episode_start_time = None
        self.current_episode_target_coords = None
        self.current_episode_image_saved = False

        self.return_deadline = None
        self.align_deadline = None
        self.last_target_reached = False

        self.get_logger().warn(f'Test mode aborted: {reason}')

        self.episode_deadline = None

    def watchdog_callback(self):
        if self.phase == TestPhase.RUNNING_EPISODE:
            if self.episode_deadline is not None and self.get_clock().now() > self.episode_deadline:
                self.get_logger().error('Episode timeout expired')
                self._finish_episode_and_return_home('fail')
            return
    
        if self.phase == TestPhase.RETURNING_HOME:
            if self.return_deadline is None:
                return
    
            if self.get_clock().now() > self.return_deadline:
                self.get_logger().error('Return-home timeout expired')
                self._abort_test_mode('Return-home timeout expired')
            return
    
        if self.phase == TestPhase.ALIGNING_HOME:
            if self.align_deadline is not None and self.get_clock().now() > self.align_deadline:
                self.get_logger().error('Home alignment timeout expired')
                self._abort_test_mode('Home alignment timeout expired')
                return
    
            self._align_home_orientation_step()


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeTestManager()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()