"""
tracker_node_dynamic.py
=======================
Узел для преследования и перехвата ДВИЖУЩИХСЯ объектов.

Наследует TrackerNode из tracker_node.py и переопределяет только те
методы, которые отличаются для динамического режима:

  Переопределено:
    __init__               — добавляет параметры контроллера, создаёт
                             DynamicPursuitController, убирает Nav2 клиент
    _should_skip_frame     — не блокируется на nav2_goal_active
    _handle_missing_detection — немедленный поворот-поиск без ожидания SINGLE_GOAL
    _handle_detected_target   — cmd_vel напрямую вместо Nav2 goal
    timer_callback         — следит за таймаутом потери цели
    _reset_tracking_state  — дополнительно сбрасывает историю контроллера

  Унаследовано без изменений (весь «скелет»):
    camera_info_callback   — читает fx/fy/cx/cy из /camera/camera_info
    prompt_callback        — принимает промпт из /target_prompt
    synced_image_depth_callback — синхронизирует RGB + depth, вызывает сегментатор
    _pixel_to_camera_point — пиксель + глубина → 3D в camera frame
    _camera_point_to_world — TF camera → map
    _get_robot_position_in_map / _get_robot_yaw_in_map
    _start/stop_search_rotation, _update_discrete_search_rotation
    _publish_episode_start_artifacts
    _add_segmentation_time, _log_avg_seg_time, _announce_mode_if_needed

Запуск:
    ros2 run object_tracking_2 tracker_node_dynamic \
        --segmentator SAM2 \
        --ros-args \
            -p control_mode:=intercept \
            -p desired_distance:=0.6 \
            -p max_linear_vel:=0.5

Параметры (все опциональные):
    control_mode          string  'pursuit'  — pursuit | intercept
    desired_distance      float    0.6       — целевая дистанция до объекта (м)
    max_linear_vel        float    0.5       — макс. линейная скорость (м/с)
    max_angular_vel       float    1.5       — макс. угловая скорость (рад/с)
    k_linear              float    0.6       — P-коэфф. линейной скорости
    k_angular             float    1.2       — P-коэфф. угловой скорости
    predict_horizon       float    1.0       — горизонт предсказания (с)
    target_history_size   int      10        — точек истории для оценки скорости
    target_lost_timeout   float    1.0       — с без обнаружения → поворот-поиск

    (плюс все параметры оригинального TrackerNode:
     search_angular_speed, stop_distance, stop_distance_tolerance,
     realtime_threshold_sec, segmentation_time_ema_alpha,
     slow_search_step_angle_deg, slow_search_angle_tolerance_deg)
"""

import argparse
import math
from collections import deque

import numpy as np

import rclpy
from geometry_msgs.msg import Point, Twist

from object_tracking_2.tracker_node import TrackerNode
from object_tracking_2.tracking_performance import TrackingMode


# ─────────────────────────────────────────────────────────────────────────────
# P-контроллер преследования / перехвата
# ─────────────────────────────────────────────────────────────────────────────

class DynamicPursuitController:
    """
    Реактивный P-контроллер для преследования или перехвата цели.

    Режим pursuit:
        Едет прямо к текущей позиции цели, останавливается на
        desired_distance.

    Режим intercept:
        Оценивает скорость цели по истории наблюдений (линейная
        регрессия первого порядка: v = Δpos / Δt) и едет в
        предсказанную точку target_pos + v * T, где
        T = min(dist_to_target / v_max_robot, predict_horizon).

    Все вычисления в системе координат map (2D, xy).
    """

    def __init__(
        self,
        control_mode: str = 'pursuit',
        desired_distance: float = 0.6,
        max_linear_vel: float = 0.5,
        max_angular_vel: float = 1.5,
        k_linear: float = 0.6,
        k_angular: float = 1.2,
        predict_horizon: float = 1.0,
        stop_distance: float = 0.4,
        history_size: int = 10,
    ):
        self.control_mode = control_mode
        self.d_ref        = desired_distance
        self.v_max        = max_linear_vel
        self.w_max        = max_angular_vel
        self.k_linear     = k_linear
        self.k_angular    = k_angular
        self.T_pred       = predict_horizon
        self.d_stop       = stop_distance

        # Кольцевой буфер наблюдений: (timestamp_sec, np.array([x, y]))
        self._history: deque = deque(maxlen=history_size)

    # ------------------------------------------------------------------
    def update_target(self, x: float, y: float, timestamp_sec: float) -> None:
        """Сохраняет новое наблюдение позиции цели."""
        self._history.append((timestamp_sec, np.array([x, y])))

    def reset(self) -> None:
        """Сбрасывает историю (вызывается при смене промпта или потере цели)."""
        self._history.clear()

    # ------------------------------------------------------------------
    def _estimate_velocity(self) -> np.ndarray:
        """
        Линейная оценка скорости цели по крайним точкам истории (м/с).
        Возвращает нулевой вектор, если истории недостаточно.
        """
        if len(self._history) < 3:
            return np.zeros(2)
        t0, p0 = self._history[0]
        t1, p1 = self._history[-1]
        dt = t1 - t0
        if dt < 1e-3:
            return np.zeros(2)
        return (p1 - p0) / dt

    # ------------------------------------------------------------------
    def compute_cmd_vel(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        target_x: float,
        target_y: float,
    ) -> Twist:
        """
        Вычисляет Twist для публикации в /cmd_vel.

        Логика:
          1. Выбираем точку назначения (goal):
             - pursuit  → текущая позиция цели
             - intercept → экстраполированная позиция цели
          2. Вычисляем ошибки дистанции и угла.
          3. P-регулятор с насыщением.
          4. Замедление при большом угловом отклонении (> 45°).
        """
        cmd = Twist()

        # ── Выбор точки назначения ─────────────────────────────────
        if self.control_mode == 'intercept' and len(self._history) >= 3:
            v = self._estimate_velocity()
            dx_now   = target_x - robot_x
            dy_now   = target_y - robot_y
            dist_now = math.sqrt(dx_now ** 2 + dy_now ** 2)
            # Адаптивный горизонт: не больше времени, нужного роботу
            T      = min(dist_now / max(self.v_max, 0.01), self.T_pred)
            goal_x = target_x + v[0] * T
            goal_y = target_y + v[1] * T
        else:
            goal_x = target_x
            goal_y = target_y

        # ── Ошибки ────────────────────────────────────────────────
        dx       = goal_x - robot_x
        dy       = goal_y - robot_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance < self.d_stop:
            return cmd  # нулевой Twist → стоп

        dist_error   = distance - self.d_ref
        target_angle = math.atan2(dy, dx)
        angle_error  = self._normalize(target_angle - robot_yaw)

        # ── P-регулятор ───────────────────────────────────────────
        v_lin = self.k_linear  * dist_error
        v_ang = self.k_angular * angle_error

        # Замедление при большом угловом отклонении
        if abs(angle_error) > math.pi / 4:
            v_lin *= 0.3

        # Насыщение
        cmd.linear.x  = float(max(-self.v_max, min(self.v_max, v_lin)))
        cmd.angular.z = float(max(-self.w_max, min(self.w_max, v_ang)))
        return cmd

    @staticmethod
    def _normalize(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))


# ─────────────────────────────────────────────────────────────────────────────
# Динамический узел
# ─────────────────────────────────────────────────────────────────────────────

class DynamicTrackerNode(TrackerNode):
    """
    Узел преследования/перехвата движущихся объектов.

    Наследует всю инфраструктуру TrackerNode (подписки, TF, сегментатор,
    поворот-поиск, артефакты эпизода) и переопределяет только логику
    принятия решений о движении.

    Nav2 action client создаётся родителем, но не используется —
    управление передаётся напрямую через /cmd_vel.
    """

    def __init__(self, segmentator_name: str):
        # Родитель создаёт все подписки, публикации, TF, таймер
        super().__init__(segmentator_name)

        # Меняем имя ноды для различия в ros2 node list
        # (нельзя после super().__init__, но можно залогировать)
        self.get_logger().info('DynamicTrackerNode: overriding to dynamic mode')

        # ── Дополнительные параметры ──────────────────────────────
        self.declare_parameter('control_mode',        'pursuit')
        self.declare_parameter('desired_distance',     0.6)
        self.declare_parameter('max_linear_vel',       0.5)
        self.declare_parameter('max_angular_vel',      1.5)
        self.declare_parameter('k_linear',             0.6)
        self.declare_parameter('k_angular',            1.2)
        self.declare_parameter('predict_horizon',      1.0)
        self.declare_parameter('target_history_size',  10)
        self.declare_parameter('target_lost_timeout',  1.0)

        def gd(n): return self.get_parameter(n).get_parameter_value().double_value
        def gi(n): return self.get_parameter(n).get_parameter_value().integer_value
        def gs(n): return self.get_parameter(n).get_parameter_value().string_value

        control_mode       = gs('control_mode')
        desired_distance   = gd('desired_distance')
        max_linear_vel     = gd('max_linear_vel')
        max_angular_vel    = gd('max_angular_vel')
        k_linear           = gd('k_linear')
        k_angular          = gd('k_angular')
        predict_horizon    = gd('predict_horizon')
        target_history_size = gi('target_history_size')
        self._lost_timeout = gd('target_lost_timeout')

        # ── Контроллер ────────────────────────────────────────────
        self._controller = DynamicPursuitController(
            control_mode    = control_mode,
            desired_distance= desired_distance,
            max_linear_vel  = max_linear_vel,
            max_angular_vel = max_angular_vel,
            k_linear        = k_linear,
            k_angular       = k_angular,
            predict_horizon = predict_horizon,
            stop_distance   = self.stop_distance,   # из родителя
            history_size    = target_history_size,
        )
        self._last_detection_time: float | None = None

        self.get_logger().info(
            f'DynamicTrackerNode ready: '
            f'segmentator={self.segmentator.name} '
            f'control_mode={control_mode} '
            f'desired_distance={desired_distance} m'
        )

    # ------------------------------------------------------------------
    # Переопределённые методы
    # ------------------------------------------------------------------

    def _should_skip_frame(self, mode: TrackingMode, now) -> bool:
        """
        В динамическом режиме обрабатываем каждый кадр пока цель не достигнута.
        Nav2 не используется, поэтому nav2_goal_active и search_rotation_active
        не блокируют обработку.
        """
        if self.current_prompt is None or self.current_prompt.strip() == '':
            return True
        return self.target_reached

    def _handle_missing_detection(self, mode: TrackingMode) -> None:
        """
        Объект не найден на текущем кадре.
        Таймаут потери (target_lost_timeout) отслеживается в timer_callback.
        Здесь ничего не делаем — не прерываем движение из-за одного пропущенного кадра.
        """
        pass

    def _handle_detected_target(
        self,
        center_coords,
        depth,
        now,
        mode: TrackingMode,
        segmented_image,
        image_header,
    ) -> None:
        """
        Объект найден на кадре.

        1. Вычисляем 3D-позицию цели в map frame.
        2. Обновляем историю контроллера.
        3. Публикуем артефакты первого обнаружения.
        4. Проверяем достижение цели.
        5. Вычисляем и публикуем cmd_vel.
        """
        self._stop_search_rotation()

        if not self.target_found:
            self.get_logger().info(
                f'Цель найдена: px=({center_coords[0]}, {center_coords[1]})'
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
        except Exception as e:
            self.get_logger().error(f'Ошибка TF camera→map: {e}')
            return

        # ── Обновление контроллера ────────────────────────────────
        now_sec = now.nanoseconds * 1e-9
        self._controller.update_target(point_world.x, point_world.y, now_sec)
        self._last_detection_time = now_sec

        # ── Артефакты первого обнаружения ────────────────────────
        self._publish_episode_start_artifacts(
            point_world=point_world,
            segmented_image=segmented_image,
            image_header=image_header,
        )

        # ── Поза робота ───────────────────────────────────────────
        try:
            robot_x, robot_y = self._get_robot_position_in_map()
            robot_yaw        = self._get_robot_yaw_in_map()
        except Exception as e:
            self.get_logger().error(f'Ошибка TF base_link→map: {e}')
            return

        # ── Дистанция до цели ─────────────────────────────────────
        dx   = point_world.x - robot_x
        dy   = point_world.y - robot_y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        self.get_logger().info(
            f'target=({point_world.x:.2f}, {point_world.y:.2f}) '
            f'dist={dist:.2f} m  mode={self._controller.control_mode}'
        )

        # ── Достижение цели ───────────────────────────────────────
        if dist <= self.stop_distance + self.stop_distance_tolerance:
            if not self.target_reached:
                self.target_reached = True
                self._publish_target_reached(True)
                self._publish_stop()
                self.get_logger().info(
                    f'Цель достигнута на расстоянии {dist:.2f} м!'
                )
                self._log_avg_seg_time('Average segmentation time:')
            return

        # ── Команда скорости ──────────────────────────────────────
        cmd = self._controller.compute_cmd_vel(
            robot_x, robot_y, robot_yaw,
            point_world.x, point_world.y,
        )
        self.search_cmd_pub.publish(cmd)

    def timer_callback(self) -> None:
        """
        Таймер 10 Гц.

        Отвечает за:
          1. Стоп при отсутствии промпта.
          2. Таймаут потери цели → стоп + поворот-поиск.
          3. Обновление текущего дискретного поворота-поиска.
        """
        if self.current_prompt is None or self.current_prompt.strip() == '':
            self._stop_search_rotation()
            return

        if self.target_reached:
            return

        # Таймаут потери цели
        if self._last_detection_time is not None:
            now_sec  = self.get_clock().now().nanoseconds * 1e-9
            lost_for = now_sec - self._last_detection_time

            if lost_for > self._lost_timeout:
                self._publish_stop()
                if not self.search_rotation_active:
                    self.get_logger().warn(
                        f'Цель потеряна {lost_for:.1f} с — начинаем поворот-поиск'
                    )
                    self.target_found = False
                    self._controller.reset()
                    self._start_search_rotation()

        # Обновление текущего поворота
        if self.search_rotation_active:
            self._update_discrete_search_rotation()

    def _reset_tracking_state(self) -> None:
        """Сброс состояния + история контроллера."""
        super()._reset_tracking_state()
        self._controller.reset()
        self._last_detection_time = None


# ─────────────────────────────────────────────────────────────────────────────

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='ROS2 dynamic tracker node')
    parser.add_argument(
        '--segmentator',
        required=True,
        choices=['CLIP', 'DinoSAM', 'SEEM', 'OpenSeeD', 'SAM2', 'DINOv2'],
        help='Имя используемого сегментатора',
    )
    return parser.parse_args(args)


def main(args=None):
    non_ros_args = rclpy.utilities.remove_ros_args(args=args)[1:]
    parsed_args  = parse_args(non_ros_args)

    rclpy.init(args=args)

    node = DynamicTrackerNode(segmentator_name=parsed_args.segmentator)

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
