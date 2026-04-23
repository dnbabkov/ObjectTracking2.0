"""
tracker_node_dynamic.py
=======================
Узел для преследования и перехвата ДВИЖУЩИХСЯ объектов с объездом препятствий.

Все параметры робота зашиты из XACRO-описания:

  Камера (depth_camera.xacro):
    topic=/depth_camera, horizontal_fov=1.089, width=640, height=480
    → fx = fy = 640 / (2*tan(1.089/2)) = 528.43
    → cx = 320.0, cy = 240.0
    ROS2 топики (gz-ros2 bridge):
      RGB:   /camera/image_raw
      Depth: /depth_camera/depth/image_raw
      Info:  /camera/camera_info  ← camera_info_callback читает fx/fy/cx/cy оттуда
             (захардкоженные значения используются как fallback если /camera_info не пришёл)

  Лидар (lidar.xacro):
    topic=/lidar, samples=360, range_min=0.3, range_max=12.0, rate=10 Hz
    ROS2 топик через gz-ros2 bridge: /lidar
    Если в вашем bridge настроен remapping на /scan — измените LASER_TOPIC ниже.

  Размеры робота (robot_core.xacro):
    chassis: 0.45 × 0.30 × 0.15 м
    wheel_offset_y = 0.175 м (от центра до колеса)
    wheel_thickness = 0.04 м
    Внешний радиус (до кромки колеса) = 0.175 + 0.04/2 = 0.195 м

  Привод (ros2_control.xacro):
    дифференциальный привод, /cmd_vel → gz_ros2_control

Архитектура управления (два слоя):
─────────────────────────────────────────────────────────────────
  Слой 1 — DynamicPursuitController
      Вычисляет желаемое направление (heading) и линейную скорость.
      Режим pursuit:   едет к текущей позиции цели.
      Режим intercept: экстраполирует позицию цели на T секунд вперёд.

  Слой 2 — VFHAvoidance (Vector Field Histogram)
      Читает /lidar (LaserScan), строит полярную гистограмму препятствий,
      находит ближайшую свободную «долину» к желаемому направлению,
      корректирует угловую скорость и при необходимости тормозит.
      Если препятствий нет — прозрачен (Слой 1 проходит без изменений).
─────────────────────────────────────────────────────────────────

Запуск (все параметры опциональны — дефолты соответствуют роботу):
    ros2 run object_tracking_2 tracker_node_dynamic --segmentator SAM2

Переопределение при необходимости:
    ros2 run object_tracking_2 tracker_node_dynamic --segmentator SAM2 \\
        --ros-args \\
            -p control_mode:=intercept \\
            -p desired_distance:=0.8
"""

import argparse
import math
from collections import deque
from typing import Optional

import numpy as np

import rclpy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

from object_tracking_2.tracker_node import TrackerNode
from object_tracking_2.tracking_performance import TrackingMode


# ─────────────────────────────────────────────────────────────────────────────
# Константы из XACRO-описания робота
# ─────────────────────────────────────────────────────────────────────────────

# ── Топики (gz-ros2 bridge) ───────────────────────────────────────────────
# Из lidar.xacro: <topic>/lidar</topic>
# Стандартный gz-ros2 bridge публикует LaserScan на том же имени.
# Если в вашем bridge.yaml настроен remapping → поменяйте здесь.
LASER_TOPIC = '/lidar'

# ── Параметры камеры (depth_camera.xacro) ─────────────────────────────────
# horizontal_fov = 1.089 рад,  width = 640,  height = 480
# fx = fy = width / (2 * tan(fov/2))
_FOV_H     = 1.089
_IMG_W     = 640
_IMG_H     = 480
CAMERA_FX  = _IMG_W / (2.0 * math.tan(_FOV_H / 2.0))   # = 528.43 px
CAMERA_FY  = CAMERA_FX                                   # квадратные пиксели
CAMERA_CX  = _IMG_W / 2.0                               # = 320.0  px
CAMERA_CY  = _IMG_H / 2.0                               # = 240.0  px

# ── Размеры робота (robot_core.xacro) ─────────────────────────────────────
# wheel_offset_y = 0.175 м,  wheel_thickness = 0.04 м
# Внешняя кромка колеса от центра base_link:
ROBOT_WHEEL_OUTER_RADIUS = 0.175 + 0.04 / 2.0           # = 0.195 м
# Прибавляем зазор безопасности
ROBOT_SAFETY_MARGIN      = 0.10                          # м
ROBOT_RADIUS             = ROBOT_WHEEL_OUTER_RADIUS      # = 0.195 м

# ── Лидар (lidar.xacro) ───────────────────────────────────────────────────
LIDAR_RANGE_MIN = 0.3    # м — из <range><min>
LIDAR_RANGE_MAX = 12.0   # м — из <range><max>
LIDAR_RATE_HZ   = 10     # Гц — из <update_rate>

# ── TF-фреймы (robot_core.xacro, depth_camera.xacro) ─────────────────────
# depth_camera_link_optical — оптический фрейм глубинной камеры
# Используется в TrackerNode._camera_point_to_world()
DEPTH_CAMERA_FRAME = 'depth_camera_link_optical'


# ═════════════════════════════════════════════════════════════════════════════
# Слой 1: Pursuit / Intercept контроллер
# ═════════════════════════════════════════════════════════════════════════════

class DynamicPursuitController:
    """
    P-контроллер для преследования или перехвата движущейся цели.

    compute() → (v_linear, desired_heading_map, dist_to_goal)
    Эти значения передаются в VFH, который их при необходимости корректирует.

    Режим pursuit:
        desired_heading → направление на текущую позицию цели.

    Режим intercept:
        desired_heading → направление на предсказанную позицию цели
        target_pos + v_target * T, где T = min(dist/v_max, predict_horizon).
    """

    def __init__(
        self,
        control_mode:     str   = 'pursuit',
        desired_distance: float = 0.6,
        max_linear_vel:   float = 0.5,
        max_angular_vel:  float = 1.5,
        k_linear:         float = 0.6,
        k_angular:        float = 1.2,
        predict_horizon:  float = 1.0,
        stop_distance:    float = 0.4,
        history_size:     int   = 10,
    ):
        self.control_mode = control_mode
        self.d_ref        = desired_distance
        self.v_max        = max_linear_vel
        self.w_max        = max_angular_vel
        self.k_linear     = k_linear
        self.k_angular    = k_angular
        self.T_pred       = predict_horizon
        self.d_stop       = stop_distance
        self._history: deque = deque(maxlen=history_size)

    def update_target(self, x: float, y: float, t: float) -> None:
        self._history.append((t, np.array([x, y])))

    def reset(self) -> None:
        self._history.clear()

    def _estimate_velocity(self) -> np.ndarray:
        """Линейная оценка скорости цели по крайним точкам истории."""
        if len(self._history) < 3:
            return np.zeros(2)
        t0, p0 = self._history[0]
        t1, p1 = self._history[-1]
        dt = t1 - t0
        if dt < 1e-3:
            return np.zeros(2)
        return (p1 - p0) / dt

    def compute(
        self,
        robot_x: float, robot_y: float, robot_yaw: float,
        target_x: float, target_y: float,
    ) -> tuple[float, float, float]:
        """
        Возвращает (v_linear, desired_heading_map, dist_to_goal).
        desired_heading_map — абсолютный угол в map frame (рад).
        """
        if self.control_mode == 'intercept' and len(self._history) >= 3:
            v = self._estimate_velocity()
            dx_now   = target_x - robot_x
            dy_now   = target_y - robot_y
            dist_now = math.sqrt(dx_now ** 2 + dy_now ** 2)
            T      = min(dist_now / max(self.v_max, 0.01), self.T_pred)
            goal_x = target_x + v[0] * T
            goal_y = target_y + v[1] * T
        else:
            goal_x = target_x
            goal_y = target_y

        dx   = goal_x - robot_x
        dy   = goal_y - robot_y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < self.d_stop:
            return 0.0, robot_yaw, dist

        desired_heading = math.atan2(dy, dx)
        dist_error      = max(0.0, dist - self.d_ref)
        v_linear        = min(self.v_max, self.k_linear * dist_error)

        return v_linear, desired_heading, dist


# ═════════════════════════════════════════════════════════════════════════════
# Слой 2: VFH (Vector Field Histogram) obstacle avoidance
# ═════════════════════════════════════════════════════════════════════════════

class VFHAvoidance:
    """
    Реактивный объезд препятствий на основе Vector Field Histogram.

    Настроен под лидар робота:
      360 лучей, range_min=0.3 м, range_max=12.0 м, 10 Гц.

    Алгоритм за один вызов compute_cmd_vel():

    1. Полярная гистограмма плотности препятствий строится в update_scan():
         h[k] += (1 - r/d_inf)^2  для каждого луча в сектор k.
       Ближние препятствия дают квадратично больший вклад.

    2. Бинаризация: сектор занят если h[k] >= obstacle_threshold.

    3. Поиск долин — непрерывных последовательностей свободных секторов.
       Выбираем долину с центром, ближайшим к desired_heading.

    4. Blend угловой скорости:
         blend=0 → VFH не вмешивается (путь свободен)
         blend=1 → VFH полностью управляет (препятствие перекрывает путь)
       Переход плавный по количеству отклонённых секторов.

    5. Торможение пропорционально расстоянию до ближайшего препятствия
       в переднем конусе ±30°.

    6. Аварийная остановка при препятствии ближе d_stop: стоп +
       разворот в сторону с меньшей плотностью препятствий.
    """

    def __init__(
        self,
        # Параметры рассчитаны под конкретного робота (robot_core.xacro)
        num_sectors:          int   = 72,    # 5° на сектор, 360/72=5°
        obstacle_threshold:   float = 5.0,   # порог занятости сектора
        robot_radius:         float = ROBOT_RADIUS,          # 0.195 м
        safety_dist:          float = ROBOT_SAFETY_MARGIN,   # 0.10 м
        influence_dist:       float = 1.5,   # дальность влияния (м)
        max_steer_correction: float = 0.8,   # макс. поправка VFH (рад/с)
        slow_down_dist:       float = 0.6,   # начало торможения (м от препятствия)
        stop_dist:            float = 0.25,  # полная остановка (м от препятствия)
        max_angular_vel:      float = 1.5,   # рад/с
    ):
        self.N         = num_sectors
        self.threshold = obstacle_threshold
        self.r_robot   = robot_radius
        self.d_safety  = safety_dist
        self.d_inf     = influence_dist
        self.k_steer   = max_steer_correction
        self.d_slow    = slow_down_dist
        self.d_stop    = stop_dist
        self.w_max     = max_angular_vel

        self._histogram:      Optional[np.ndarray] = None
        self._min_front_dist: float                = float('inf')

    def update_scan(self, msg: LaserScan) -> None:
        """
        Строит полярную гистограмму из одного лидарного скана.
        Вызывается в _scan_callback() при каждом новом скане (~10 Гц).
        """
        histogram  = np.zeros(self.N)
        min_front  = float('inf')
        sector_rad = 2.0 * math.pi / self.N

        angle = msg.angle_min
        for r in msg.ranges:
            angle += msg.angle_increment
            if math.isnan(r) or math.isinf(r):
                continue
            if r < msg.range_min or r > msg.range_max:
                continue
            if r > self.d_inf:
                continue

            sector = int((angle % (2.0 * math.pi)) / sector_rad) % self.N
            weight = (1.0 - r / self.d_inf) ** 2
            histogram[sector] += weight

            # Передний конус ±30° для отслеживания минимальной дистанции
            if abs(self._norm(angle)) < math.radians(30):
                min_front = min(min_front, r)

        self._histogram      = histogram
        self._min_front_dist = min_front

    def compute_cmd_vel(
        self,
        v_desired:       float,   # желаемая линейная скорость (от Слоя 1)
        desired_heading: float,   # желаемое направление в map frame (рад)
        robot_yaw:       float,   # текущий yaw робота (рад)
        w_from_pursuit:  float,   # угловая скорость от pursuit-контроллера
    ) -> Twist:
        """
        Корректирует команду движения с учётом препятствий.
        Возвращает итоговый Twist.
        """
        cmd = Twist()

        # Нет данных лидара — прозрачный режим (pursuit без изменений)
        if self._histogram is None:
            cmd.linear.x  = float(v_desired)
            cmd.angular.z = float(w_from_pursuit)
            return cmd

        # ── Аварийная остановка ───────────────────────────────────
        hard_stop_dist = self.d_stop + self.r_robot + self.d_safety
        if self._min_front_dist < hard_stop_dist:
            cmd.angular.z = float(self._emergency_turn())
            return cmd

        # ── Бинарная маска свободных секторов ────────────────────
        free_mask = self._histogram < self.threshold

        # Желаемое направление в robot frame → индекс сектора
        sector_rad     = 2.0 * math.pi / self.N
        heading_robot  = self._norm(desired_heading - robot_yaw)
        desired_sector = int((heading_robot % (2.0 * math.pi)) / sector_rad) % self.N

        # ── Поиск лучшей свободной долины ────────────────────────
        best_sector = self._find_best_valley(free_mask, desired_sector)

        if best_sector is None:
            # Полная блокировка — аварийный разворот
            cmd.angular.z = float(self._emergency_turn())
            return cmd

        # ── Угловая коррекция (blend pursuit ↔ VFH) ──────────────
        best_heading_robot = best_sector * sector_rad
        if best_heading_robot > math.pi:
            best_heading_robot -= 2.0 * math.pi

        # Отклонение лучшей долины от желаемого сектора
        sector_diff = abs(
            (best_sector - desired_sector + self.N // 2) % self.N - self.N // 2
        )
        # blend нарастает от 0 до 1 при отклонении от 0 до 60° (N/6 секторов)
        blend = min(1.0, sector_diff / (self.N / 6.0))

        angle_to_free = self._norm(best_heading_robot - heading_robot)
        w_vfh    = self.k_steer * angle_to_free
        w_result = (1.0 - blend) * w_from_pursuit + blend * w_vfh
        w_result = max(-self.w_max, min(self.w_max, w_result))

        # ── Торможение при приближении к препятствию впереди ──────
        v_result      = v_desired
        eff_stop_dist = self.d_stop + self.r_robot + self.d_safety
        eff_slow_dist = self.d_slow + self.r_robot + self.d_safety

        if self._min_front_dist < eff_slow_dist:
            ratio    = max(
                0.0,
                (self._min_front_dist - eff_stop_dist)
                / (eff_slow_dist - eff_stop_dist),
            )
            v_result = v_desired * ratio

        # Дополнительное замедление при активном рулении
        if abs(w_result) > self.w_max * 0.4:
            v_result *= max(0.2, 1.0 - abs(w_result) / self.w_max)

        cmd.linear.x  = float(max(0.0, v_result))
        cmd.angular.z = float(w_result)
        return cmd

    def _find_best_valley(
        self,
        free_mask:      np.ndarray,
        desired_sector: int,
    ) -> Optional[int]:
        """
        Возвращает индекс центрального сектора ближайшей свободной долины
        к desired_sector, или None если свободных секторов нет.

        Используем тайлирование ×3 для корректной обработки перехода 0/2π.
        """
        if not np.any(free_mask):
            return None

        N      = self.N
        mask3  = np.tile(free_mask, 3)
        offset = N   # центральная копия

        valleys = []
        in_v    = False
        start   = 0
        for i in range(N * 3):
            if mask3[i] and not in_v:
                start = i; in_v = True
            elif not mask3[i] and in_v:
                valleys.append((start, i - 1)); in_v = False
        if in_v:
            valleys.append((start, N * 3 - 1))

        if not valleys:
            return None

        target     = desired_sector + offset
        best_diff  = float('inf')
        best_center = None

        for (s, e) in valleys:
            center = (s + e) // 2
            diff   = abs(center - target)
            if diff < best_diff:
                best_diff   = diff
                best_center = center

        return None if best_center is None else (best_center - offset) % N

    def _emergency_turn(self) -> float:
        """
        Разворачивается в сторону с наименьшей суммарной плотностью препятствий.
        """
        if self._histogram is None:
            return self.w_max
        left  = np.sum(self._histogram[: self.N // 2])
        right = np.sum(self._histogram[self.N // 2:])
        return self.w_max if right < left else -self.w_max

    @staticmethod
    def _norm(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))


# ═════════════════════════════════════════════════════════════════════════════
# Основной узел
# ═════════════════════════════════════════════════════════════════════════════

class DynamicTrackerNode(TrackerNode):
    """
    Узел преследования/перехвата движущихся объектов с объездом препятствий.

    Наследует TrackerNode и переопределяет только методы принятия решений
    о движении. Топики, TF, сегментатор, поворот-поиск, артефакты эпизода —
    всё унаследовано без изменений.

    Поток управления при обнаружении цели:
        pixel + depth → Point camera → Point map  (унаследовано)
        DynamicPursuitController.compute()
            → (v_desired, heading_desired, dist)
        w_pursuit = k_angular * angle_error        (P-регулятор угла)
        VFHAvoidance.compute_cmd_vel()
            → Twist → /cmd_vel
    """

    # ── Дефолтные параметры pursuit (можно переопределить через --ros-args) ─
    _DEFAULT_CONTROL_MODE        = 'pursuit'   # pursuit | intercept
    _DEFAULT_DESIRED_DISTANCE    = 0.6         # м — желаемая дистанция до цели
    _DEFAULT_MAX_LINEAR_VEL      = 0.5         # м/с
    _DEFAULT_MAX_ANGULAR_VEL     = 1.5         # рад/с
    _DEFAULT_K_LINEAR            = 0.6         # P-коэфф. линейной скорости
    _DEFAULT_K_ANGULAR           = 1.2         # P-коэфф. угловой скорости
    _DEFAULT_PREDICT_HORIZON     = 1.0         # с (для intercept)
    _DEFAULT_TARGET_HISTORY_SIZE = 10          # точек для оценки скорости
    _DEFAULT_TARGET_LOST_TIMEOUT = 1.0         # с без обнаружения → поиск

    # ── Дефолтные параметры VFH (рассчитаны под робота) ─────────────────────
    _DEFAULT_USE_AVOIDANCE        = True
    _DEFAULT_VFH_NUM_SECTORS      = 72          # 5° на сектор
    _DEFAULT_VFH_THRESHOLD        = 5.0         # порог занятости
    _DEFAULT_VFH_ROBOT_RADIUS     = ROBOT_RADIUS           # 0.195 м
    _DEFAULT_VFH_SAFETY_DIST      = ROBOT_SAFETY_MARGIN    # 0.10  м
    _DEFAULT_VFH_INFLUENCE_DIST   = 1.5         # м
    _DEFAULT_VFH_STEER_CORRECTION = 0.8         # рад/с
    _DEFAULT_VFH_SLOW_DOWN_DIST   = 0.6         # м
    _DEFAULT_VFH_STOP_DIST        = 0.25        # м

    def __init__(self, segmentator_name: str):
        super().__init__(segmentator_name)
        self.get_logger().info('DynamicTrackerNode: initializing')

        # ── Объявление параметров с захардкоженными дефолтами ─────
        self.declare_parameter('control_mode',           self._DEFAULT_CONTROL_MODE)
        self.declare_parameter('desired_distance',        self._DEFAULT_DESIRED_DISTANCE)
        self.declare_parameter('max_linear_vel',          self._DEFAULT_MAX_LINEAR_VEL)
        self.declare_parameter('max_angular_vel',         self._DEFAULT_MAX_ANGULAR_VEL)
        self.declare_parameter('k_linear',                self._DEFAULT_K_LINEAR)
        self.declare_parameter('k_angular',               self._DEFAULT_K_ANGULAR)
        self.declare_parameter('predict_horizon',         self._DEFAULT_PREDICT_HORIZON)
        self.declare_parameter('target_history_size',     self._DEFAULT_TARGET_HISTORY_SIZE)
        self.declare_parameter('target_lost_timeout',     self._DEFAULT_TARGET_LOST_TIMEOUT)

        self.declare_parameter('use_obstacle_avoidance',  self._DEFAULT_USE_AVOIDANCE)
        self.declare_parameter('laser_topic',             LASER_TOPIC)
        self.declare_parameter('vfh_num_sectors',         self._DEFAULT_VFH_NUM_SECTORS)
        self.declare_parameter('vfh_obstacle_threshold',  self._DEFAULT_VFH_THRESHOLD)
        self.declare_parameter('vfh_robot_radius',        self._DEFAULT_VFH_ROBOT_RADIUS)
        self.declare_parameter('vfh_safety_dist',         self._DEFAULT_VFH_SAFETY_DIST)
        self.declare_parameter('vfh_influence_dist',      self._DEFAULT_VFH_INFLUENCE_DIST)
        self.declare_parameter('vfh_max_steer_correction', self._DEFAULT_VFH_STEER_CORRECTION)
        self.declare_parameter('vfh_slow_down_dist',      self._DEFAULT_VFH_SLOW_DOWN_DIST)
        self.declare_parameter('vfh_stop_dist',           self._DEFAULT_VFH_STOP_DIST)

        def gd(n): return self.get_parameter(n).get_parameter_value().double_value
        def gi(n): return self.get_parameter(n).get_parameter_value().integer_value
        def gs(n): return self.get_parameter(n).get_parameter_value().string_value
        def gb(n): return self.get_parameter(n).get_parameter_value().bool_value

        control_mode        = gs('control_mode')
        desired_distance    = gd('desired_distance')
        max_linear_vel      = gd('max_linear_vel')
        max_angular_vel     = gd('max_angular_vel')
        k_linear            = gd('k_linear')
        k_angular           = gd('k_angular')
        predict_horizon     = gd('predict_horizon')
        target_history_size = gi('target_history_size')
        self._lost_timeout  = gd('target_lost_timeout')
        self._use_avoidance = gb('use_obstacle_avoidance')
        laser_topic         = gs('laser_topic')

        # ── Контроллер преследования ──────────────────────────────
        self._controller = DynamicPursuitController(
            control_mode    = control_mode,
            desired_distance= desired_distance,
            max_linear_vel  = max_linear_vel,
            max_angular_vel = max_angular_vel,
            k_linear        = k_linear,
            k_angular       = k_angular,
            predict_horizon = predict_horizon,
            stop_distance   = self.stop_distance,   # из TrackerNode
            history_size    = target_history_size,
        )
        self._k_angular = k_angular
        self._w_max     = max_angular_vel

        # ── VFH — параметры рассчитаны под робота ─────────────────
        self._vfh = VFHAvoidance(
            num_sectors          = gi('vfh_num_sectors'),
            obstacle_threshold   = gd('vfh_obstacle_threshold'),
            robot_radius         = gd('vfh_robot_radius'),
            safety_dist          = gd('vfh_safety_dist'),
            influence_dist       = gd('vfh_influence_dist'),
            max_steer_correction = gd('vfh_max_steer_correction'),
            slow_down_dist       = gd('vfh_slow_down_dist'),
            stop_dist            = gd('vfh_stop_dist'),
            max_angular_vel      = max_angular_vel,
        )

        if self._use_avoidance:
            self.create_subscription(
                LaserScan,
                laser_topic,
                self._scan_callback,
                rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value,
            )
            self.get_logger().info(
                f'VFH obstacle avoidance ON  '
                f'(laser={laser_topic}  '
                f'robot_radius={ROBOT_RADIUS:.3f} m  '
                f'safety={ROBOT_SAFETY_MARGIN:.2f} m)'
            )
        else:
            self.get_logger().info('VFH obstacle avoidance DISABLED')

        self._last_detection_time: Optional[float] = None

        self.get_logger().info(
            f'DynamicTrackerNode ready | '
            f'segmentator={self.segmentator.name}  '
            f'control_mode={control_mode}  '
            f'desired_distance={desired_distance} m  '
            f'avoidance={self._use_avoidance}'
        )

        # ── Лог параметров камеры для диагностики ────────────────
        self.get_logger().info(
            f'Camera intrinsics (from depth_camera.xacro): '
            f'fx={CAMERA_FX:.2f}  fy={CAMERA_FY:.2f}  '
            f'cx={CAMERA_CX:.1f}  cy={CAMERA_CY:.1f}  '
            f'(override via /camera/camera_info)'
        )

    # ------------------------------------------------------------------
    def _scan_callback(self, msg: LaserScan) -> None:
        """Обновляем VFH-гистограмму при каждом новом скане лидара (10 Гц)."""
        self._vfh.update_scan(msg)

    # ------------------------------------------------------------------
    # Переопределённые методы TrackerNode
    # ------------------------------------------------------------------

    def _should_skip_frame(self, mode: TrackingMode, now) -> bool:
        """Обрабатываем каждый кадр пока цель не достигнута."""
        if self.current_prompt is None or self.current_prompt.strip() == '':
            return True
        return self.target_reached

    def _handle_missing_detection(self, mode: TrackingMode) -> None:
        """
        Одиночный пропуск кадра не останавливает движение.
        Таймаут потери цели обрабатывается в timer_callback.
        """
        pass

    def _handle_detected_target(
        self,
        center_coords,
        depth,
        now,
        mode:            TrackingMode,
        segmented_image,
        image_header,
    ) -> None:
        """
        Главный метод управления движением при обнаружении цели.

        1. pixel + depth → Point camera → Point map
        2. Обновить историю DynamicPursuitController
        3. Проверить достижение цели
        4. DynamicPursuitController.compute() → (v_desired, heading, dist)
        5. w_pursuit = k_angular * angle_error
        6. VFHAvoidance.compute_cmd_vel() → Twist → /cmd_vel
        """
        self._stop_search_rotation()

        if not self.target_found:
            self.get_logger().info(
                f'Цель найдена: px=({center_coords[0]}, {center_coords[1]})'
            )
            self.target_found = True

        # ── 1. pixel → 3D map ─────────────────────────────────────
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

        # ── 2. Обновление истории контроллера ─────────────────────
        now_sec = now.nanoseconds * 1e-9
        self._controller.update_target(point_world.x, point_world.y, now_sec)
        self._last_detection_time = now_sec

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

        # ── 3. Расстояние и проверка достижения цели ──────────────
        dx   = point_world.x - robot_x
        dy   = point_world.y - robot_y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        self.get_logger().info(
            f'target=({point_world.x:.2f}, {point_world.y:.2f})  '
            f'dist={dist:.2f} m  ctrl={self._controller.control_mode}'
        )

        if dist <= self.stop_distance + self.stop_distance_tolerance:
            if not self.target_reached:
                self.target_reached = True
                self._publish_target_reached(True)
                self._publish_stop()
                self.get_logger().info(f'Цель достигнута на {dist:.2f} м!')
                self._log_avg_seg_time('Average segmentation time:')
            return

        # ── 4. Слой 1: желаемое движение ─────────────────────────
        v_desired, heading_desired, _ = self._controller.compute(
            robot_x, robot_y, robot_yaw,
            point_world.x, point_world.y,
        )

        # ── 5. Угловая скорость от pursuit ────────────────────────
        angle_error = math.atan2(
            math.sin(heading_desired - robot_yaw),
            math.cos(heading_desired - robot_yaw),
        )
        w_pursuit = max(-self._w_max, min(self._w_max, self._k_angular * angle_error))

        # ── 6. Слой 2: VFH корректирует команду ──────────────────
        if self._use_avoidance:
            cmd = self._vfh.compute_cmd_vel(
                v_desired       = v_desired,
                desired_heading = heading_desired,
                robot_yaw       = robot_yaw,
                w_from_pursuit  = w_pursuit,
            )
        else:
            cmd = Twist()
            if abs(angle_error) > math.pi / 4:
                v_desired *= 0.3
            cmd.linear.x  = float(v_desired)
            cmd.angular.z = float(w_pursuit)

        self.search_cmd_pub.publish(cmd)

    # ------------------------------------------------------------------
    def timer_callback(self) -> None:
        """10 Гц. Таймаут потери цели → стоп + поворот-поиск."""
        if self.current_prompt is None or self.current_prompt.strip() == '':
            self._stop_search_rotation()
            return

        if self.target_reached:
            return

        if self._last_detection_time is not None:
            now_sec  = self.get_clock().now().nanoseconds * 1e-9
            lost_for = now_sec - self._last_detection_time

            if lost_for > self._lost_timeout:
                self._publish_stop()
                if not self.search_rotation_active:
                    self.get_logger().warn(
                        f'Цель потеряна {lost_for:.1f} с — поворот-поиск'
                    )
                    self.target_found = False
                    self._controller.reset()
                    self._start_search_rotation()

        if self.search_rotation_active:
            self._update_discrete_search_rotation()

    # ------------------------------------------------------------------
    def _reset_tracking_state(self) -> None:
        super()._reset_tracking_state()
        self._controller.reset()
        self._last_detection_time = None


# ═════════════════════════════════════════════════════════════════════════════

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Dynamic tracker with VFH obstacle avoidance'
    )
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

if __name__ == '__main__':
    main()