from enum import Enum


class TrackingMode(str, Enum):
    REALTIME = 'realtime'
    SINGLE_GOAL = 'single_goal'


class SegmentatorPerformanceMonitor:
    def __init__(self, realtime_threshold_sec: float, ema_alpha: float = 0.3):
        self.realtime_threshold_sec = realtime_threshold_sec
        self.ema_alpha = ema_alpha

        self.ema_segmentation_time = None
        self.last_segmentation_time = None
        self.sample_count = 0

        self._mode = TrackingMode.REALTIME

    def update(self, segmentation_time: float) -> TrackingMode:
        self.last_segmentation_time = segmentation_time
        self.sample_count += 1

        if self.ema_segmentation_time is None:
            self.ema_segmentation_time = segmentation_time
        else:
            self.ema_segmentation_time = (
                self.ema_alpha * segmentation_time
                + (1.0 - self.ema_alpha) * self.ema_segmentation_time
            )

        if (
            self._mode == TrackingMode.REALTIME
            and self.ema_segmentation_time > self.realtime_threshold_sec
        ):
            self._mode = TrackingMode.SINGLE_GOAL

        return self._mode

    def current_mode(self) -> TrackingMode:
        return self._mode