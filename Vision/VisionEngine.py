import math

import cv2
import numpy as np
import Quartz
import Quartz.CoreGraphics as CG
from ultralytics import YOLO

from Vision.DiscardTracker import DiscardTracker

class VisionEngine:
    def __init__(self, model_path):
        """
        Initializes the YOLO model and Quartz window capture.
        """
        self.model = YOLO(model_path)
        self.discard_tracker = None
        self.discard_tracker_right = None
        self.discard_tracker_up = None
        self.discard_tracker_left = None
        self.discard_tracker_enabled = False

    # Capture Methods
    def _capture_window_image(self, window_id):
        if not window_id:
            return None

        cg_image = CG.CGWindowListCreateImage(
            CG.CGRectNull,
            CG.kCGWindowListOptionIncludingWindow,
            window_id,
            CG.kCGWindowImageBoundsIgnoreFraming
        )
        if not cg_image:
            return None

        width = CG.CGImageGetWidth(cg_image)
        height = CG.CGImageGetHeight(cg_image)
        bytes_per_row = CG.CGImageGetBytesPerRow(cg_image)

        provider = CG.CGImageGetDataProvider(cg_image)
        data = CG.CGDataProviderCopyData(provider)
        if not data:
            return None

        arr = np.frombuffer(data, dtype=np.uint8)
        arr = arr.reshape((height, bytes_per_row // 4, 4))
        arr = arr[:, :width, :]
        if arr.size == 0:
            return None

        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    def get_window_id(self, app_name):
        """Finds the window ID matching the app name."""
        
        if not app_name:
            return None
            
        window_list = CG.CGWindowListCopyWindowInfo(
            CG.kCGWindowListOptionAll | CG.kCGWindowListExcludeDesktopElements,
            CG.kCGNullWindowID
        )
        for win in window_list:
            name = win.get(CG.kCGWindowName, '')
            bounds = win.get(CG.kCGWindowBounds)
            win_id = win.get(CG.kCGWindowNumber)
            
            if name.lower() == app_name.lower() and bounds and bounds['Height'] > 50 and bounds['Width'] > 50:
                return win_id
        return None
    
    # Prediction

    def predict_hand(self, frame, roi):
        if frame is None or roi is None:
            return [], None
        
        x1, y1, x2, y2 = map(int, roi)
        hand_region = frame[y1:y2, x1:x2]
        results = self.model.predict(hand_region, imgsz=640, conf=0.15, iou=0.45, agnostic_nms=True, verbose=False)
        clean_boxes = self.remove_duplicates_by_distance(results[0].boxes, min_distance_px=25)
        return clean_boxes, results[0].plot()

    def predict(self, frame):
        """
        Run the YOLO model against the current frame.
        Use agnostic NMS to prevent identical overlapping bounds across classes.
        """
        if frame is None:
            return [], None
            
        # Hardcoding confidence and iou parameters for now
        results = self.model.predict(frame, imgsz=1280, conf=0.15, iou=0.45, agnostic_nms=True, verbose=False)
        
        clean_boxes = self.remove_duplicates_by_distance(results[0].boxes, min_distance_px=25)
        debug_frame = results[0].plot()
        discarded_tiles = []

        if self.discard_tracker_enabled and self.discard_tracker is not None:
            roi_down = self.discard_tracker.active_roi
            roi_right = self.discard_tracker_right.active_roi
            roi_up = self.discard_tracker_up.active_roi
            roi_left = self.discard_tracker_left.active_roi
            roi_boxes_down = []
            roi_boxes_right = []
            roi_boxes_up = []
            roi_boxes_left = []
            
            for i in range(len(clean_boxes)):
                box = clean_boxes[i]
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                b_cx = (bx1 + bx2) / 2
                b_cy = (by1 + by2) / 2
                if roi_down[0] <= b_cx <= roi_down[2] and roi_down[1] <= b_cy <= roi_down[3]:
                    roi_boxes_down.append(box)
                if roi_right[0] <= b_cx <= roi_right[2] and roi_right[3] <= b_cy <= roi_right[1]:
                    roi_boxes_right.append(box)
                if roi_up[2] <= b_cx <= roi_up[0] and roi_up[3] <= b_cy <= roi_up[1]:
                    roi_boxes_up.append(box)
                if roi_left[2] <= b_cx <= roi_left[0] and roi_left[1] <= b_cy <= roi_left[3]:
                    roi_boxes_left.append(box)
            
            discarded_tiles += self.discard_tracker.process_frame(frame, roi_boxes_down, self.model.names)
            debug_frame = self.discard_tracker.draw_active_roi(debug_frame)
            debug_frame = self.discard_tracker.draw_discard_region(debug_frame)
            discarded_tiles += self.discard_tracker_right.process_frame(frame, roi_boxes_right, self.model.names)
            debug_frame = self.discard_tracker_right.draw_active_roi(debug_frame)
            debug_frame = self.discard_tracker_right.draw_discard_region(debug_frame)
            discarded_tiles += self.discard_tracker_up.process_frame(frame, roi_boxes_up, self.model.names)
            debug_frame = self.discard_tracker_up.draw_active_roi(debug_frame)
            debug_frame = self.discard_tracker_up.draw_discard_region(debug_frame)
            discarded_tiles += self.discard_tracker_left.process_frame(frame, roi_boxes_left, self.model.names)
            debug_frame = self.discard_tracker_left.draw_active_roi(debug_frame)
            debug_frame = self.discard_tracker_left.draw_discard_region(debug_frame)

        return clean_boxes, debug_frame, discarded_tiles
    
    # Prediction Helpers

    def remove_duplicates_by_distance(self, yolo_boxes, min_distance_px=20):
        """
        Filters out duplicate tiles based on physical distance using vectorized operations.
        """
        if len(yolo_boxes) == 0:
            return yolo_boxes

        xyxy = yolo_boxes.xyxy.cpu().numpy()
        centers = np.column_stack(((xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2))
        confidences = yolo_boxes.conf.cpu().numpy()

        sorted_indices = np.argsort(-confidences)
        keep_indices = []

        if len(sorted_indices) > 0:
            keep_indices.append(sorted_indices[0])
            for idx in sorted_indices[1:]:
                # Compute distance between the current center and all kept centers
                dist = np.linalg.norm(centers[keep_indices] - centers[idx], axis=1)
                if not np.any(dist < min_distance_px):
                    keep_indices.append(idx)

        return yolo_boxes[keep_indices]
    
    # Discard Handlers

    def update_discard_tracker(self, settings):
        enabled = settings.get("discard_tracker_enabled", False)
        confirm_frames = max(1, int(settings.get("discard_tracker_confirm_frames", 3)))
        
        p_bottom = settings.get("player_bottom", {})
        p_right = settings.get("player_right", {})
        p_top = settings.get("player_top", {})
        p_left = settings.get("player_left", {})

        tracker_is_valid = enabled and p_bottom.get("box_w", 0) > 0 and p_bottom.get("box_h", 0) > 0
        if not tracker_is_valid:
            self.discard_tracker = None
            self.discard_tracker_right = None
            self.discard_tracker_up = None
            self.discard_tracker_left = None
            self.discard_tracker_enabled = False
            return

        desired_state = (
            p_bottom.get("start_x", 0), p_bottom.get("start_y", 0), p_bottom.get("box_w", 0), p_bottom.get("box_h", 0),
            p_right.get("start_x", 0), p_right.get("start_y", 0), p_right.get("box_w", 0), p_right.get("box_h", 0),
            p_top.get("start_x", 0), p_top.get("start_y", 0), p_top.get("box_w", 0), p_top.get("box_h", 0),
            p_left.get("start_x", 0), p_left.get("start_y", 0), p_left.get("box_w", 0), p_left.get("box_h", 0),
            confirm_frames
        )
        current_state = getattr(self.discard_tracker, "_config_state", None)
        
        if self.discard_tracker is None or current_state != desired_state:
            self.discard_tracker = DiscardTracker(
                p_bottom.get("start_x", 0), p_bottom.get("start_y", 0), p_bottom.get("box_w", 0), p_bottom.get("box_h", 0), confirm_frames=confirm_frames
            )
            self.discard_tracker_right = DiscardTracker(
                p_right.get("start_x", 0), p_right.get("start_y", 0), p_right.get("box_w", 0), p_right.get("box_h", 0), confirm_frames=confirm_frames, direction=1
            )
            self.discard_tracker_up = DiscardTracker(
                p_top.get("start_x", 0), p_top.get("start_y", 0), p_top.get("box_w", 0), p_top.get("box_h", 0), confirm_frames=confirm_frames, direction=2
            )
            self.discard_tracker_left = DiscardTracker(
                p_left.get("start_x", 0), p_left.get("start_y", 0), p_left.get("box_w", 0), p_left.get("box_h", 0), confirm_frames=confirm_frames, direction=3
            )
            self.discard_tracker._config_state = desired_state
        
        self.discard_tracker_enabled = True

    def mask_frame(self, frame):

        if frame is None:
            return None
        
        if self.discard_tracker is not None:
            for box in self.discard_tracker.masked_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (148, 88, 59), -1)

        if self.discard_tracker_right is not None:
            for box in self.discard_tracker_right.masked_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (148, 88, 59), -1)

        if self.discard_tracker_up is not None:
            for box in self.discard_tracker_up.masked_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (148, 88, 59), -1)
        
        if self.discard_tracker_left is not None:
            for box in self.discard_tracker_left.masked_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (148, 88, 59), -1)

        return frame


    
    # Debug Helpers

    def draw_hand_roi(self, frame, roi):
        if frame is None or roi is None:
            return frame
        
        x1, y1, x2, y2 = map(int, roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return frame
        