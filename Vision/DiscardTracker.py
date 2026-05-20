import cv2
import numpy as np

class DiscardTracker:
    def __init__(self, pond_start_x, pond_start_y, tile_dx, tile_dy, confirm_frames=3, direction=0):
        # 1. Grid Definition
        self.start_x = pond_start_x
        self.start_y = pond_start_y
        self.tile_dx = tile_dx
        self.tile_dy = tile_dy
        self.direction = direction
        self._set_deltas()

        #Set offsets 
        self.dx = 0
        self.dy = 0
        self._set_offsets()
        
        # 2. Engine Memory
        self.discard_memory = []
        
        # 3. State Tracking for the newest tile
        self.confirm_frames = confirm_frames
        self.current_tile_seen_count = 0
        self.roi_index = 0
        self.side = -1
        self.masked_boxes = []
        
        # Calculate the very first ROI bounding box [x1, y1, x2, y2]
        self.active_roi = self._calculate_next_roi()

    def _set_deltas(self):
        if self.direction == 0:  
            return
        elif self.direction == 1:
            self.tile_dy *= -1
        elif self.direction == 2: 
            self.tile_dx *= -1
            self.tile_dy *= -1
        elif self.direction == 3:
            self.tile_dx *= -1

    def _set_offsets(self):
        if self.direction % 2 == 0:
            self.dx = 0
            self.dy = 0
        elif self.direction == 1:
            self.dx = -0.04 * self.tile_dx
            self.dy = 0
        elif self.direction == 3:
            self.dx = 0.04 * self.tile_dx
            self.dy = 0

    def _calculate_next_roi(self, side=False):
        if self.direction % 2 == 0:
            row = self.roi_index // 6
            col = self.roi_index % 6
            if row > 3:
                row = 3
                col += 6
        else:
            col = self.roi_index // 6
            row = self.roi_index % 6
            if col > 3:
                col = 3
                row += 6

        if self.direction % 2 == 0:
            if side or row == self.side:
                self.side = row
                x1 = self.start_x + (col - 1) * self.tile_dx + self.tile_dy
            else:
                x1 = self.start_x + col * self.tile_dx
            y1 = self.start_y + row * self.tile_dy
        else:
            if side or col == self.side:
                self.side = col
                y1 = self.start_y + (row - 1) * self.tile_dy - (self.tile_dx * 0.6)
            else:
                y1 = self.start_y + row * self.tile_dy
            x1 = self.start_x + col * self.tile_dx + (row * self.dx)
        x2 = x1 + self.tile_dx
        y2 = y1 + self.tile_dy

        return [x1, y1, x2, y2]

    def _advance_roi(self, side=False):
        self.roi_index += 1
        self.active_roi = self._calculate_next_roi(side)

    def draw_active_roi(self, frame):
        if frame is None:
            return frame
        x1, y1, x2, y2 = map(int, self.active_roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame
    
    def draw_discard_region(self, frame):
        if frame is None:
            return frame
        
        
        x1 = self.start_x
        y1 = self.start_y
        if self.direction % 2 == 0: # Down
            x2 = self.start_x + 6 * self.tile_dx
            y2 = self.start_y + 3 * self.tile_dy
        else:
            x2 = self.start_x + 3 * self.tile_dx
            y2 = self.start_y + 6 * self.tile_dy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return frame
    
    def process_frame(self, frame, boxes, yolo_names):

        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            class_id = int(box.cls[0].item())
            class_name = yolo_names.get(class_id)
            if class_name:
                self.current_tile_seen_count += 1

                if self.current_tile_seen_count >= self.confirm_frames:
                    self.discard_memory.append(class_name)
                    box_coords = box.xyxy[0].cpu().numpy() 
                    self.masked_boxes.append(box_coords)
                    if self.direction % 2 == 0:
                        if abs(box_coords[3] - box_coords[1]) + 10 < abs(self.tile_dy):
                            self._advance_roi(side=True)
                        else:   
                            self._advance_roi()
                    else:
                        if abs(box_coords[2] - box_coords[0])  + 10 < abs(self.tile_dx):
                            print("Right side true")
                            self._advance_roi(side=True)
                        else:
                            self._advance_roi()
                    self.current_tile_seen_count = 0
                    
        else:
            self.current_tile_seen_count = 0

        if frame is None:
            return self.discard_memory
        
        return self.discard_memory
