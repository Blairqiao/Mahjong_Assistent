import time
import os
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import pyqtSignal, QThread, pyqtSlot

from Vision.VisionEngine import VisionEngine
from Logic.GameState import GameState
from Logic.LogicCalculator import LogicCalculator
from UI.UIOverlay import AssistantWindow

class WorkerThread(QThread):
    update_ui_signal = pyqtSignal(dict)

    def __init__(self, initial_settings, model_path):
        super().__init__()
        self.settings = initial_settings
        self.model_path = model_path
        self._is_running = True

    @pyqtSlot(dict)
    def update_settings(self, new_settings):
        self.settings = new_settings

    def run(self):
        state = GameState()
        calc = LogicCalculator()
        vision = VisionEngine(self.model_path)

        print("Starting Main Logic Loop...")
        while self._is_running:
            loop_start = time.perf_counter()
            current_settings = self.settings

            app_name = current_settings.get("app_name", "")
            crop_x = current_settings.get("crop_x", 0)
            crop_y = current_settings.get("crop_y", 0)
            crop_w = current_settings.get("crop_w", 0)
            crop_h = current_settings.get("crop_h", 0)
            hand_roi = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
            vision.update_discard_tracker(current_settings)
            manual_hand_enabled = current_settings.get("manual_hand_enabled", False)
            manual_hand_string = current_settings.get("manual_hand_string", "")

            if manual_hand_enabled:
                try:
                    state.hand_array = GameState.build_hand_array_from_string(manual_hand_string)
                    hand_source_status = "Manual hand active"
                except ValueError as error:
                    self.update_ui_signal.emit({
                        "status": f"Invalid manual hand: {error}",
                        "hand_data": [],
                        "discards": [],
                        "shanten": "--"
                    })
                    time.sleep(0.5)
                    continue
            else:
                hand_source_status = None
            
            debug_frame = None
            if not manual_hand_enabled:
                window_id = vision.get_window_id(app_name)
                if not window_id:
                    self.update_ui_signal.emit({
                        "status": f"Application '{app_name}' not found.",
                        "hand_data": [],
                        "discards": [],
                        "shanten": "--"
                    })
                    time.sleep(1.0)
                    continue
                    
                frame = vision._capture_window_image(window_id)
                if frame is None:
                    self.update_ui_signal.emit({"status": f"Failed to crop frame from '{app_name}'.", "debug_frame": None})
                    time.sleep(0.5)
                    continue
                
                vision.update_discard_tracker(current_settings)
                frame = vision.mask_frame(frame)
                boxes, debug_frame, discard_tiles = vision.predict(frame)
                # boxes, _ = vision.predict_hand(frame, hand_roi)
                debug_frame = vision.draw_hand_roi(debug_frame, hand_roi)
                
                state.update_from_vision(boxes, vision.model.names, hand_roi, discard_tiles)

            
            # Calculate exactly how many tiles the vision engine sees in our hand
            tile_count = sum(state.hand_array)
            
            # Map the current hand regardless of turn condition so we can see what the model sees
            hand_data = []
            for index, count in enumerate(state.hand_array):
                for _ in range(count):
                    hand_data.append({'filename': GameState.TILE_INDEX_TO_ASSET.get(index, ""), 'ukeire': 0})
                    
            if tile_count == 14:
                current_shanten = calc.calculate_shanten(state.hand_array)
                min_shanten_result, best_discards = calc.calculate_best_discards(state.hand_array, state.discarded_tiles)
                
                ukeire_map = dict(best_discards)
                
                hand_data = []
                for index, count in enumerate(state.hand_array):
                    for _ in range(count):
                        ukeire = ukeire_map.get(index, 0)
                        hand_data.append({'filename': GameState.TILE_INDEX_TO_ASSET.get(index, ""), 'ukeire': ukeire})
                
                discard_tiles = [
                    {
                        "filename": GameState.TILE_INDEX_TO_ASSET.get(tile_index, ""),
                        "ukeire": ukeire_count,
                    }
                    for tile_index, ukeire_count in best_discards[:3]
                ]
                
                self.update_ui_signal.emit({
                    "status": hand_source_status or "Locked (Turn active)",
                    "shanten": current_shanten,
                    "hand_data": hand_data,
                    "discards": discard_tiles,
                    "debug_frame": debug_frame
                })
            else:
                # We don't have 14 tiles, meaning it's someone else's turn or we're mid-animation
                if tile_count == 13:
                    current_shanten = calc.calculate_shanten(state.hand_array)
                else:
                    current_shanten = "--"
                self.update_ui_signal.emit({
                    "status": hand_source_status or f"Locked (Waiting for turn: {tile_count} tiles)",
                    "shanten": current_shanten,
                    "hand_data": hand_data,
                    "discards": [],
                    "debug_frame": debug_frame
                })
                
            # Cap at roughly ~5 FPS (0.2s per frame) instead of hard blocking for 0.5s which impacts snappiness
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.01, 0.2 - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        self._is_running = False
        self.wait()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = AssistantWindow()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "runs", "detect", "mahjong_nano_v4", "weights", "best.pt")
    
    worker = WorkerThread(window.settings, model_path)
    worker.update_ui_signal.connect(window.update_data)
    window.settings_changed_signal.connect(worker.update_settings)
    
    worker.start()
    
    window.show()
    
    sys.exit(app.exec())
    
    emitter = SignalEmitter()
    emitter.update_ui_signal.connect(window.update_data)
    
    # Hook the settings changed signal from UI back into the main loop via a custom attribute
    window.settings_changed_signal.connect(lambda settings: getattr(emitter, 'settings_changed', lambda s: None)(settings))
    
    backend_thread = threading.Thread(target=main_loop, args=(emitter, window.settings), daemon=True)
    backend_thread.start()
    
    sys.exit(app.exec())
