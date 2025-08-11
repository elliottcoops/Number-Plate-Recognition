from ultralytics import YOLO
import cv2 
from src.NumPlateVision.FrameDraw import FrameDraw
from src.NumPlateVision.CharacterExtractor import CharacterExtractor

class NumPlateReader:
    def __init__(self):
        self.model = YOLO("src/model/my_custom_model.pt")
        self.frame_draw = FrameDraw()
        self.character_extractor = CharacterExtractor()

    def track_num_plate(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)
        return results
    
    def read_extracted_num_plates(self, frame, frame_count, results):
        for box in results[0].boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
            plate_crop = frame[ymin:ymax, xmin:xmax]
            track_id = int(box.id) if hasattr(box, 'id') and box.id is not None else None
            plate_text = self.character_extractor.read_num_plate(track_id, frame_count, plate_crop)
            self.frame_draw.draw_num_plate_reading(xmin, ymin, xmax, ymax, plate_text, track_id, frame)

            
