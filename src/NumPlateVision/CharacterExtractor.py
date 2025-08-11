import cv2
import easyocr

class CharacterExtractor:
    def __init__(self):
        self.plate_cache = {}
        self.reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if no GPU
        self.OCR_INTERVAL = 15 
    
    def read_characters(self, plate_crop):
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        ocr_result = self.reader.readtext(gray, detail=0)
        plate_text = "".join(ocr_result).strip().upper() if ocr_result else "N/A"
        return plate_text


    def read_num_plate(self, track_id, frame_count, plate_crop):
        plate_text = "N/A"
        if track_id is not None:
            # Only run OCR if not cached or enough frames have passed
            if track_id not in self.plate_cache or (frame_count - self.plate_cache[track_id]["last_ocr_frame"] >= self.OCR_INTERVAL):
                if plate_crop.size > 0:
                    plate_text = self.read_characters(plate_crop)
                    self.plate_cache[track_id] = {"text": plate_text, "last_ocr_frame": frame_count}
            else:
                plate_text = self.plate_cache[track_id]["text"]
        return plate_text