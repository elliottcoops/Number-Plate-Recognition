import cv2

class FrameDraw:

    def __init__(self):
        self.font_scale = 1.0
        self.font_thickness = 2

    def write_label(self, track_id, plate_text):
        label_text = None
        if track_id is not None:
            label_text = f"ID: {track_id} | Plate: {plate_text}"
        else:
            label_text = f"Plate: {plate_text}"
        return label_text
    
    def draw_label(self, frame, xmin, ymin, label_text):
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                              self.font_scale, self.font_thickness)
        text_x, text_y = xmin, max(ymin - 10, text_height + 10)

        cv2.rectangle(frame,
                    (text_x, text_y - text_height - 4),
                    (text_x + text_width, text_y + baseline),
                    (255, 255, 255), thickness=cv2.FILLED)

        cv2.putText(frame, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0), self.font_thickness)

    def draw_num_plate_reading(self, xmin, ymin, xmax, ymax, plate_text, track_id, frame):

        # Draw bounding box around plate
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Get the label text
        label_text = self.write_label(track_id, plate_text)

        # Draw the label text onto screen
        self.draw_label(frame, xmin, ymin, label_text)

        
        
