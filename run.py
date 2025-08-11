import cv2
from src.NumPlateVision.NumPlateReader import NumPlateReader

npr = NumPlateReader()

cap = cv2.VideoCapture("src/car_cropped.mov")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = npr.track_num_plate(frame)
    npr.read_extracted_num_plates(frame, frame_count, results)

    # Show video
    cv2.imshow("YOLO Plate Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
