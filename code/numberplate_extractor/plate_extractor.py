import torch
import numpy as np
import cv2 as cv
from PIL import Image
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from keras.models import load_model


class PlateExtractor:
    def __init__(self, license_model_path = 'nickmuchi/yolos-small-finetuned-license-plate-detection', character_model_path = "../model/my_model.h5", labels = None):
        self.feature_extractor = YolosFeatureExtractor.from_pretrained(license_model_path)
        self.model = YolosForObjectDetection.from_pretrained(license_model_path)

        self.character_model = load_model(character_model_path)
        if labels is None:
            self.labels = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]

    def make_prediction(self, img):
        inputs = self.feature_extractor(img, return_tensors="pt")
        outputs = self.model(**inputs)
        img_size = torch.tensor([tuple(reversed(img.size))])
        processed_outputs = self.feature_extractor.post_process(outputs, img_size)
        return processed_outputs[0]

    def get_bounding_box(self, output_dict, threshold=0.5):
        # (xmin, ymin) is bottom left of the bounding box
        id2label=self.model.config.id2label
        keep = output_dict["scores"] > threshold
        boxes = output_dict["boxes"][keep].tolist()
        labels = output_dict["labels"][keep].tolist()
        
        if id2label is not None:
            labels = [id2label[x] for x in labels]
            
        for (xmin, ymin, xmax, ymax), label in zip(boxes, labels):
            if label == 'license-plates':
                return xmin, ymin, xmax, ymax
            
    def get_extracted_numberplate(self, image, xmin, ymin, xmax, ymax):
        numberplate = image.crop((xmin, ymin, xmax, ymax))
        return numberplate

    def get_extracted_numberplate_as_np(self, image):
        numberplate = self.get_extracted_numberplate(image)
        return np.array(numberplate)
    
    def extract_chars(self, width, height, contours, output_image):
        str_out = ""

        # Define realistic bounds for character size
        min_width = width / 20   # Minimum width of a character
        max_width = width / 4    # Maximum width of a character
        min_height = height / 2  # Minimum height of a character (almost full height)
        max_height = height      # Maximum height of a character (close to full height)

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if min_width < w < max_width and min_height < h < max_height:
                char = output_image[y:y+h, x:x+w]
                char = cv.cvtColor(char, cv.COLOR_BGR2GRAY)
                char = cv.merge([char, char, char])
                char = cv.resize(char, (64, 64))    
                char = np.array(char, dtype=np.float32)
                char = np.expand_dims(char, axis=0)
                char = char / 255
                y_pred = self.character_model.predict(char, verbose=0)
                str_out = str_out + self.labels[np.argmax(y_pred)]

        return str_out

