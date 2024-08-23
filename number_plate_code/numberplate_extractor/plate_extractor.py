import torch
import numpy as np
from PIL import Image
from transformers import YolosFeatureExtractor, YolosForObjectDetection

"""
Extract number plates from images with vehicles in them
"""
class PlateExtractor:

    def __init__(self, license_model_path = 'nickmuchi/yolos-small-finetuned-license-plate-detection'):
        """
        Load and initialise license plate detection model

        Parameters:
        license_model_path (str): Path to the license detection model in hugging face 
        """

        self.feature_extractor = YolosFeatureExtractor.from_pretrained(license_model_path)
        self.model = YolosForObjectDetection.from_pretrained(license_model_path)

    def make_prediction(self, img) -> list[dict[str]]:
        """
        Make prediction based on input image of vehicle

        Parameters:
        img (Image): PIL Image of the vehicle of the license plate to be detected
        """

        # Set input and tensors
        inputs = self.feature_extractor(img, return_tensors="pt")
        outputs = self.model(**inputs)
        img_size = torch.tensor([tuple(reversed(img.size))])
        processed_outputs = self.feature_extractor.post_process(outputs, img_size)

        return processed_outputs[0]

    def get_bounding_box(self, output_dict, threshold=0.5) -> tuple[int, int, int, int]:
        """
        Get the physical bounding box of the license plate detected
        
        Parameters:
        output_dict (str): Prediction directory
        threshold (float): Threshold of the confidence needed to label bounding box
        """

        # Get the id2label for the output layer
        id2label=self.model.config.id2label

        # Keep the plates that are above the threshold value (i.e. above a confidence level 'threshold')
        keep = output_dict["scores"] > threshold

        # Get list of boxes and lables to keep
        boxes = output_dict["boxes"][keep].tolist()
        labels = output_dict["labels"][keep].tolist()

        # Raise exception if no number plate has been detected
        if len(boxes) == 0:
            raise Exception("No number plate detected")
        
        if id2label is not None:
            labels = [id2label[x] for x in labels]
        
        # Return the number plate bounding box (first one)
        for (xmin, ymin, xmax, ymax), label in zip(boxes, labels):
            if label == 'license-plates':
                return xmin, ymin, xmax, ymax
            
    def get_extracted_numberplate(self, image, xmin, ymin, xmax, ymax) -> Image:
        """
        Extract the numberplate from the original image as PIL Image
        
        Parameters: 

        image (Image): Original image to be extracted from
        xmin, ymin, xmax, ymax (float): Coordinates of the bounding box with (xmin, ymin) being the bottom left of bounding box
        """

        numberplate = image.crop((xmin, ymin, xmax, ymax))

        return numberplate

    def get_extracted_numberplate_as_np(self, image, xmin, ymin, xmax, ymax) -> np.ndarray:
        """
        Extract the numberplate from the original image as np array
        
        Parameters: 

        image (Image): Original image to be extracted from
        xmin, ymin, xmax, ymax (float): Coordinates of the bounding box with (xmin, ymin) being the bottom left of bounding box
        """

        numberplate = self.get_extracted_numberplate(image, xmin, ymin, xmax, ymax)

        return np.array(numberplate)