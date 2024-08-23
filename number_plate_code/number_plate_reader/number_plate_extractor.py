import torch
import numpy as np
from PIL import Image
from transformers import YolosFeatureExtractor, YolosForObjectDetection

"""
Extract number plates from images with vehicles in them
"""
class PlateExtractor:

    def __init__(self, number_plate_model_path = 'nickmuchi/yolos-small-finetuned-license-plate-detection'):
        """
        Load and initialise number plate detection model

        Parameters:
            number_plate_model_path (str): Path to the number detection model in hugging face 
        """

        self.feature_extractor = YolosFeatureExtractor.from_pretrained(number_plate_model_path)
        self.model = YolosForObjectDetection.from_pretrained(number_plate_model_path)

    def make_prediction(self, original_image) -> dict[str]:
        """
        Get number plate from input image

        Parameters:
            original_image (Image): PIL Image of the vehicle of the number plate to be detected

        Returns:
            dict[str]: Dictionary of the bounding boxes scores and labels predicted
        """

        # Process image to suitable format for model (pytorch tensor)
        inputs = self.feature_extractor(original_image, return_tensors="pt")
        
        # Pass the inputs through the model to get outputs
        # Unpack the key, value pairs to get the values
        outputs = self.model(**inputs)
        
        # Compute the original image size and prepare for post processing
        # tf wants (height, width) input, so we need to reverse PIL image
        img_size = torch.tensor([tuple(reversed(original_image.size))])
        
        # Post process the raw model outputs and convert to usable results 
        processed_outputs = self.feature_extractor.post_process(outputs, img_size)

        return processed_outputs[0]

    def get_bounding_box(self, output_dict, threshold=0.5) -> tuple[int, int, int, int]:
        """
        Get the physical bounding box of the number plate detected
        
        Parameters:
            output_dict (str): Prediction directory
            threshold (float): Threshold of the confidence needed to label bounding box

        Returns: 
            tuple[int, int, int, int]: Tuple of x, y, w, h of the bounding box
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
            
    def get_extracted_number_plate(self, image, xmin, ymin, xmax, ymax) -> Image:
        """
        Extract the number plate from the original image as PIL Image
        
        Parameters: 
            image (Image): Original image to be extracted from
            xmin, ymin, xmax, ymax (float): Coordinates of the bounding box with (xmin, ymin) being the bottom left of bounding box

        Returns:
            Image: Number plate cropped from the original image
        """

        number_plate = image.crop((xmin, ymin, xmax, ymax))

        return number_plate

    def get_extracted_number_plate_as_np(self, image, xmin, ymin, xmax, ymax) -> np.ndarray:
        """
        Extract the number plate from the original image as np array
        
        Parameters: 
            image (Image): Original image to be extracted from
            xmin, ymin, xmax, ymax (float): Coordinates of the bounding box with (xmin, ymin) being the bottom left of bounding box
        
        Returns:
            np.ndarray: Numpy array of number plate cropped from original image 
        """

        number_plate = self.get_extracted_number_plate(image, xmin, ymin, xmax, ymax)
        number_plate_as_np = np.array(number_plate)

        return number_plate_as_np