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
            original_image (Image/np.ndarray): Image of the vehicle of the number plate to be detected

        Returns:
            dict[str]: Dictionary of the bounding boxes scores and labels predicted
        """

        # Process image to suitable format for model (pytorch tensor)
        inputs = self.feature_extractor(original_image, return_tensors="pt")
        
        # Pass the inputs through the model to get outputs
        # Unpack the key, value pairs to get the values
        outputs = self.model(**inputs)
        
        # Compute the original image size and prepare for post processing
        # tf wants (height, width) input
        image_type = self.get_image_type(original_image)
        img_size = None

        if image_type == 0:
            # PIL image format is (width, height) so need to reverse
            img_size = torch.tensor([tuple(reversed(original_image.size))])
        elif image_type == 1:
            img_size = torch.tensor([tuple(original_image.shape[:2])])
        else:
            raise Exception("Image must be Numpy array or PIL Image")
        
        # Post process the raw model outputs and convert to usable results 
        processed_outputs = self.feature_extractor.post_process(outputs, img_size)

        return processed_outputs[0]

    def get_bounding_box(self, output_dict, threshold=0.5) -> tuple[float, float, float, float]:
        """
        Get the physical bounding box of the number plate detected
        
        Parameters:
            output_dict (str): Prediction directory
            threshold (float): Threshold of the confidence needed to label bounding box

        Returns: 
            tuple[float, float, float, float]: Tuple of x, y, w, h of the bounding box
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
            image (Image/np.ndarray): Original image to be extracted from
            xmin, ymin, xmax, ymax (float): Coordinates of the bounding box with (xmin, ymin) being the bottom left of bounding box

        Returns:
            Image: Number plate cropped from the original image
        """

        image_type = self.get_image_type(image)

        if image_type == 0:
            return np.array(image.crop((xmin, ymin, xmax, ymax)))
        elif image_type == 1:
            return image[int(ymin):int(ymax), int(xmin):int(xmax)]
        else:
            raise Exception("Image must be Numpy array or PIL Image")
        
    def get_image_type(self, image):
        """
        Get the image type from PIL Image or numpy array (OpenCV)

        Parameters:
            image (Image/np.ndarray): Image of type to be found
        
            Returns: 
            int: Integer code of the type of image
        """

        if isinstance(image, Image.Image):
            return 0
        elif isinstance(image, np.ndarray):
            return 1
        else:
            return -1 
        
