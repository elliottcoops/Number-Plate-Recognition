import cv2 as cv
import numpy as np
from keras.models import load_model

"""
Extract and read characters from a number plate
"""
class CharacterExtraction:

    def __init__(self, character_model_path = "../model/my_model.h5", labels = None):
        """
        Load character recognition model and define labels for model output

        Parameters:
        character_model_path (str): Path to character model stored locally
        labels (list): List of outputs for the character detection model    
        """
         
        self.character_model = load_model(character_model_path)

        if labels is None:
            self.labels = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]

    def get_x_coordinate(self, contour):
        """
        Get the x coordinate of a detected contour
        
        Parameters:
        contour (tuple): Coordinates, width and height of boundinng box around detected character
        """

        x, _, _, _ = cv.boundingRect(contour)
        return x

    def preprocess_char(self, char):
        """
        Preprocess the character image to be passed into the model

        Parameters:
        char (np.ndarray): Extracted character from number plate
        """

        # Create 3 channels
        char = cv.merge([char, char, char])
        
        # Resize to 64x64 pixels
        char = cv.resize(char, (64, 64))    
        
        # Convert to float32 type
        char = np.array(char, dtype=np.float32)

        # Add dimension to first axis
        char = np.expand_dims(char, axis=0)

        # Normalise image
        char = char / 255

        return char


    def find_contours(self, segmented_chars):
        """
        Find contours (characters) in binary image
        
        Parameters:
        segmented_chars (np.ndarray): Binary image of segmented characters
        """

        # Find contours in image
        cntrs, _ = cv.findContours(segmented_chars.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Sort the detected contours by their x-coordinate (so they appear in order)
        cntrs = sorted(cntrs, key=self.get_x_coordinate)

        return cntrs
    
    def get_dimension_estimation(self, number_plate):
        """
        Get the estimation dimension of a character on a number plate
        
        Parameters:
        number_plate (np.ndarray): Image of extracted number plate
        """

        w, h = number_plate.shape[0], number_plate.shape[1]

        dimensions = [w/5,
                        w/2,
                        h/10,
                        2*h/5]
        
        return dimensions
    
    def extract_characters(self, cntrs, segmented_chars):
        """
        Extract each individual character on the number plate
        
        Parameters:
        cntrs (list(tuple)): Detected contours on the number plate
        segmented_chars (np.ndarray): Image of segmented characters
        """

        # Get estimations of character contours sizes of cropped license plates
        lower_width, upper_width, lower_height, upper_height = self.get_dimension_estimation(segmented_chars)
       
        extracted_characters = []

        for cntr in cntrs:
            # Get the bounding rectangle of the contour around the character
            intX, intY, intWidth, intHeight = cv.boundingRect(cntr)
            
            # Check the contour falls within the boundaries of estimations
            if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
                # Extract the character from the number plate
                char = segmented_chars[intY:intY+intHeight, intX:intX+intWidth]
                
                extracted_characters.append(char)
    
        return extracted_characters
    
    def read_number_plate(self, extracted_chars):
        """
        Read each detected character on the number plate and string together
        
        Parameters:
        extracted_chars (list(np.ndarray)): Extracted characters from number plate
        """
        
        plate_reading = ""

        for char in extracted_chars:
            # Preprocess extracted character for prediction
            char = self.preprocess_char(char)

            # Predict and append to plate reading
            y_pred = self.character_model.predict(char, verbose=0)
            plate_reading = plate_reading + self.labels[np.argmax(y_pred)]

        return plate_reading
                    
    def segment_characters(self, number_plate):
        """
        Segment characters and remove irrelevent background noise
        
        Parameters:
        number_plate (np.ndarray): Image of number plate
        """

        # Resize to researched dimensions
        img_lp = cv.resize(number_plate, (333, 75))

        # Convert to greyscale
        img_gray_lp = cv.cvtColor(img_lp, cv.COLOR_BGR2GRAY)

        # Sharpen image edges using laplacian
        laplacian = cv.Laplacian(img_gray_lp, cv.CV_64F)
        laplacian_8u = cv.convertScaleAbs(laplacian)
        sharpened_image = cv.addWeighted(img_gray_lp, 1.5, laplacian_8u, -0.5, 0)

        # Otsu thresholding to segment characters
        _, img_binary_lp = cv.threshold(sharpened_image, 240, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        # Attempt to remove background noise 
        img_binary_lp = cv.erode(img_binary_lp, (7,7))

        # Make borders white
        img_binary_lp[0:3,:] = 255
        img_binary_lp[:,0:3] = 255
        img_binary_lp[72:75,:] = 255
        img_binary_lp[:,330:333] = 255

        return img_binary_lp
    
    

    