import cv2 as cv
import numpy as np


class ProcessingPipeline:

    def apply_edge_sharpening(self, image):
        """
        Apply edge sharpening using the laplacian operator

        Parameters:
        image (np.ndarray): Numpy array which holds the image to be processed
        """

        # Gaussian blur the image
        gaussian_blurred_image = cv.GaussianBlur(image, (9, 9), 3)
        # Apply laplacian operator
        laplacian_image = cv.Laplacian(gaussian_blurred_image, cv.CV_64F)
        # Convert back to uint8 type
        laplacian_image = np.uint8(np.absolute(laplacian_image))
        # Add the weighted image onto the original image
        sharpened_image = cv.addWeighted(image, 1.5, laplacian_image, -1, 0)

        return sharpened_image
    
    def apply_gamma_correction(self, image):
        """
        Apply gamma correction to reduce the contrast of image

        Parameters:
        image (np.ndarray): Numpy array which holds the image to be processed
        """

        # Initialise gamma value
        gamma = 1.45 
        # Apply the gamma correction
        gamma_corrected_image = np.power(image / 255.0, gamma) * 255.0
        # Convert back to uint8 type
        image = np.uint8(gamma_corrected_image)

        return image

    def invert_image(self, image):
        """
        Invert the image
        
        Parameters:
        image (np.ndarray): Numpy array which holds the image to be processed
        """

        return 255 - image


    def apply_bilateral_filtering(self, image):
        """
        Apply bilateral filtering 
        
        Parameters:
        image (np.ndarray): Numpy array which holds the image to be filtered
        """

        return cv.bilateralFilter(image, d=21, sigmaColor=75, sigmaSpace=75)

    def extract_original_image(self, image, mask):
        """
        Extract the original image using the segmented binary mask

        Parameters:
        image (np.ndarray): Numpy array which holds the image to have the mask placed over
        mask (np.ndarray): Numpy array which holds the (binary) mask for extraction
        """

        return cv.bitwise_and(image, image, mask=mask)

    def apply_pipeline(self, image):
        """
        Apply all stages of pipeline to the image

        Parameters:
        image (np.ndarray): Image to be processed
        """

        # Convert to grey colour space
        grey_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        # Invert the image
        inverted_image = self.invert_image(grey_image)
        # Apply gamma correction
        gamma_corrected_image = self.apply_gamma_correction(inverted_image)
        # Shapren edges
        sharpened_image = self.apply_edge_sharpening(gamma_corrected_image)
        # Blur the image again but bear in mind the edges
        bilateral_filtered_image = self.apply_bilateral_filtering(sharpened_image)

        _, thresh = cv.threshold(bilateral_filtered_image,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

        # Perform morphological operations
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 1)

        return 255-opening
    
    def find_sorted_contours(self, processed_numberplate):
        """
        Find contous (chars in the numberplate) and sort the in order of x coordinate
        
        Parameters:
        processed_numberplate (np.ndarray): Binary representation of extracted numberplate
        """

        contours, _ = cv.findContours(processed_numberplate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv.boundingRect(c)[0])
        return contours
    
    def crop_numberplate_from_original(self, image, xmin, ymin, xmax, ymax):
        """
        Crop the numberplate from the original image
        
        Parameters:
        image (np.ndarray): Original image to crop numberplate from
        xmin, ymin, xmax, ymax (int): Coordinates of bounding box found
        """
        
        numberplate = image.crop((xmin, ymin, xmax, ymax))
        return np.array(numberplate)