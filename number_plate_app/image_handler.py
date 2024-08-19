import cv2 as cv
import os
from PIL import Image
import cv2 as cv
from number_plate_code.numberplate_extractor.number_plate_reader import NumberPlateReader

class ImageHandler:

    def __init__(self, upload_path):
        """
        Initialise number plate reader and upload path

        Parameters:
        upload_path (str): Path of server side upload folder 
        """
        
        self.number_plate_reader = NumberPlateReader()
        self.upload_path = upload_path

    def write_image(self, file_path, image):
        """
        Write image to upload folder server side
        
        Parameters:
        file_path (str): Image file path
        upload_path (str): Upload dir path
        image (np.ndarray): Image to be uploaded
        """

        output_path = os.path.join(self.upload_path, file_path)
        cv.imwrite(output_path, image)
        return file_path
    
    def write_images(self, images):
        """
        Write images to upload folder server side
        
        Parameters:
        upload_folder (str): Upload dir path
        images (list(np.ndarray)): List of images to be uploaded
        """

        plate_path = self.write_image("number_plate.jpeg", images[0])
        seg_path = self.write_image("segmented_number_plate.jpeg", images[1])
        cntr_path = self.write_image("cntr_plate.jpeg", images[2])

        return plate_path, seg_path, cntr_path
    
    def allowed_file(self, file_path, allowed_extensions):
        """
        Check if file is in the allowed extensions
        
        Parameters:
        file_path (str): Path of file to be checked
        """

        return '.' in file_path and \
            file_path.rsplit('.', 1)[1].lower() in allowed_extensions
    
    def read_number_plate(self, file_path):
        """
        Read the number plate of the given file path
        
        Parameters:
        file_path (str): File path of image to read number plate from
        """

        # Read image using PIL for transformers
        image = Image.open(os.path.join(self.upload_path, file_path))

        # Extract number plate
        number_plate = self.number_plate_reader.get_number_plate(image)

        # Read the number plate and get images from each stage
        plate_text, seg_plate, cntr_plate = self.number_plate_reader.read_number_plate(number_plate, True)

        # Write images from process server side
        images_to_upload = [cv.cvtColor(number_plate, cv.COLOR_BGR2RGB), seg_plate, cv.cvtColor(cntr_plate, cv.COLOR_BGR2RGB)]
        plate_path, seg_path, cntr_path = self.write_images(images_to_upload)

        return plate_path, seg_path, cntr_path, plate_text