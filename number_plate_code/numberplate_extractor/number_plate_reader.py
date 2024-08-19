
from numberplate_extractor.plate_extractor import PlateExtractor
from numberplate_extractor.character_extractor import CharacterExtraction

"""
Read the number plate of vehicle in an image
"""
class NumberPlateReader:

    def __init__(self):
        """
        Load and initialise the plate and character extractor objects
        """

        self.plate_extractor = PlateExtractor()
        self.character_extractor = CharacterExtraction()

    def get_number_plate(self, image):
        """
        Get the number plate from a vehicle in image
        
        Parameters:
        image (np.ndarray): Image to extract plate from
        """

        # Make prediction from the original image
        numberplate_prediction = self.plate_extractor.make_prediction(image)

        # Get the bounding box of the detected plate

        xmin, ymin, xmax, ymax = self.plate_extractor.get_bounding_box(numberplate_prediction)

        # Get the image as an np array
        number_plate = self.plate_extractor.get_extracted_numberplate_as_np(image, xmin, ymin, xmax, ymax)

        return number_plate

    def read_number_plate(self, number_plate, each_stage=False):
        """
        Extract and read the individual characters from the number plate
        
        Parameters:
        number_plate (np.ndarray): Number plate for characters to be extracted
        each_stage (boolean): Flag to see each stage (original, numberplate, ...)
        """
        
        # Segment the characters in the extracted number plate
        seg_number_plate = self.character_extractor.segment_characters(number_plate)

        # Find the contours (characters) in the number plate
        cntrs = self.character_extractor.find_contours(seg_number_plate)
        
        # Extract each individual character from the number plate
        extracted_chars, cntr_plate = self.character_extractor.extract_characters(cntrs, seg_number_plate, number_plate)

        # Read each character and place into string
        plate_reading = self.character_extractor.read_number_plate(extracted_chars)
        
        if not each_stage:
            return plate_reading

        return plate_reading, seg_number_plate, cntr_plate