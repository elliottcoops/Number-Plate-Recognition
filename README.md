# Number plate recognition research project

## About

This repository contains a number plate reader, using a YOLO model for license plate extraction and a CNN trained from scratch on an OCR dataset for character recognition.

![example_recognition](docs/example_reading.png)

## Detecting a number plate

The number plate of a car can be obtained by using an object detection model. State of the art currently is YOLO (You Only Live Once) which is a deep CNN. 

The model for number plate extraction can be found [here](https://huggingface.co/nickmuchi/yolos-small-finetuned-license-plate-detection) on the hugging face and imported using the transformers module.

### Extraction

The model finds a bounding box of what it thinks is a number plate. With a set confidence level, we can obtain the bounding box predicted by the model, and extract the number plate from the original image.

![Original](docs/original.png) ![Extracted](docs/extracted.png)

## Reading a number plate

A word can be read by extracting each character from the number plate, passing it into the character model and then stringing together a word.

### Filtering characters

One similarity among all number plates is that the letters are black. Therefore, after applying image processing techniques such as bilateral filtering, gamma correction, inversion, and Otsu thresholding, we can create a binary image with the letters as the foreground and everything else as the background.

![Original](docs/processed.png)

### Extracting characters and predicting words

To extract a character, we can use the contours method from OpenCV, which identifies foreground objects within an image. This method creates a bounding box around each character, allowing us to extract that portion of the image.

Once we have the characters, we can feed each one into the model, obtain the predictions, and concatenate them into a string.

## Character recognition

### Training

Neural network is trained on the standard OCR dataset which contains 50k images of characters.

In addition, data is augmented 5 times per image from a mix of rotation, translation and zooming. This attains a total train set of size 100k

### Constants

Constants used during training:

- Loss: Categorical crossentropy
- Epochs: 10
- Optimiser: Adam

### Model evaluation

After training, the test set attains an accuracy of 98.7%

Looking at the loss and accuracy per epoch we see that there are no signs of overfitting:

![eval](docs/model_eval.png)

In addition, looking at a few test examples: 

![example_test](docs/character_example.png)