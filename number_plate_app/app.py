from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from config import Config
import sys
import os

# Append number plate code directory to system path so we can access number plate classes 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../number_plate_code')))

from image_handler import ImageHandler

app = Flask(__name__)

# Load config data from class
app.config.from_object(Config)

# Object which handles uploading images and reading of number plate code
image_handler = ImageHandler(app.config["UPLOAD_FOLDER"])

@app.route('/', methods=['GET', 'POST'])
def upload_file(): 
    """
    Accept both GET and POST methods
    
    Handle uploading an image file and reading the number plate of it
     
    Successful uploads yield a redirect to the image page with the read plate and each stage of the pipeline 
    """
    
    # Check if the request is a POST one
    if request.method == 'POST':
        # If the file is not in request files then stay at the same page
        if 'file' not in request.files:
            return redirect(request.url)
        
        # Get the physical file name
        file = request.files['file']

        # If the file name is empty then stay on the same page
        if file.filename == '':
            return redirect(request.url)
        
        # If the file name is not empty and the correct extension then save to upload folder
        if file and image_handler.allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        plate_path, seg_path, cntr_path, plate_text = image_handler.read_number_plate(filename)

        # Redirect to the uploaded_file app route with each stage of the pipeline and the extracted plate
        return redirect(url_for('uploaded_file', original_path=filename, plate_path=plate_path, 
                                    seg_path=seg_path, cntr_path=cntr_path, plate_text=plate_text))

    # Remain on the same page     
    return render_template('upload.html')

@app.errorhandler(Exception)
def detection_error(e) -> str:
    return render_template('500.html')

@app.route('/uploads/<original_path>/<plate_path>/<seg_path>/<cntr_path>/<plate_text>')
def uploaded_file(original_path, plate_path, seg_path, cntr_path, plate_text):
    """
    Get HTMl template and path of plate images and plate text

    Paramaters:
    *_path (str): Path of plate image files server side
    plate_text (str): Characters extracted from number plate
    """

    return render_template('display_image.html', original_path=original_path, plate_path=plate_path, seg_path=seg_path, 
                           cntr_path=cntr_path, plate_text=plate_text)

@app.route('/uploads/<filename>/view')
def view_file(filename):
    """
    Get file server side from upload folder
    
    Parameters:
    filename (str): Name of file in upload folder
    """

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
