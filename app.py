from flask import Flask, request, jsonify, url_for
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)

# Load the pretrained YOLO model
model = YOLO('best.pt')

# Directory for saving images, ensure it exists
STATIC_DIR = 'static'
os.makedirs(STATIC_DIR, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # Save the uploaded file temporarily
        temp_filename = 'input_image.jpg'
        file.save(temp_filename)

        # Run inference
        output = model.predict(temp_filename, save=True, imgsz=320, conf=0.30)

        # Retrieve the directory where the output is saved
        output_dir = output[0].save_dir

        # Assuming the output image has a predictable naming convention, it's renamed and moved
        input_output_filename = os.path.join(output_dir, 'input_image.jpg')
        unique_filename = str(uuid.uuid4()) + '.jpg'
        final_output_filename = os.path.join(STATIC_DIR, unique_filename)

        # Move the file to the static directory
        os.rename(input_output_filename, final_output_filename)

        # Generate the URL for the output image
        output_url = url_for('static', filename=unique_filename, _external=True)

        # Return the URL of the processed image
        return jsonify({'url': output_url})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
