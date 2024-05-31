from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
from model.model_utils import predict_and_save  # Correct import

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure the upload and processed folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def generate_unique_filename(filename):
    ext = os.path.splitext(filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    return unique_filename

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        original_filename = file.filename
        unique_filename = generate_unique_filename(original_filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], unique_filename)
        
        result = predict_and_save(filepath, processed_filepath)

        return jsonify({
            'message': 'File successfully uploaded and processed',
            'processed_image_url': f'/processed/{unique_filename}',
            'result': result
        }), 201

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
