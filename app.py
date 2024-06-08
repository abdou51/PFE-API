from flask import Flask, request, jsonify, url_for
from ultralytics import YOLO
import os
import uuid
import cv2

app = Flask(__name__)

# Load the pretrained YOLO model
model = YOLO('best.pt')

# Directory for saving videos, ensure it exists
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

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # Save the uploaded file temporarily
        temp_filename = 'input_video.mp4'
        file.save(temp_filename)

        # Open the video file
        cap = cv2.VideoCapture(temp_filename)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
        unique_filename = str(uuid.uuid4()) + '.mp4'
        output_filepath = os.path.join(STATIC_DIR, unique_filename)
        out = cv2.VideoWriter(output_filepath, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLO inference on the frame
                results = model(frame, conf=0.5)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Write the frame into the output video file
                out.write(annotated_frame)

                # Display the annotated frame (optional)
                # cv2.imshow("YOLOv8 Inference", annotated_frame)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release everything when the job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Generate the URL for the output video
        output_url = url_for('static', filename=unique_filename, _external=True)

        # Return the URL of the processed video
        return jsonify({'url': output_url})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
