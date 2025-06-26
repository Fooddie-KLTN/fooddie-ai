from app import app
from flask import make_response, render_template, send_file
from flask import Flask, Response, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import base64
import os
import io
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# Load YOLOv5 model using ultralytics package
try:
    from ultralytics import YOLO
    detection_model = YOLO('detection.pt')
    print("Model loaded successfully with ultralytics")
except Exception as e:
    print(f"Error loading model: {e}")
    detection_model = None

# Load Keras model for classification
try:
    classifier_model = keras.models.load_model('classifier.keras')
    print("Classifier loaded successfully")
except Exception as e:
    print(f"Error loading classifier: {e}")
    classifier_model = None

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Food detection counters (adjust these categories based on your classifier)
food_counts = {}

# Vietnamese food labels
FOOD_LABELS = [
    'Banh beo', 'Banh bot loc', 'Banh can', 'Banh canh', 'Banh chung', 'Banh cuon', 
    'Banh duc', 'Banh gio', 'Banh khot', 'Banh mi', 'Banh pia', 'Banh tet', 
    'Banh trang nuong', 'Banh xeo', 'Bun bo Hue', 'Bun dau mam tom', 'Bun mam', 
    'Bun rieu', 'Bun thit nuong', 'Ca kho to', 'Canh chua', 'Cao lau', 'Chao long', 
    'Com tam', 'Goi cuon', 'Hu tieu', 'Mi quang', 'Nem chua', 'Pho', 'Xoi xeo'
]

def detect_and_classify_food(image):
    """
    Detect food items using YOLOv5 and classify them using Keras model
    """
    global food_counts
    
    if detection_model is None:
        return []
    
    # Convert BGR to RGB for YOLOv5
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run YOLOv5 detection using ultralytics
    results = detection_model(rgb_image)
    
    detection_results = []
    
    # Process results
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                if confidence > 0.5:  # Confidence threshold
                    # Extract ROI for classification
                    roi = image[int(y1):int(y2), int(x1):int(x2)]
                    
                    if roi.size > 0 and classifier_model is not None:
                        # Preprocess ROI for EfficientNet classifier
                        roi_resized = cv2.resize(roi, (224, 224))
                        
                        # Convert BGR to RGB for the classifier
                        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                        
                        # Convert to PIL Image format
                        roi_pil = Image.fromarray(roi_rgb)
                        
                        # Convert back to array and preprocess
                        roi_array = keras_image.img_to_array(roi_pil)
                        roi_processed = preprocess_input(np.copy(roi_array))  # EfficientNet preprocessing
                        roi_batch = np.expand_dims(roi_processed, axis=0)
                        
                        # Classify the detected food
                        classification_result = classifier_model.predict(roi_batch, verbose=0)
                        predicted_class_idx = np.argmax(classification_result)
                        classification_confidence = float(np.max(classification_result))
                        
                        # Get food label name
                        food_label = FOOD_LABELS[predicted_class_idx] if predicted_class_idx < len(FOOD_LABELS) else f"Unknown_{predicted_class_idx}"
                        
                        # Add detection result
                        detection_results.append({
                            'bbox': {
                                'x1': int(x1),
                                'y1': int(y1),
                                'x2': int(x2),
                                'y2': int(y2)
                            },
                            'detection_confidence': float(confidence),
                            'class_id': int(predicted_class_idx),
                            'class_name': food_label,
                            'classification_confidence': classification_confidence
                        })
                        
                        # Update food counts using food label name
                        if food_label in food_counts:
                            food_counts[food_label] += 1
                        else:
                            food_counts[food_label] = 1
    
    return detection_results

@app.route('/')
@app.route('/image', methods=['GET', 'POST'])
def image():
    global food_counts
    
    if request.method == "POST":
        try:
            # Reset counts for new detection
            food_counts = {}
            
            # Check if file is in request
            if 'file' not in request.files:
                return jsonify({"error": "No file part"}), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
            
            if file:
                # Read image directly from memory
                file_bytes = file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return jsonify({"error": "Invalid image file"}), 400
                
                # Process the image and get detection results
                detection_results = detect_and_classify_food(image)
                
                # Prepare response with detection results
                response = {
                    'success': True,
                    'detections': detection_results,
                    'total_detections': len(detection_results),
                    'class_counts': food_counts
                }
                
                return jsonify(response)
            
            else:
                return jsonify({"error": "File upload failed"}), 400
                
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    
    elif request.method == "GET":
        return jsonify({"error": "GET method not supported for image processing"}), 405

@app.route("/video", methods=['GET', 'POST'])
def video():
    global food_counts
    
    if request.method == "POST":
        try:
            # Reset counts for new detection
            food_counts = {}
            
            if 'file' not in request.files:
                return jsonify({"error": "No file part"}), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
            
            if file:
                # Save uploaded video
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
                file.save(video_path)
                
                # Process video
                cap = cv2.VideoCapture(video_path)
                
                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Setup video writer
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_video.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                frame_count = 0
                process_every_n_frames = 5  # Process every 5th frame for performance
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % process_every_n_frames == 0:
                        processed_frame = detect_and_classify_food(frame)
                    else:
                        processed_frame = frame
                    
                    out.write(processed_frame)
                    frame_count += 1
                
                cap.release()
                out.release()
                
                # Prepare response
                food_detection_results = []
                for food_name, count in food_counts.items():
                    food_detection_results.append({
                        'food_name': food_name,
                        'count': count
                    })
                
                response = {
                    'success': True,
                    'video_processed': True,
                    'food_detections': food_detection_results,
                    'total_items': sum(food_counts.values())
                }
                
                return jsonify(response)
            
            else:
                return jsonify({"error": "File upload failed"}), 400
                
        except Exception as e:
            return jsonify({"error": f"Video processing failed: {str(e)}"}), 500
    
    elif request.method == "GET":
        # Return the last processed video if it exists
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_video.mp4')
        if os.path.exists(output_path):
            return send_file(output_path, mimetype='video/mp4')
        else:
            return jsonify({"error": "No processed video found"}), 404

@app.route("/download/<file_type>")
def download_file(file_type):
    """Download processed files"""
    if file_type == 'image':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.jpg')
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name='food_detection_result.jpg')
    elif file_type == 'video':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_video.mp4')
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name='food_detection_result.mp4')
    
    return jsonify({"error": "File not found"}), 404

