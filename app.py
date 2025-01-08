import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
from reportlab.pdfgen import canvas
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFlashDetector:
    def __init__(self):
        # Initialize a Random Forest Classifier
        self.model = RandomForestClassifier(n_estimators=50)
        
    def extract_features(self, frame):
        """Extract simple features from a frame."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate basic statistics as features
        features = [
            np.mean(gray),           # Average brightness
            np.std(gray),            # Brightness variation
            np.max(gray),            # Maximum brightness
            np.min(gray),            # Minimum brightness
            np.median(gray),         # Median brightness
        ]
        return features
    
    def predict_frame(self, frame):
        """Predict if a frame contains a flash."""
        features = self.extract_features(frame)
        features = np.array(features).reshape(1, -1)
        
        # For initial use, we'll use a simple threshold-based approach
        # Later, we can train the model with actual data
        brightness_mean = features[0][0]
        brightness_std = features[0][1]
        
        # Increased thresholds to reduce sensitivity
        is_flash = brightness_mean > 230 or brightness_std > 70  # Previously 200 and 50
        confidence = min((brightness_mean / 255.0) * 1.2, 1.0)
        
        return is_flash, confidence

def detect_flashing_lights(video_path):
    """Enhanced flash detection using ML."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    flash_events = []
    
    # Initialize detector
    detector = SimpleFlashDetector()
    frame_count = 0
    
    # Keep track of recent brightness values for pattern detection
    brightness_history = []
    window_size = int(fps * 0.5)  # Half-second window
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get ML prediction
        is_flash, confidence = detector.predict_frame(frame)
        
        # Store brightness history
        current_brightness = detector.extract_features(frame)[0]  # Mean brightness
        brightness_history.append(current_brightness)
        
        # Keep history within window size
        if len(brightness_history) > window_size:
            brightness_history.pop(0)
        
        # Analyze patterns if we have enough history
        if len(brightness_history) == window_size:
            # Calculate frame-to-frame changes
            changes = np.diff(brightness_history)
            
            # More strict pattern detection
            alternating_pattern = (
                len(changes) >= 4 and  # Need at least 4 changes
                all(abs(change) > 50 for change in changes) and  # Significant changes
                all(changes[i] * changes[i+1] < 0 for i in range(len(changes)-1))  # Alternating
            )
            
            if is_flash or alternating_pattern:
                timestamp = frame_count / fps
                
                # More strict risk level determination
                risk_level = "HIGH" if (confidence > 0.9 or  # Increased from 0.8
                                     (alternating_pattern and abs(np.mean(changes)) > 70)  # Added intensity check
                                     ) else "MEDIUM"
                
                # Add to flash events if not too close to previous event
                if not flash_events or abs(timestamp - flash_events[-1]["timestamp"]) > 0.5:  # Increased from 0.3
                    flash_events.append({
                        "timestamp": timestamp,
                        "intensity": confidence * 100,
                        "risk_level": risk_level
                    })
        
        frame_count += 1
    
    cap.release()
    return flash_events

def generate_preview(video_path, output_path):
    """Generate a preview version of the video with flash warnings."""
    logger.info(f"Generating preview for: {os.path.basename(video_path)}")
    
    # Create temporary file for processing
    temp_path = f"{output_path}.temp.avi"
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add warning text
        cv2.putText(frame, "Preview Version", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Convert to MP4 using ffmpeg with specific codec settings
    os.system(f'ffmpeg -y -i {temp_path} -i {video_path} -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 128k -movflags +faststart -map 0:v:0 -map 1:a:0? {output_path}')
    os.remove(temp_path)

def create_safe_version(video_path, output_path, flash_events, sensitivity='all'):
    """Create a version of the video with dangerous flash segments blacked out."""
    logger.info(f"Creating safe version for: {os.path.basename(video_path)}")
    
    # Create temporary file for processing
    temp_path = f"{output_path}.temp.avi"
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    frame_count = 0
    buffer_frames = int(fps * 0.2)  # Reduced buffer to 0.2 seconds
    
    # Pre-process flash events based on sensitivity
    filtered_events = []
    for event in flash_events:
        if sensitivity == 'high':
            if event["risk_level"] == "HIGH":
                filtered_events.append(event)
        else:  # 'all' - include both medium and high risk flashes
            filtered_events.append(event)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate current timestamp
        timestamp = frame_count / fps
        
        # Check if current frame is during a flash event
        is_flash_frame = False
        for event in filtered_events:
            # Check if within buffer period of a flash event
            time_diff = abs(event["timestamp"] - timestamp)
            if time_diff < 0.3:  # Reduced to 0.3 second window
                is_flash_frame = True
                break
        
        if is_flash_frame:
            # Create black frame with warning message
            warning_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add warning text
            text = "⚠️ WARNING: Flashing Lights Detected"
            subtext = "This segment has been modified for safety"
            
            # Calculate font size based on video dimensions
            base_font_scale = min(width, height) / 2500
            main_font_scale = max(0.4, base_font_scale * 1.1)
            sub_font_scale = max(0.3, base_font_scale * 0.8)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            main_thickness = max(1, int(main_font_scale * 1.5))
            sub_thickness = max(1, int(sub_font_scale * 1.5))
            
            # Get text sizes for centering
            main_text_size = cv2.getTextSize(text, font, main_font_scale, main_thickness)[0]
            sub_text_size = cv2.getTextSize(subtext, font, sub_font_scale, sub_thickness)[0]
            
            # Center the text
            main_text_x = (width - main_text_size[0]) // 2
            main_text_y = (height // 2)
            sub_text_x = (width - sub_text_size[0]) // 2
            sub_text_y = main_text_y + int(height * 0.1)
            
            def draw_outlined_text(img, text, x, y, font_scale, thickness):
                outline_color = (0, 0, 0)
                text_color = (255, 255, 255)
                outline_thickness = thickness + 2
                
                offsets = [(ox, oy) for ox in range(-2, 3, 1) for oy in range(-2, 3, 1)]
                for ox, oy in offsets:
                    cv2.putText(img, text, (x + ox, y + oy), font, font_scale, outline_color, outline_thickness)
                
                cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
            
            # Draw text with outline
            draw_outlined_text(warning_frame, text, main_text_x, main_text_y, main_font_scale, main_thickness)
            draw_outlined_text(warning_frame, subtext, sub_text_x, sub_text_y, sub_font_scale, sub_thickness)
            
            # Add warning border
            border_thickness = max(3, int(min(width, height) * 0.01))
            cv2.rectangle(warning_frame, (border_thickness, border_thickness), 
                         (width-border_thickness, height-border_thickness), 
                         (0, 0, 255), border_thickness)
            
            frame = warning_frame
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Convert to MP4 using ffmpeg, preserving audio
    os.system(f'ffmpeg -y -i {temp_path} -i {video_path} -c:v libx264 -c:a aac -map 0:v:0 -map 1:a:0? {output_path}')
    os.remove(temp_path)

def generate_pdf_report(video_name, flash_events, output_path):
    """Generate a PDF report with analysis results."""
    logger.info(f"Generating PDF report for: {video_name}")
    
    c = canvas.Canvas(output_path)
    
    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(72, 800, "Flash Analysis Report")
    
    # Video details
    c.setFont("Helvetica", 12)
    c.drawString(72, 770, "Video:")
    
    # Handle long video names by starting a new line
    if len(video_name) > 60:  # If name is too long
        c.drawString(92, 750, video_name[:60])
        c.drawString(92, 730, video_name[60:])
        y = 710  # Adjust starting y position for next items
    else:
        c.drawString(92, 750, video_name)
        y = 730  # Standard y position if name fits on one line
    
    c.drawString(72, y, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20
    c.drawString(72, y, f"Total Flash Events Detected: {len(flash_events)}")
    
    # Summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y-40, "Risk Level Summary:")
    c.setFont("Helvetica", 12)
    high_risk = sum(1 for event in flash_events if event["risk_level"] == "HIGH")
    medium_risk = sum(1 for event in flash_events if event["risk_level"] == "MEDIUM")
    c.drawString(92, y-60, f"• High Risk Events: {high_risk}")
    c.drawString(92, y-80, f"• Medium Risk Events: {medium_risk}")
    
    # Detailed Events List
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y-120, "Detailed Flash Events:")
    
    y = y-160
    for i, event in enumerate(flash_events, 1):
        if y < 100:  # Start new page if near bottom
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(72, 800, "Detailed Flash Events (continued):")
            y = 770
            
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, f"Event #{i}")
        c.setFont("Helvetica", 12)
        y -= 20
        c.drawString(92, y, f"Timestamp: {event['timestamp']:.2f} seconds")
        y -= 20
        c.drawString(92, y, f"Risk Level: {event['risk_level']}")
        y -= 20
        c.drawString(92, y, f"Intensity: {event['intensity']:.1f}%")
        y -= 40
    
    # Footer with disclaimer
    c.setFont("Helvetica", 10)
    c.drawString(72, 50, "Note: This report is generated automatically and should be used as a reference only.")
    c.drawString(72, 35, "Please consult with medical professionals for specific photosensitivity concerns.")
    
    c.save()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['PREVIEW_FOLDER'] = 'static/previews'
app.config['SAFE_VERSIONS_FOLDER'] = 'static/safe_versions'
app.config['REPORTS_FOLDER'] = 'static/reports'

# Ensure required directories exist
@app.before_request
def create_directories():
    for folder in [app.config['UPLOAD_FOLDER'], 
                   app.config['PREVIEW_FOLDER'],
                   app.config['SAFE_VERSIONS_FOLDER'], 
                   app.config['REPORTS_FOLDER']]:
        os.makedirs(folder, exist_ok=True)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files from various folders."""
    logger.info(f"Attempting to serve static file: {filename}")
    if 'previews' in filename:
        return send_from_directory('static/previews', filename.replace('previews/', ''))
    elif 'safe_versions' in filename:
        return send_from_directory('static/safe_versions', filename.replace('safe_versions/', ''))
    elif 'reports' in filename:
        return send_from_directory('static/reports', filename.replace('reports/', ''))
    return 'File not found', 404

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info("Upload endpoint called")
    
    if 'video' not in request.files:
        logger.error("No video file in request")
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    sensitivity = request.form.get('sensitivity', 'all')
    
    logger.info(f"Received file: {file.filename} with sensitivity: {sensitivity}")
    
    try:
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to: {file_path}")
        file.save(file_path)
        
        # Analyze video for flashing lights
        logger.info(f"Analyzing video: {filename}")
        flash_events = detect_flashing_lights(file_path)
        logger.info(f"Found {len(flash_events)} flash events")
        
        # Generate preview version
        preview_filename = f"preview_{filename}.mp4"
        preview_path = os.path.join(app.config['PREVIEW_FOLDER'], preview_filename)
        logger.info(f"Generating preview at: {preview_path}")
        generate_preview(file_path, preview_path)
        logger.info("Preview generation complete")
        
        # Create safe version
        safe_filename = f"safe_{filename}.mp4"
        safe_path = os.path.join(app.config['SAFE_VERSIONS_FOLDER'], safe_filename)
        logger.info(f"Generating safe version at: {safe_path}")
        create_safe_version(file_path, safe_path, flash_events, sensitivity)
        logger.info("Safe version generation complete")
        
        # Generate PDF report
        report_filename = f"report_{filename}.pdf"
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
        logger.info(f"Generating report at: {report_path}")
        generate_pdf_report(filename, flash_events, report_path)
        logger.info("PDF report generation complete")
        
        # Verify files exist and log their sizes
        for path, desc in [(preview_path, "Preview"), (safe_path, "Safe version"), (report_path, "Report")]:
            if os.path.exists(path):
                size = os.path.getsize(path)
                logger.info(f"{desc} file exists, size: {size} bytes")
            else:
                raise Exception(f"{desc} file was not created")
        
        # Clean up uploaded file
        os.remove(file_path)
        logger.info("Cleaned up uploaded file")
        
        response_data = {
            'message': 'Video processed successfully',
            'flash_events': flash_events,
            'preview_url': f"/static/previews/{preview_filename}",
            'safe_version_url': f"/static/safe_versions/{safe_filename}",
            'report_url': f"/static/reports/{report_filename}",
            'filename': filename
        }
        logger.info("Sending response with results")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False) 