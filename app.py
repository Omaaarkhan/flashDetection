import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
from reportlab.pdfgen import canvas

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['PREVIEW_FOLDER'] = 'static/previews'
app.config['SAFE_VERSIONS_FOLDER'] = 'static/safe_versions'
app.config['REPORTS_FOLDER'] = 'static/reports'

# Ensure required directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['PREVIEW_FOLDER'], 
               app.config['SAFE_VERSIONS_FOLDER'], app.config['REPORTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

def detect_flashing_lights(video_path):
    """Detect flashing lights in video and return timestamps."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    flash_events = []
    
    # Parameters for flash detection
    frame_window = int(fps * 0.25)  # Reduced to quarter-second windows for faster response
    brightness_history = []
    last_flash_frame = -frame_window
    
    frame_count = 0
    prev_brightness = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness (normalized)
        current_brightness = np.mean(gray) / 255.0
        brightness_history.append(current_brightness)
        
        # Immediate frame-to-frame check for sudden changes
        if prev_brightness is not None:
            instant_change = abs(current_brightness - prev_brightness)
            if instant_change > 0.1:  # Detect sudden changes immediately
                timestamp = frame_count / fps
                if not flash_events or abs(timestamp - flash_events[-1]["timestamp"]) > 0.1:
                    flash_events.append({
                        "timestamp": timestamp,
                        "intensity": instant_change * 100,
                        "risk_level": "HIGH" if instant_change > 0.2 else "MEDIUM"
                    })
                    last_flash_frame = frame_count
        
        # Keep only the recent history
        if len(brightness_history) > frame_window:
            brightness_history.pop(0)
        
        # Pattern analysis when we have enough frames
        if len(brightness_history) == frame_window:
            # Calculate frame-to-frame differences
            differences = np.diff(brightness_history)
            
            # Look for alternating patterns (positive to negative changes)
            alternating_count = 0
            for i in range(len(differences)-1):
                if (differences[i] * differences[i+1]) < 0:  # Sign change
                    alternating_count += 1
            
            # Calculate max brightness change in window
            max_change = max(abs(np.diff(brightness_history)))
            
            # Detect flashes based on patterns
            if (max_change > 0.12 and  # More sensitive threshold
                alternating_count >= 2 and  # Detect fewer alternations
                frame_count - last_flash_frame > frame_window/4):  # Shorter minimum gap
                
                timestamp = frame_count / fps
                
                # Only add if not too close to previous detection
                if not flash_events or abs(timestamp - flash_events[-1]["timestamp"]) > 0.1:
                    risk_level = "MEDIUM"
                    if max_change > 0.2 or alternating_count >= 4:
                        risk_level = "HIGH"
                    
                    flash_events.append({
                        "timestamp": timestamp,
                        "intensity": max_change * 100,
                        "risk_level": risk_level
                    })
                    last_flash_frame = frame_count
        
        prev_brightness = current_brightness
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
            
        # Add warning text if needed
        cv2.putText(frame, "Preview Version", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Convert to MP4 using ffmpeg, preserving audio
    os.system(f'ffmpeg -y -i {temp_path} -i {video_path} -c:v libx264 -c:a aac -map 0:v:0 -map 1:a:0? {output_path}')
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
    c.drawString(72, 800, f"Flash Analysis Report - {video_name}")
    c.drawString(72, 780, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    y = 740
    c.drawString(72, y, "Detected Flash Events:")
    y -= 20
    
    for event in flash_events:
        c.drawString(72, y, f"Timestamp: {event['timestamp']:.2f}s")
        c.drawString(72, y-15, f"Risk Level: {event['risk_level']}")
        c.drawString(72, y-30, f"Intensity: {event['intensity']:.2f}")
        y -= 50
    
    c.save()

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
    
    if file.filename == '':
        logger.error("Empty filename received")
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to: {file_path}")
        file.save(file_path)
        
        logger.info(f"Processing file: {filename}")
        
        # Analyze video for flashing lights
        logger.info(f"Analyzing video: {filename}")
        flash_events = detect_flashing_lights(file_path)
        logger.info(f"Found {len(flash_events)} flash events")
        
        # Generate preview version
        preview_filename = f"preview_{filename}.mp4"
        preview_path = os.path.join(app.config['PREVIEW_FOLDER'], preview_filename)
        logger.info(f"Generating preview at: {preview_path}")
        generate_preview(file_path, preview_path)
        
        # Create safe version
        safe_filename = f"safe_{filename}.mp4"
        safe_path = os.path.join(app.config['SAFE_VERSIONS_FOLDER'], safe_filename)
        logger.info(f"Generating safe version at: {safe_path}")
        create_safe_version(file_path, safe_path, flash_events, sensitivity)
        
        # Generate PDF report
        report_filename = f"report_{filename}.pdf"
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
        logger.info(f"Generating report at: {report_path}")
        generate_pdf_report(filename, flash_events, report_path)
        
        # Clean up uploaded file
        os.remove(file_path)
        logger.info("Cleaned up uploaded file")
        
        # Verify files exist
        if not os.path.exists(preview_path):
            raise Exception("Preview file was not created")
        if not os.path.exists(safe_path):
            raise Exception("Safe version file was not created")
        if not os.path.exists(report_path):
            raise Exception("Report file was not created")
        
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 