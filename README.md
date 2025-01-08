# Flash Detection Web App

This web application analyzes video files to detect instances of flashing lights and provides timestamps where they occur. This can be useful for content warnings or identifying potentially problematic content for photosensitive individuals. You can test it out at `https://flashdetection.onrender.com/`

## Features

- Upload video files through a web interface
- Analyze videos for sudden changes in brightness (flashing lights)
- Get precise timestamps of detected flashing sequences
- Generate preview videos with flash warnings
- Create safe versions with dangerous flashes removed
- Generate detailed PDF reports
- Modern, user-friendly interface
- Supports common video formats
- Batch processing support

## Prerequisites

1. Python 3.8 or higher
2. ffmpeg (required for video processing)

### Installing ffmpeg

#### On macOS:
```bash
brew install ffmpeg
```

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

#### On Windows:
1. Download ffmpeg from https://ffmpeg.org/download.html
2. Add it to your system PATH

## Setup

1. Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Click the "Choose Video Files" button or drag and drop videos onto the upload area
2. Click "Analyze Videos" to process the files
3. Wait for the analysis to complete
4. View the results, including:
   - Flash detection timestamps
   - Preview video with warnings
   - Safe version with dangerous flashes removed
   - Detailed PDF report

## Technical Details

- Flash detection uses OpenCV for video processing
- Detection is based on analyzing frame-to-frame brightness changes
- Videos are processed server-side with Python
- Maximum file size: 32MB
- Supports common video formats (mp4, avi, mov)
- Generated videos use H.264 codec for web compatibility

## Notes

- Processing time depends on the video length and resolution
- The application follows WCAG 2.1 guidelines for flash detection
- Uploaded videos are automatically deleted after processing
- Preview and safe versions are stored in the static directory 
