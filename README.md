# 🧮 People Counter

A real-time people counting application built with Streamlit, OpenCV, and YOLOv8 for detecting and tracking people in videos or webcam feeds.

## Features

- **Real-time Detection**: Uses YOLOv8 for accurate person detection
- **Object Tracking**: Implements centroid-based tracking with Hungarian algorithm
- **Directional Counting**: Counts people moving up and down across a virtual line
- **Multiple Input Sources**: Support for video uploads and webcam feeds
- **Configurable Parameters**: Adjustable tracking sensitivity and detection settings
- **Live Statistics**: Real-time display of counting results
- **Progress Tracking**: Visual progress bar for video processing

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd People_counter
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 model (optional):**
   The application will automatically download the YOLOv8 model on first run, or you can manually download it:
   ```bash
   python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
   ```

## Usage

### Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run people_counter.py
   ```

2. **Open your browser** and navigate to the displayed URL (usually `http://localhost:8501`)

### Configuration

Use the sidebar to adjust these parameters:

- **Max Disappeared Frames**: How many frames an object can be missing before being removed (20-100)
- **Max Distance Threshold**: Maximum distance for object association (50-150 pixels)
- **Detection Confidence**: YOLO detection confidence threshold (0.1-1.0)
- **Detection Interval**: Run detection every N frames for performance (1-10)
- **Frame Dimensions**: Output frame size (400x300 to 800x600)

### Input Sources

1. **Upload Video**: Upload MP4, AVI, MOV, or MKV files
2. **Use Webcam**: Real-time processing from your camera

### How It Works

1. **Detection**: YOLOv8 detects people in video frames
2. **Tracking**: Centroid tracker maintains object IDs across frames
3. **Direction Analysis**: Movement direction is calculated based on centroid positions
4. **Counting**: People crossing the virtual line are counted as "up" or "down"

## Project Structure

```
People_counter/
├── people_counter.py      # Main application file
├── tracker.py            # Object tracking implementation
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Technical Details

### Tracker Implementation

- **SimpleCentroidTracker**: Uses centroid-based tracking with Hungarian algorithm for optimal object association
- **SimpleTrackableObject**: Stores tracking history and counting status for each object
- **Performance Optimizations**: Configurable detection intervals and distance thresholds

### Detection Pipeline

1. Frame preprocessing and resizing
2. YOLO person detection (class 0 in COCO dataset)
3. Bounding box extraction and centroid calculation
4. Object tracking and ID assignment
5. Direction analysis and counting
6. Visualization and statistics display

## Performance Tips

- **Detection Interval**: Increase for faster processing (trade-off with accuracy)
- **Frame Size**: Smaller frames process faster but may reduce detection accuracy
- **Confidence Threshold**: Higher values reduce false positives but may miss people
- **Distance Threshold**: Adjust based on your video's camera setup and movement patterns

## Troubleshooting

### Common Issues

1. **Model Download Fails**: Check internet connection and try manual download
2. **Webcam Not Working**: Ensure camera permissions and try different camera index
3. **Slow Performance**: Reduce frame size, increase detection interval, or use GPU acceleration
4. **Memory Issues**: Process shorter videos or reduce frame dimensions

### GPU Acceleration

For better performance, install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Dependencies

- **Streamlit**: Web application framework
- **OpenCV**: Computer vision and video processing
- **Ultralytics**: YOLOv8 object detection
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing (Hungarian algorithm)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- YOLOv8 by Ultralytics for object detection
- OpenCV community for computer vision tools
- Streamlit team for the web framework
