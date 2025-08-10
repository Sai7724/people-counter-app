import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
from tracker import SimpleCentroidTracker, SimpleTrackableObject
import os
import time
import threading


class PeopleCounter:
    """Main people counter application class."""
    
    def __init__(self):
        # Add CSS for maximize/minimize styling
        self.add_maximize_css()
        
        # Initialize session state for model
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'replay_frames' not in st.session_state:
            st.session_state.replay_frames = []
        if 'replay_count' not in st.session_state:
            st.session_state.replay_count = 0
        if 'replay_history' not in st.session_state:
            st.session_state.replay_history = []
        if 'replay_paused' not in st.session_state:
            st.session_state.replay_paused = False
        if 'replay_stopped' not in st.session_state:
            st.session_state.replay_stopped = False
        
        # Initialize webcam-related session state with better defaults
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'paused' not in st.session_state:
            st.session_state.paused = False
        if 'webcam_stop_requested' not in st.session_state:
            st.session_state.webcam_stop_requested = False
        if 'webcam_pause_requested' not in st.session_state:
            st.session_state.webcam_pause_requested = False
        if 'webcam_initialized' not in st.session_state:
            st.session_state.webcam_initialized = False
        if 'video_maximized' not in st.session_state:
            st.session_state.video_maximized = False
            
        self.model = st.session_state.model
        self.tracker = None
        self.trackable_objects = {}
        self.total_up = 0
        self.total_down = 0
        self.total_frames = 0
        self.vs = None  # Video capture object
        self.processing_thread = None  # Thread for processing
    
    def add_maximize_css(self):
        """Add CSS styling for maximize/minimize functionality and responsive design."""
        st.markdown("""
        <style>
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-header {
                font-size: 1.5rem !important;
                text-align: center !important;
            }
            .sidebar-content {
                font-size: 0.9rem !important;
            }
            .metric-container {
                margin: 5px 0 !important;
            }
            .button-container {
                margin: 10px 0 !important;
            }
            .video-container {
                max-width: 100% !important;
                margin: 10px 0 !important;
            }
        }
        
        @media (max-width: 480px) {
            .main-header {
                font-size: 1.2rem !important;
            }
            .sidebar-content {
                font-size: 0.8rem !important;
            }
            .metric-container {
                margin: 3px 0 !important;
            }
            .button-container {
                margin: 5px 0 !important;
            }
        }
        
        /* Video Styling */
        .maximized-video {
            border: 4px solid #00ff00 !important;
            border-radius: 15px !important;
            box-shadow: 0 8px 16px rgba(0,255,0,0.3) !important;
            transform: scale(1.05) !important;
            transition: all 0.3s ease !important;
            position: relative !important;
            z-index: 1000 !important;
            max-width: 100% !important;
            height: auto !important;
        }
        
        .normal-video {
            border: 2px solid #ddd !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            max-width: 100% !important;
            height: auto !important;
        }
        
        /* Responsive Video */
        .stImage > img {
            max-width: 100% !important;
            height: auto !important;
            object-fit: contain !important;
        }
        
        /* Mobile-friendly buttons */
        .stButton > button {
            min-height: 44px !important;
            font-size: 14px !important;
            padding: 8px 16px !important;
        }
        
        @media (max-width: 768px) {
            .stButton > button {
                min-height: 48px !important;
                font-size: 16px !important;
                padding: 12px 20px !important;
            }
        }
        
        /* Responsive columns */
        .responsive-columns {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 10px !important;
        }
        
        @media (max-width: 768px) {
            .responsive-columns {
                flex-direction: column !important;
            }
        }
        
        /* Floating maximize button */
        .floating-maximize-btn {
            position: fixed !important;
            top: 20px !important;
            right: 20px !important;
            z-index: 9999 !important;
            background: rgba(0,255,0,0.9) !important;
            border: 2px solid #00ff00 !important;
            border-radius: 50% !important;
            width: 60px !important;
            height: 60px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }
        
        @media (max-width: 768px) {
            .floating-maximize-btn {
                width: 50px !important;
                height: 50px !important;
                top: 10px !important;
                right: 10px !important;
            }
        }
        
        .floating-maximize-btn:hover {
            background: rgba(0,255,0,1) !important;
            transform: scale(1.1) !important;
        }
        
        /* Touch-friendly interface */
        @media (max-width: 768px) {
            .stSlider > div > div > div > div {
                min-height: 44px !important;
            }
            
            .stSelectbox > div > div > div {
                min-height: 44px !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def display_video_with_maximize(self, frame, video_placeholder, is_maximized=False):
        """Display video with maximize/minimize styling."""
        if is_maximized:
            # Add CSS class for maximized video
            st.markdown("""
            <style>
            .stImage > img {
                border: 4px solid #00ff00 !important;
                border-radius: 15px !important;
                box-shadow: 0 8px 16px rgba(0,255,0,0.3) !important;
                transform: scale(1.05) !important;
                transition: all 0.3s ease !important;
                position: relative !important;
                z-index: 1000 !important;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Display the video
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        
    def load_model(self):
        """Load YOLO model with error handling."""
        try:
            st.info("Loading YOLO model... This may take a few minutes on first run.")
            # Try to load the model with better error handling
            model = YOLO("yolov8n.pt")  # Use smaller model first
            st.session_state.model = model
            st.session_state.model_loaded = True
            self.model = model
            st.success("YOLO model loaded successfully!")
            return True
        except Exception as e:
            st.error(f"Failed to load YOLO model: {str(e)}")
            st.info("Trying alternative model...")
            try:
                # Try with a different model variant
                model = YOLO("yolov8s.pt")
                st.session_state.model = model
                st.session_state.model_loaded = True
                self.model = model
                st.success("Alternative YOLO model loaded successfully!")
                return True
            except Exception as e2:
                st.error(f"Failed to load alternative model: {str(e2)}")
                st.error("Please check your internet connection and try again.")
                return False
        
    def setup_ui(self):
        """Setup the Streamlit user interface."""
        st.set_page_config(
            page_title="People Counter",
            page_icon="🧮",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add mobile-friendly viewport
        st.markdown("""
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        """, unsafe_allow_html=True)
        
        st.title("🧮 People Counter")
        st.markdown("---")
        
        # Add responsive container
        st.markdown('<div class="responsive-container">', unsafe_allow_html=True)
        
        # Sidebar for configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model loading section
            if not st.session_state.model_loaded or st.session_state.model is None:
                st.info("📋 Model Status: Not Loaded")
                if st.button("Load YOLO Model", use_container_width=True):
                    self.load_model()
            else:
                st.success("✅ Model Ready")
                st.info(f"📋 Model Status: Loaded (yolov8n.pt)")
                
                # Add reload model option
                if st.button("🔄 Reload Model", use_container_width=True):
                    st.session_state.model_loaded = False
                    st.session_state.model = None
                    self.model = None
                    st.rerun()
                
                # Detection settings
                st.subheader("Detection Settings")
                confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
                
                # Tracking settings
                st.subheader("Tracking Settings")
                max_disappeared = st.slider("Max Disappeared Frames", 10, 100, 40, 5)
                max_distance = st.slider("Max Distance", 25, 150, 75, 5)
                process_every_n_frames = st.slider("Process Every N Frames", 1, 10, 5, 1)
                
                # Webcam status
                st.subheader("📷 Webcam Status")
                if st.session_state.webcam_initialized:
                    st.success("✅ Webcam Connected")
                else:
                    st.warning("⚠️ Webcam Not Connected")
                
                # Display current counts
                st.subheader("📊 Current Counts")
                st.metric("People Moving Up", self.total_up)
                st.metric("People Moving Down", self.total_down)
                st.metric("Total Frames", self.total_frames)
                
                # Reset button
                if st.button("🔄 Reset Counts"):
                    self.reset_counts()
                
                # Session reset button (for debugging)
                if st.button("🔄 Reset Session"):
                    self.reset_session()
        
        # Main content area
        if not st.session_state.model_loaded or st.session_state.model is None:
            st.info("👆 Use the sidebar to load the YOLO model before starting.")
            return
        
        # Update the model reference
        self.model = st.session_state.model
        
        # Show model status
        st.success("🚀 YOLO Model is loaded and ready for people detection!")
        
        # Add replay section if frames are available
        if st.session_state.replay_frames:
            st.subheader("🎬 Replay & Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"📹 {len(st.session_state.replay_frames)} frames available for replay")
                replay_speed = st.slider("Replay Speed", 0.5, 3.0, 1.0, 0.1, key="main_replay_speed")
                if st.button("🔄 Replay Video", type="primary", key="main_replay_button"):
                    self.replay_video(confidence, max_disappeared, max_distance, process_every_n_frames, replay_speed)
            
            with col2:
                if st.button("📹 Download Processed Video", type="secondary", key="main_download_video"):
                    self.download_processed_video()
                if st.button("📊 Download Results CSV", type="secondary", key="main_download_csv"):
                    self.download_results_csv()
            
            with col3:
                if st.button("🗑️ Clear Replay Data", type="secondary"):
                    st.session_state.replay_frames = []
                    st.session_state.replay_history = []
                    st.rerun()
            
            # Show replay history
            if st.session_state.replay_history:
                st.subheader("📋 Replay History")
                for replay in st.session_state.replay_history:
                    with st.expander(f"Replay #{replay['replay_number']} - {replay['timestamp']}"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Up", replay['up_count'])
                        with col2:
                            st.metric("Down", replay['down_count'])
                        with col3:
                            st.metric("Frames", replay['total_frames'])
                        with col4:
                            st.metric("Time", f"{replay['processing_time']:.2f}s")
            
            st.markdown("---")
        
        # Input source selection - responsive layout
        st.subheader("📹 Input Source")
        
        # Use responsive columns that stack on mobile
        col1, col2 = st.columns([2, 1])
        
        with col1:
            option = st.radio("Select Input Source", ["Upload Video", "Use Webcam"], 
                             help="Choose between uploading a video file or using your webcam")
        
        with col2:
            st.subheader("🎯 Detection Line")
            st.info("The yellow line represents the counting boundary. People crossing above the line count as 'Up', below as 'Down'.")
        
        # Video input handling
        video_path = None
        if option == "Upload Video":
            uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                st.success(f"✅ Video uploaded: {uploaded_file.name}")
                
                # Start counting button for uploaded video
                if st.button("🚀 Start Counting", type="primary", key="upload_start"):
                    self.start_counting(video_path, confidence, max_disappeared, max_distance, process_every_n_frames)
        elif option == "Use Webcam":
            video_path = 0  # Webcam
            
            # Check webcam availability
            if self.check_webcam_availability():
                st.success("📷 Webcam is available and accessible!")
                
                # Webcam controls with responsive layout
                st.subheader("🎮 Webcam Controls")
                
                # Primary controls row
                col1, col2 = st.columns(2)
                with col1:
                    if not st.session_state.webcam_active and not st.session_state.processing:
                        if st.button("🚀 Start Webcam", type="primary", key="webcam_start", use_container_width=True):
                            st.session_state.webcam_active = True
                            st.session_state.processing = True
                            st.session_state.webcam_stop_requested = False
                            st.session_state.webcam_pause_requested = False
                            self.start_counting(video_path, confidence, max_disappeared, max_distance, process_every_n_frames)
                
                with col2:
                    if st.session_state.webcam_active or st.session_state.processing:
                        if st.button("🛑 Stop Webcam", type="secondary", key="webcam_stop", use_container_width=True):
                            st.session_state.webcam_stop_requested = True
                            self.stop_webcam()
                
                # Secondary controls row
                col3, col4 = st.columns(2)
                with col3:
                    if st.button("⏸️ Pause/Resume", type="secondary", key="webcam_pause", use_container_width=True):
                        if st.session_state.webcam_active:
                            st.session_state.webcam_pause_requested = not st.session_state.webcam_pause_requested
                            st.rerun()
                
                with col4:
                    if st.button("🔄 Reconnect", type="secondary", key="webcam_reconnect", use_container_width=True):
                        with st.spinner("Reconnecting to webcam..."):
                            if self.reconnect_webcam():
                                st.session_state.webcam_initialized = True
                                st.success("✅ Webcam reconnected successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to reconnect. Please check your camera connection.")
            else:
                st.error("❌ Webcam is not accessible. Please check your camera connection and permissions.")
                
                # Enhanced troubleshooting section
                with st.expander("🔧 Troubleshooting Tips"):
                    st.info("**Common Issues & Solutions:**")
                    st.markdown("""
                    • **Camera in use**: Close other applications that might be using your camera (Zoom, Teams, etc.)
                    • **Windows permissions**: Go to Settings > Privacy > Camera and ensure camera access is enabled
                    • **Driver issues**: Update your camera drivers in Device Manager
                    • **Hardware**: Check if your camera is properly connected and not disabled
                    • **Browser permissions**: Ensure your browser has camera access
                    """)
                    
                    # Try to reconnect button
                    if st.button("🔄 Try to Reconnect", type="primary", key="troubleshoot_reconnect"):
                        with st.spinner("Attempting to reconnect..."):
                            if self.reconnect_webcam():
                                st.success("✅ Webcam reconnected successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Reconnection failed. Please check the troubleshooting tips above.")
                
                return
        
        # Show processing status
        if st.session_state.get('processing', False):
            st.info("🔄 Processing in progress... Please wait.")
            if option == "Use Webcam" and st.session_state.get('webcam_active', False):
                if st.button("🛑 Stop Processing", type="secondary", key="stop_processing"):
                    self.stop_webcam()
        
        # Show warning if no input selected
        if video_path is None:
            st.warning("⚠️ Please select an input source first.")
        
        # Close responsive container
        st.markdown('</div>', unsafe_allow_html=True)
    
    def reset_counts(self):
        """Reset all counting variables."""
        self.total_up = 0
        self.total_down = 0
        self.total_frames = 0
        self.trackable_objects = {}
        st.success("Counts reset successfully!")
    
    def reset_session(self):
        """Reset the entire session state."""
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.replay_frames = []
        st.session_state.replay_count = 0
        st.session_state.replay_history = []
        self.model = None
        self.tracker = None
        self.trackable_objects = {}
        self.total_up = 0
        self.total_down = 0
        self.total_frames = 0
        st.success("Session reset successfully!")
        st.rerun()
    
    def store_replay_history(self, up_count, down_count, frame_count, processing_time):
        """Store replay results in history."""
        replay_data = {
            'replay_number': st.session_state.replay_count + 1,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'up_count': up_count,
            'down_count': down_count,
            'total_frames': frame_count,
            'processing_time': processing_time
        }
        st.session_state.replay_history.append(replay_data)
        st.session_state.replay_count += 1
    
    def replay_video(self, confidence, max_disappeared, max_distance, process_every_n_frames, replay_speed=1.0):
        """Replay the stored video with current settings."""
        if not st.session_state.replay_frames:
            st.error("❌ No video frames available for replay. Please process a video first.")
            return
        
        if not st.session_state.model_loaded or st.session_state.model is None:
            st.error("❌ YOLO model is not loaded. Please load the model first.")
            return
        
        try:
            # Reset counts for new replay
            self.total_up = 0
            self.total_down = 0
            self.total_frames = 0
            self.trackable_objects = {}
            
            # Initialize tracker
            self.tracker = SimpleCentroidTracker(max_disappeared=max_disappeared, max_distance=max_distance)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Video display
            st.subheader(f"🔄 Replay #{st.session_state.replay_count + 1}")
            video_placeholder = st.empty()
            
            # Replay controls - responsive layout
            st.subheader("🎮 Replay Controls")
            
            # Primary controls row
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("⏸️ Pause/Resume", type="secondary", key="replay_pause", use_container_width=True):
                    st.session_state.replay_paused = not st.session_state.get('replay_paused', False)
                    st.rerun()
            with col2:
                if st.button("⏹️ Stop Replay", type="secondary", key="replay_stop", use_container_width=True):
                    st.session_state.replay_stopped = True
                    st.rerun()
            with col3:
                if st.button("🔄 Reset Counts", type="secondary", key="replay_reset", use_container_width=True):
                    self.reset_replay_counts()
                    st.rerun()
            
            # Secondary controls row
            col4, col5 = st.columns(2)
            with col4:
                if st.button("📊 Download Results", type="secondary", key="replay_download", use_container_width=True):
                    self.download_results_csv()
            with col5:
                if st.button("🔍 Maximize" if not st.session_state.get('video_maximized', False) else "📱 Minimize", 
                            type="secondary", key="replay_maximize", use_container_width=True):
                    st.session_state.video_maximized = not st.session_state.get('video_maximized', False)
                    st.rerun()
            
            start_time = time.time()
            frame_count = 0
            total_frames = len(st.session_state.replay_frames)
            
            for frame_data in st.session_state.replay_frames:
                # Check for stop request
                if st.session_state.get('replay_stopped', False):
                    st.warning("⏹️ Replay stopped by user")
                    break
                
                # Check for pause request
                if st.session_state.get('replay_paused', False):
                    st.info("⏸️ Replay paused. Click Pause/Resume to continue.")
                    time.sleep(0.1)
                    continue
                frame = frame_data['frame']
                rectangles = frame_data['detections']
                
                # Process every N frames for performance
                if frame_count % process_every_n_frames == 0:
                    # Update tracker
                    objects = self.tracker.update(rectangles)
                    
                    # Process tracked objects
                    for (object_id, centroid) in objects.items():
                        to = self.trackable_objects.get(object_id, None)
                        if to is None:
                            to = SimpleTrackableObject(object_id, centroid)
                        else:
                            # Calculate direction
                            y_coords = [c[1] for c in to.centroids]
                            if len(y_coords) > 0:
                                direction = centroid[1] - np.mean(y_coords)
                                to.centroids.append(centroid)
                                
                                # Count crossing the line
                                if not to.counted:
                                    if direction < 0 and centroid[1] < frame.shape[0] // 2:
                                        self.total_up += 1
                                        to.counted = True
                                    elif direction > 0 and centroid[1] > frame.shape[0] // 2:
                                        self.total_down += 1
                                        to.counted = True
                        
                        self.trackable_objects[object_id] = to
                        
                        # Draw tracking info
                        cv2.putText(frame, f"ID {object_id}", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                
                # Draw counting line
                cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), (0, 255, 255), 2)
                
                # Display counts on frame
                cv2.putText(frame, f"Up: {self.total_up} | Down: {self.total_down}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Update display
                video_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Replaying frame {frame_count}/{total_frames}")
                
                # Control replay speed
                time.sleep(1.0 / (30 * replay_speed))  # 30 FPS base speed
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Store replay history
            self.store_replay_history(self.total_up, self.total_down, frame_count, processing_time)
            
            # Final results
            progress_bar.progress(1.0)
            status_text.text("✅ Replay complete!")
            
            st.success(f"🔄 Replay #{st.session_state.replay_count} Complete!")
            st.metric("Total People Moving Up", self.total_up)
            st.metric("Total People Moving Down", self.total_down)
            st.metric("Total Frames Processed", frame_count)
            st.metric("Processing Time", f"{processing_time:.2f}s")
            
        except Exception as e:
            st.error(f"❌ Error during replay: {str(e)}")
    
    def download_processed_video(self):
        """Download the processed video with annotations."""
        if not st.session_state.replay_frames:
            st.error("❌ No processed video available for download.")
            return
        
        try:
            # Create video writer
            output_path = "processed_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (600, 400))
            
            # Write frames
            for frame_data in st.session_state.replay_frames:
                frame = frame_data['frame']
                out.write(frame)
            
            out.release()
            
            # Read the file and create download button
            with open(output_path, "rb") as file:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=file.read(),
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
                
        except Exception as e:
            st.error(f"❌ Error creating video download: {str(e)}")
    
    def download_results_csv(self):
        """Download the counting results as CSV."""
        if not st.session_state.replay_frames:
            st.error("❌ No results available for download.")
            return
        
        try:
            # Prepare CSV data
            csv_data = []
            for i, frame_data in enumerate(st.session_state.replay_frames):
                csv_data.append({
                    'Frame': i + 1,
                    'People_Count': frame_data.get('people_count', 0),
                    'People_In': frame_data.get('people_in', 0),
                    'People_Out': frame_data.get('people_out', 0),
                    'Timestamp': frame_data.get('timestamp', '')
                })
            
            # Convert to CSV
            import pandas as pd
            df = pd.DataFrame(csv_data)
            csv_string = df.to_csv(index=False)
            
            # Create download button
            st.download_button(
                label="📊 Download Results CSV",
                data=csv_string,
                file_name="people_counting_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error creating CSV download: {str(e)}")
    
    def stop_webcam(self):
        """Stop the webcam and reset processing state."""
        try:
            # Stop any processing thread
            if hasattr(self, 'processing_thread') and self.processing_thread is not None:
                if self.processing_thread.is_alive():
                    st.session_state.webcam_stop_requested = True
                    self.processing_thread.join(timeout=2.0)  # Wait up to 2 seconds
            
            # Release video capture
            if hasattr(self, 'vs') and self.vs is not None:
                self.vs.release()
                self.vs = None
            
            # Reset all processing state
            st.session_state.processing = False
            st.session_state.webcam_active = False
            st.session_state.webcam_stop_requested = False
            st.session_state.webcam_pause_requested = False
            st.session_state.webcam_initialized = False
            
            # Clear any existing video capture objects
            cv2.destroyAllWindows()
            
            st.success("🛑 Webcam stopped successfully!")
            
        except Exception as e:
            st.error(f"Error stopping webcam: {str(e)}")
        finally:
            # Ensure state is reset even if there's an error
            st.session_state.processing = False
            st.session_state.webcam_active = False
            st.session_state.webcam_stop_requested = False
            st.session_state.webcam_pause_requested = False
            st.session_state.webcam_initialized = False
    
    def reconnect_webcam(self):
        """Attempt to reconnect to the webcam."""
        try:
            # Release existing connection
            if self.vs is not None:
                self.vs.release()
                self.vs = None
            
            # Wait a moment for cleanup
            time.sleep(1.0)
            
            # Try multiple camera indices and backends for better compatibility
            camera_connected = False
            
            # Try DirectShow first (Windows-specific, better compatibility)
            try:
                self.vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if self.vs.isOpened():
                    ret, frame = self.vs.read()
                    if ret and frame is not None:
                        camera_connected = True
                        # Reset frame position
                        self.vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        self.vs.release()
                        self.vs = None
            except:
                pass
            
            # If DirectShow failed, try default backend
            if not camera_connected:
                try:
                    self.vs = cv2.VideoCapture(0)
                    if self.vs.isOpened():
                        ret, frame = self.vs.read()
                        if ret and frame is not None:
                            camera_connected = True
                            # Reset frame position
                            self.vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        else:
                            self.vs.release()
                            self.vs = None
                except:
                    pass
            
            # Try alternative camera indices if primary fails
            if not camera_connected:
                for camera_index in [1, 2]:
                    try:
                        self.vs = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                        if self.vs.isOpened():
                            ret, frame = self.vs.read()
                            if ret and frame is not None:
                                camera_connected = True
                                break
                            else:
                                self.vs.release()
                                self.vs = None
                    except:
                        pass
                    
                    if not camera_connected:
                        try:
                            self.vs = cv2.VideoCapture(camera_index)
                            if self.vs.isOpened():
                                ret, frame = self.vs.read()
                                if ret and frame is not None:
                                    camera_connected = True
                                    break
                                else:
                                    self.vs.release()
                                    self.vs = None
                        except:
                            pass
            
            if camera_connected:
                # Reset webcam state
                st.session_state.webcam_stop_requested = False
                st.session_state.webcam_pause_requested = False
                st.session_state.webcam_initialized = True
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Error reconnecting to webcam: {str(e)}")
            return False

    def check_webcam_availability(self):
        """Check if webcam is available and accessible."""
        try:
            # Try multiple camera indices for Windows compatibility
            for camera_index in [0, 1, 2]:
                test_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use DirectShow on Windows
                if test_cap.isOpened():
                    # Try to read a frame to ensure it's working
                    ret, frame = test_cap.read()
                    test_cap.release()
                    if ret and frame is not None:
                        st.session_state.webcam_initialized = True
                        return True
                test_cap.release()
            
            # If DirectShow fails, try default backend
            test_cap = cv2.VideoCapture(0)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                test_cap.release()
                if ret and frame is not None:
                    st.session_state.webcam_initialized = True
                    return True
                test_cap.release()
            
            return False
        except Exception as e:
            st.error(f"Webcam check error: {str(e)}")
            return False
    
    def get_webcam_status(self):
        """Get detailed webcam status information."""
        status = {
            'connected': False,
            'camera_index': None,
            'resolution': None,
            'fps': None,
            'backend': None,
            'error': None
        }
        
        try:
            # Try to get webcam info
            if self.vs is not None and self.vs.isOpened():
                status['connected'] = True
                status['camera_index'] = 0  # Default assumption
                status['resolution'] = (
                    int(self.vs.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
                status['fps'] = self.vs.get(cv2.CAP_PROP_FPS)
                status['backend'] = "DirectShow" if hasattr(self.vs, 'getBackendName') else "Default"
            else:
                # Try to detect available cameras
                for camera_index in [0, 1, 2]:
                    test_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret and frame is not None:
                            status['connected'] = True
                            status['camera_index'] = camera_index
                            status['resolution'] = (frame.shape[1], frame.shape[0])
                            status['backend'] = "DirectShow"
                            test_cap.release()
                            break
                        test_cap.release()
                
                if not status['connected']:
                    # Try default backend
                    test_cap = cv2.VideoCapture(0)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret and frame is not None:
                            status['connected'] = True
                            status['camera_index'] = 0
                            status['resolution'] = (frame.shape[1], frame.shape[0])
                            status['backend'] = "Default"
                        test_cap.release()
                        
        except Exception as e:
            status['error'] = str(e)
            
        return status
    
    def start_counting(self, video_path, confidence, max_disappeared, max_distance, process_every_n_frames):
        """Start the people counting process."""
        # Check if model is available
        if not st.session_state.model_loaded or st.session_state.model is None:
            st.error("❌ YOLO model is not loaded. Please load the model first.")
            return
            
        try:
            # Initialize video capture
            if video_path == 0:  # Webcam
                # Try DirectShow first on Windows, then fallback
                try:
                    self.vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not self.vs.isOpened():
                        self.vs = cv2.VideoCapture(0)  # Fallback to default
                except:
                    self.vs = cv2.VideoCapture(0)  # Fallback to default
            else:
                self.vs = cv2.VideoCapture(video_path)
                
            if not self.vs.isOpened():
                st.error("❌ Failed to open video source!")
                return
            
            # Get video properties
            total_frames = int(self.vs.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.vs.get(cv2.CAP_PROP_FPS)
            
            if total_frames > 0:
                st.info(f"📹 Video loaded: {total_frames} frames, {fps:.1f} FPS")
            else:
                st.info("📹 Webcam mode - processing live stream")
            
            # Initialize tracker
            self.tracker = SimpleCentroidTracker(max_disappeared=max_disappeared, max_distance=max_distance)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Video display
            st.subheader("🎬 Live Processing")
            video_placeholder = st.empty()
            
            # Simple replay button during processing
            st.subheader("🎬 Processing Controls")
            col1, col2 = st.columns(2)
            
            with col1:
                replay_button = st.button("🔄 Start Replay", type="primary", key="live_replay")
            with col2:
                if video_path == 0:  # Only show stop for webcam
                    if st.button("⏹️ Stop Processing", type="secondary", key="live_stop"):
                        st.session_state.webcam_stop_requested = True
            
            # Quick replay controls if replay button is clicked
            if replay_button:
                st.subheader("🎬 Quick Replay")
                col1, col2, col3 = st.columns(3)
                with col1:
                    quick_replay_speed = st.slider("Speed", 0.5, 3.0, 1.0, 0.1, key="quick_replay_speed")
                with col2:
                    if st.button("▶️ Start Quick Replay", type="primary", key="quick_replay_start"):
                        self.quick_replay(confidence, max_disappeared, max_distance, process_every_n_frames, quick_replay_speed)
                with col3:
                    if st.button("🔄 Reset Replay", type="secondary", key="quick_replay_reset"):
                        self.reset_replay_counts()
                        st.rerun()
            
            # Initialize frame storage for replay
            st.session_state.replay_frames = []
            
            frame_count = 0
            consecutive_failures = 0  # Track consecutive frame read failures
            max_consecutive_failures = 10  # Maximum allowed consecutive failures
            
            while True:
                # Check for stop request
                if st.session_state.webcam_stop_requested:
                    st.warning("⏹️ Processing stopped by user")
                    break
                
                # Check for pause request
                if st.session_state.webcam_pause_requested:
                    st.info("⏸️ Processing paused. Click Pause again to resume.")
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.vs.read()
                if not ret:
                    consecutive_failures += 1
                    if video_path == 0:  # Webcam
                        if consecutive_failures >= max_consecutive_failures:
                            st.error(f"⚠️ Failed to read from webcam {consecutive_failures} times consecutively. Stopping.")
                            break
                        else:
                            st.warning(f"⚠️ Frame read failed ({consecutive_failures}/{max_consecutive_failures}). Retrying...")
                            time.sleep(0.1)
                            continue
                    else:
                        # For video files, stop on first failure
                        break
                else:
                    consecutive_failures = 0  # Reset failure counter on success
                
                # Resize frame for consistent processing
                frame = cv2.resize(frame, (600, 400))
                rectangles = []
                
                # Process every N frames for performance
                if frame_count % process_every_n_frames == 0:
                    try:
                        results = self.model.predict(frame, conf=confidence, verbose=False)
                        person_detections = results[0].boxes.data[results[0].boxes.cls == 0]
                        
                        for det in person_detections:
                            x1, y1, x2, y2 = map(int, det[:4])
                            rectangles.append((x1, y1, x2, y2))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    except Exception as e:
                        st.error(f"Detection error: {str(e)}")
                        break
                
                # Update tracker
                objects = self.tracker.update(rectangles)
                
                # Process tracked objects
                for (object_id, centroid) in objects.items():
                    to = self.trackable_objects.get(object_id, None)
                    if to is None:
                        to = SimpleTrackableObject(object_id, centroid)
                    else:
                        # Calculate direction
                        y_coords = [c[1] for c in to.centroids]
                        if len(y_coords) > 0:
                            direction = centroid[1] - np.mean(y_coords)
                            to.centroids.append(centroid)
                            
                            # Count crossing the line
                            if not to.counted:
                                if direction < 0 and centroid[1] < frame.shape[0] // 2:
                                    self.total_up += 1
                                    to.counted = True
                                elif direction > 0 and centroid[1] > frame.shape[0] // 2:
                                    self.total_down += 1
                                    to.counted = True
                    
                    self.trackable_objects[object_id] = to
                    
                    # Draw tracking info
                    cv2.putText(frame, f"ID {object_id}", (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                
                # Draw counting line
                cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), (0, 255, 255), 2)
                
                # Display counts on frame
                cv2.putText(frame, f"Up: {self.total_up} | Down: {self.total_down}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Store frame for replay (with detection data)
                frame_data = {
                    'frame': frame.copy(),
                    'detections': rectangles.copy(),
                    'frame_number': frame_count,
                    'timestamp': time.strftime("%H:%M:%S"),
                    'people_count': len(rectangles),
                    'people_in': self.total_up,
                    'people_out': self.total_down
                }
                st.session_state.replay_frames.append(frame_data)
                
                # Update display
                video_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                # Update progress
                frame_count += 1
                if total_frames > 0:
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames}")
                else:
                    status_text.text(f"Processing frame {frame_count}")
                
                # Add a small delay for webcam
                if video_path == 0:
                    time.sleep(0.1)
            
            # Cleanup
            if self.vs is not None:
                self.vs.release()
                self.vs = None
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            # Final results
            st.success(f"🎉 Counting Complete!")
            st.metric("Total People Moving Up", self.total_up)
            st.metric("Total People Moving Down", self.total_down)
            st.metric("Total Frames Processed", frame_count)
            
            # Show enhanced replay and download options after processing
            if st.session_state.replay_frames:
                st.subheader("🎬 Post-Processing Options")
                
                # Enhanced replay controls
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.subheader("🔄 Replay Options")
                    replay_speed = st.slider("Replay Speed", 0.5, 3.0, 1.0, 0.1, key="post_replay_speed")
                    if st.button("▶️ Start Replay", type="primary", key="post_replay_start"):
                        self.replay_video(confidence, max_disappeared, max_distance, process_every_n_frames, replay_speed)
                    if st.button("🔄 Reset Counts", type="secondary", key="post_replay_reset"):
                        self.reset_replay_counts()
                        st.rerun()
                
                with col2:
                    st.subheader("📥 Download Options")
                    if st.button("📹 Download Processed Video", type="secondary", key="post_download_video"):
                        self.download_processed_video()
                    if st.button("📊 Download Results CSV", type="secondary", key="post_download_csv"):
                        self.download_results_csv()
                
                with col3:
                    st.subheader("📊 Live Statistics")
                    st.metric("Current Up Count", self.total_up)
                    st.metric("Current Down Count", self.total_down)
                    st.metric("Total Frames", len(st.session_state.replay_frames))
                
                with col4:
                    st.subheader("🔧 Webcam Controls")
                    if st.button("🔄 Reconnect Webcam", type="secondary", key="post_reconnect"):
                        with st.spinner("Reconnecting webcam..."):
                            if self.reconnect_webcam():
                                st.success("✅ Webcam reconnected!")
                                st.rerun()
                            else:
                                st.error("❌ Reconnection failed")
                
                # Show replay history
                if st.session_state.replay_history:
                    st.subheader("📋 Replay History")
                    for replay in st.session_state.replay_history:
                        with st.expander(f"Replay #{replay['replay_number']} - {replay['timestamp']}"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Up", replay['up_count'])
                            with col2:
                                st.metric("Down", replay['down_count'])
                            with col3:
                                st.metric("Frames", replay['total_frames'])
                            with col4:
                                st.metric("Time", f"{replay['processing_time']:.2f}s")
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            if video_path == 0:
                st.error("Please check your webcam connection and permissions.")
            else:
                st.error("Please check your video file and try again.")
            # Ensure cleanup happens even on error
            if self.vs is not None:
                self.vs.release()
                self.vs = None

    def reset_replay_counts(self):
        """Reset replay counts while keeping frame data."""
        self.total_up = 0
        self.total_down = 0
        self.total_frames = 0
        self.trackable_objects = {}
        st.success("🔄 Replay counts reset successfully!")

    def quick_replay(self, confidence, max_disappeared, max_distance, process_every_n_frames, replay_speed=1.0):
        """Quick replay of the current frames with reset counts."""
        if not st.session_state.replay_frames:
            st.error("❌ No video frames available for replay. Please process a video first.")
            return
        
        if not st.session_state.model_loaded or st.session_state.model is None:
            st.error("❌ YOLO model is not loaded. Please load the model first.")
            return
        
        try:
            # Reset counts for new replay
            self.total_up = 0
            self.total_down = 0
            self.total_frames = 0
            self.trackable_objects = {}
            
            # Initialize tracker
            self.tracker = SimpleCentroidTracker(max_disappeared=max_disappeared, max_distance=max_distance)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Video display
            st.subheader(f"🔄 Quick Replay")
            video_placeholder = st.empty()
            
            # Quick replay controls - responsive layout
            st.subheader("🎮 Quick Replay Controls")
            
            # Primary controls row
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⏸️ Pause/Resume", type="secondary", key="quick_replay_pause", use_container_width=True):
                    st.session_state.replay_paused = not st.session_state.get('replay_paused', False)
                    st.rerun()
            with col2:
                if st.button("⏹️ Stop Replay", type="secondary", key="quick_replay_stop", use_container_width=True):
                    st.session_state.replay_stopped = True
                    st.rerun()
            
            # Secondary controls row
            col3, col4 = st.columns(2)
            with col3:
                if st.button("🔄 Reset Counts", type="secondary", key="quick_replay_reset", use_container_width=True):
                    self.reset_replay_counts()
                    st.rerun()
            with col4:
                if st.button("🔍 Maximize" if not st.session_state.get('video_maximized', False) else "📱 Minimize", 
                            type="secondary", key="quick_replay_maximize", use_container_width=True):
                    st.session_state.video_maximized = not st.session_state.get('video_maximized', False)
                    st.rerun()
            
            start_time = time.time()
            frame_count = 0
            total_frames = len(st.session_state.replay_frames)
            
            for frame_data in st.session_state.replay_frames:
                # Check for stop request
                if st.session_state.get('replay_stopped', False):
                    st.warning("⏹️ Quick replay stopped by user")
                    break
                
                # Check for pause request
                if st.session_state.get('replay_paused', False):
                    st.info("⏸️ Quick replay paused. Click Pause/Resume to continue.")
                    time.sleep(0.1)
                    continue
                frame = frame_data['frame']
                rectangles = frame_data['detections']
                
                # Process every N frames for performance
                if frame_count % process_every_n_frames == 0:
                    # Update tracker
                    objects = self.tracker.update(rectangles)
                    
                    # Process tracked objects
                    for (object_id, centroid) in objects.items():
                        to = self.trackable_objects.get(object_id, None)
                        if to is None:
                            to = SimpleTrackableObject(object_id, centroid)
                        else:
                            # Calculate direction
                            y_coords = [c[1] for c in to.centroids]
                            if len(y_coords) > 0:
                                direction = centroid[1] - np.mean(y_coords)
                                to.centroids.append(centroid)
                                
                                # Count crossing the line
                                if not to.counted:
                                    if direction < 0 and centroid[1] < frame.shape[0] // 2:
                                        self.total_up += 1
                                        to.counted = True
                                    elif direction > 0 and centroid[1] > frame.shape[0] // 2:
                                        self.total_down += 1
                                        to.counted = True
                        
                        self.trackable_objects[object_id] = to
                        
                        # Draw tracking info
                        cv2.putText(frame, f"ID {object_id}", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                
                # Draw counting line
                cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), (0, 255, 255), 2)
                
                # Display counts on frame
                cv2.putText(frame, f"Up: {self.total_up} | Down: {self.total_down}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Update display
                video_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Quick replay frame {frame_count}/{total_frames}")
                
                # Control replay speed
                time.sleep(1.0 / (30 * replay_speed))  # 30 FPS base speed
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Store replay history
            self.store_replay_history(self.total_up, self.total_down, frame_count, processing_time)
            
            # Final results
            progress_bar.progress(1.0)
            status_text.text("✅ Quick replay complete!")
            
            st.success(f"🔄 Quick Replay Complete!")
            st.metric("Total People Moving Up", self.total_up)
            st.metric("Total People Moving Down", self.total_down)
            st.metric("Total Frames Processed", frame_count)
            st.metric("Processing Time", f"{processing_time:.2f}s")
            
        except Exception as e:
            st.error(f"❌ Error during quick replay: {str(e)}")


def main():
    """Main application entry point."""
    app = PeopleCounter()
    app.setup_ui()


if __name__ == "__main__":
    main()
