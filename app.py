import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
import matplotlib.pyplot as plt
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define landmark indices based on the MediaPipe Pose diagram
# Using FOOT_INDEX for ankle angle as requested
LEFT_KNEE_INDEX = mp_pose.PoseLandmark.LEFT_KNEE.value # 25
LEFT_ANKLE_INDEX = mp_pose.PoseLandmark.LEFT_ANKLE.value # 27
LEFT_FOOT_INDEX_INDEX = mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value # 31

RIGHT_KNEE_INDEX = mp_pose.PoseLandmark.RIGHT_KNEE.value # 26
RIGHT_ANKLE_INDEX = mp_pose.PoseLandmark.RIGHT_ANKLE.value # 28
RIGHT_FOOT_INDEX_INDEX = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value # 32

def calculate_angle(a, b, c):
    """Calculates the angle at vertex b between points a, b, and c."""
    a = np.array(a) # First point (e.g., Foot Index)
    b = np.array(b) # Mid point/Vertex (e.g., Ankle)
    c = np.array(c) # End point (e.g., Knee)

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate dot product
    dot_product = np.dot(ba, bc)

    # Calculate magnitudes
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # Check for zero magnitude vectors to avoid division by zero
    if magnitude_ba == 0 or magnitude_bc == 0:
        return np.nan # Return NaN if calculation is not possible

    # Calculate cosine of the angle
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

    # Clip the value to [-1, 1] to avoid floating point errors outside the domain of arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate angle in radians and then convert to degrees
    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)

    return angle_degrees


st.title("Ankle Angle Detection in Video")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])


if uploaded_file is not None:
    # Use tempfile for handling uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1])
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.session_state.input_file_cleaned = False # Reset cleanup flag

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
    else:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Try to get fps, default to 30 if unavailable
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            st.warning("Could not determine video FPS. Defaulting to 30 FPS for processing and output.")
            fps = 30

        # Try to get total frames, handle potential errors
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            st.warning("Could not determine the total number of frames. Progress bar may be inaccurate.")
            total_frames = -1 # Indicate unknown total frames

        # Setup MediaPipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

            processed_frames = []
            # Store angles calculated *per frame*
            all_left_ankle_angles = []
            all_right_ankle_angles = []
            timestamps = []
            frame_count = 0

            st.info("Processing video... This might take a while depending on the video length.")
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # End of video

                # Recolor image to RGB for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False # Performance optimization

                # Make detection
                results = pose.process(image_rgb)

                # Recolor back to BGR for OpenCV drawing
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                image_bgr.flags.writeable = True

                current_left_angle = np.nan
                current_right_angle = np.nan

                # Extract landmarks if detected
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # --- Left Ankle ---
                    try:
                        # Get landmarks: Foot Index (a), Ankle (b), Knee (c)
                        left_foot_index_lm = landmarks[LEFT_FOOT_INDEX_INDEX]
                        left_ankle_lm = landmarks[LEFT_ANKLE_INDEX]
                        left_knee_lm = landmarks[LEFT_KNEE_INDEX]

                        # Calculate pixel coordinates
                        left_foot_index_px = tuple(np.multiply([left_foot_index_lm.x, left_foot_index_lm.y], [frame_width, frame_height]).astype(int))
                        left_ankle_px = tuple(np.multiply([left_ankle_lm.x, left_ankle_lm.y], [frame_width, frame_height]).astype(int))
                        left_knee_px = tuple(np.multiply([left_knee_lm.x, left_knee_lm.y], [frame_width, frame_height]).astype(int))

                        # Calculate angle using normalized coordinates (FootIndex-Ankle-Knee)
                        current_left_angle = calculate_angle(
                            [left_foot_index_lm.x, left_foot_index_lm.y],
                            [left_ankle_lm.x, left_ankle_lm.y],
                            [left_knee_lm.x, left_knee_lm.y]
                        )

                        # Draw lines for the left angle (Blue)
                        cv2.line(image_bgr, left_foot_index_px, left_ankle_px, (255, 0, 0), 2) # Blue
                        cv2.line(image_bgr, left_ankle_px, left_knee_px, (255, 0, 0), 2) # Blue
                        # Draw circles at joints
                        cv2.circle(image_bgr, left_foot_index_px, 3, (255, 0, 0), -1)
                        cv2.circle(image_bgr, left_ankle_px, 5, (255, 0, 0), -1) # Ankle joint larger
                        cv2.circle(image_bgr, left_knee_px, 3, (255, 0, 0), -1)

                        # Visualize left angle value (white text for contrast)
                        cv2.putText(image_bgr, f"L: {int(current_left_angle)}",
                                    (left_ankle_px[0] + 10, left_ankle_px[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                    except IndexError: # Handle cases where landmarks might be out of bounds
                         st.warning(f"Frame {frame_count}: Could not find all required left landmarks.")
                         pass
                    except Exception as e:
                         # st.warning(f"Frame {frame_count}: Error processing left ankle: {e}")
                         pass # Keep angle as NaN

                    # --- Right Ankle ---
                    try:
                        # Get landmarks: Foot Index (a), Ankle (b), Knee (c)
                        right_foot_index_lm = landmarks[RIGHT_FOOT_INDEX_INDEX]
                        right_ankle_lm = landmarks[RIGHT_ANKLE_INDEX]
                        right_knee_lm = landmarks[RIGHT_KNEE_INDEX]

                        # Calculate pixel coordinates
                        right_foot_index_px = tuple(np.multiply([right_foot_index_lm.x, right_foot_index_lm.y], [frame_width, frame_height]).astype(int))
                        right_ankle_px = tuple(np.multiply([right_ankle_lm.x, right_ankle_lm.y], [frame_width, frame_height]).astype(int))
                        right_knee_px = tuple(np.multiply([right_knee_lm.x, right_knee_lm.y], [frame_width, frame_height]).astype(int))

                        # Calculate angle using normalized coordinates (FootIndex-Ankle-Knee)
                        current_right_angle = calculate_angle(
                            [right_foot_index_lm.x, right_foot_index_lm.y],
                            [right_ankle_lm.x, right_ankle_lm.y],
                            [right_knee_lm.x, right_knee_lm.y]
                        )

                        # Draw lines for the right angle (Red)
                        cv2.line(image_bgr, right_foot_index_px, right_ankle_px, (0, 0, 255), 2) # Red (BGR)
                        cv2.line(image_bgr, right_ankle_px, right_knee_px, (0, 0, 255), 2) # Red (BGR)
                        # Draw circles at joints
                        cv2.circle(image_bgr, right_foot_index_px, 3, (0, 0, 255), -1) # Red (BGR)
                        cv2.circle(image_bgr, right_ankle_px, 5, (0, 0, 255), -1) # Ankle joint larger - Red (BGR)
                        cv2.circle(image_bgr, right_knee_px, 3, (0, 0, 255), -1) # Red (BGR)

                        # Visualize right angle value (white text)
                        cv2.putText(image_bgr, f"R: {int(current_right_angle)}",
                                    (right_ankle_px[0] + 10, right_ankle_px[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                    except IndexError:
                         st.warning(f"Frame {frame_count}: Could not find all required right landmarks.")
                         pass
                    except Exception as e:
                         # st.warning(f"Frame {frame_count}: Error processing right ankle: {e}")
                         pass # Keep angle as NaN

                # Append data for this frame
                processed_frames.append(image_bgr)
                all_left_ankle_angles.append(current_left_angle) # Append angle for this frame
                all_right_ankle_angles.append(current_right_angle) # Append angle for this frame
                timestamps.append(frame_count / fps)
                frame_count += 1

                # Update progress (handle unknown total frames)
                if total_frames > 0:
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames}...")
                else:
                    progress_bar.progress(0) # Or update based on elapsed time if desired
                    status_text.text(f"Processing frame {frame_count}...")


            cap.release()
            processing_time = time.time() - start_time
            status_text.text(f"Processing complete for {frame_count} frames in {processing_time:.2f} seconds.")
            st.success("Video processing finished!")

            if not processed_frames:
                 st.warning("No frames were processed. The input video might be empty or corrupted.")
                 # Clean up input temp file even if processing failed
                 try:
                     if os.path.exists(video_path) and not st.session_state.get('input_file_cleaned', False):
                         os.remove(video_path)
                         st.session_state.input_file_cleaned = True
                 except Exception as e:
                     st.warning(f"Could not remove temporary input file {video_path}: {e}")
            else:
                # --- Store results in session state for navigation ---
                st.session_state.processed_frames = processed_frames
                st.session_state.timestamps = timestamps
                st.session_state.left_ankle_angles = all_left_ankle_angles # Store list of angles
                st.session_state.right_ankle_angles = all_right_ankle_angles # Store list of angles
                st.session_state.total_processed_frames = len(processed_frames)

                # Initialize frame index if not already set or if a new video is processed
                if 'frame_index' not in st.session_state or st.session_state.get('processed_video_id') != uploaded_file.file_id:
                    st.session_state.frame_index = 0
                    st.session_state.processed_video_id = uploaded_file.file_id # Track the processed file

                # --- Frame Navigation ---
                st.subheader("Frame-by-Frame Navigation")

                # Ensure frame_index is valid after potential re-runs
                if st.session_state.frame_index >= st.session_state.total_processed_frames:
                     st.session_state.frame_index = st.session_state.total_processed_frames - 1
                if st.session_state.frame_index < 0:
                     st.session_state.frame_index = 0

                # Navigation buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("◀️ Previous Frame"):
                        if st.session_state.frame_index > 0:
                            st.session_state.frame_index -= 1
                        else:
                            st.warning("Already at the first frame.")
                with col2:
                     if st.button("Next Frame ▶️"):
                         if st.session_state.frame_index < st.session_state.total_processed_frames - 1:
                             st.session_state.frame_index += 1
                         else:
                            st.warning("Already at the last frame.")

                # Display current frame
                current_frame_index = st.session_state.frame_index
                current_frame_image = st.session_state.processed_frames[current_frame_index]
                # Convert BGR (OpenCV) to RGB (Streamlit)
                st.image(cv2.cvtColor(current_frame_image, cv2.COLOR_BGR2RGB), caption=f"Frame {current_frame_index + 1}/{st.session_state.total_processed_frames}", use_container_width=True)

                # Display frame information (retrieve from list using current_frame_index)
                current_time = st.session_state.timestamps[current_frame_index]
                current_left_angle = st.session_state.left_ankle_angles[current_frame_index]
                current_right_angle = st.session_state.right_ankle_angles[current_frame_index]

                left_angle_str = f"{current_left_angle:.1f}°" if not np.isnan(current_left_angle) else "N/A"
                right_angle_str = f"{current_right_angle:.1f}°" if not np.isnan(current_right_angle) else "N/A"

                st.write(f"**Timestamp:** {current_time:.2f}s")
                st.write(f"**Left Ankle Angle:** {left_angle_str}")
                st.write(f"**Right Ankle Angle:** {right_angle_str}")


                # --- Angle Graph ---
                st.subheader("Ankle Angle Over Time")
                fig, ax = plt.subplots(figsize=(10, 5)) # Make graph wider

                # Get data from session state
                timestamps_np = np.array(st.session_state.timestamps)
                left_angles_np = np.array(st.session_state.left_ankle_angles)
                right_angles_np = np.array(st.session_state.right_ankle_angles)

                # Plot only non-NaN values
                ax.plot(timestamps_np[~np.isnan(left_angles_np)], left_angles_np[~np.isnan(left_angles_np)], label='Left Ankle Angle', color='blue', marker='.', linestyle='-', markersize=2)
                ax.plot(timestamps_np[~np.isnan(right_angles_np)], right_angles_np[~np.isnan(right_angles_np)], label='Right Ankle Angle', color='red', marker='.', linestyle='-', markersize=2)

                # Highlight current frame on the graph
                ax.axvline(x=current_time, color='green', linestyle='--', label=f'Current Frame ({current_time:.2f}s)')

                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Angle (degrees)")
                ax.set_title("Ankle Angles During Video")
                ax.legend()
                ax.grid(True)
                ax.set_ylim(bottom=max(0, np.nanmin(np.concatenate((left_angles_np, right_angles_np))) - 10), 
                           top=min(180, np.nanmax(np.concatenate((left_angles_np, right_angles_np))) + 10)) # Dynamic Y axis or fallback
                st.pyplot(fig)

                # Clean up the initial temporary input video file
                # Moved cleanup here to ensure it happens after all processing/display setup
                try:
                    if os.path.exists(video_path) and not st.session_state.get('input_file_cleaned', False): # Check if it exists before removing and flag
                        os.remove(video_path)
                        st.session_state.input_file_cleaned = True
                except Exception as e:
                     st.warning(f"Could not remove temporary input file {video_path}: {e}")


else:
    # Clear session state if no file is uploaded or a new page load occurs without upload
    if 'processed_frames' in st.session_state:
        del st.session_state.processed_frames
    if 'frame_index' in st.session_state:
        del st.session_state.frame_index
    if 'processed_video_id' in st.session_state:
        del st.session_state.processed_video_id
    # ... clear other related states if necessary ...

    st.info("Upload a video file (MP4, MOV, AVI, MKV) to begin processing.")

st.markdown("---")
st.markdown("Built with [Streamlit](https://streamlit.io), [MediaPipe](https://developers.google.com/mediapipe), and [OpenCV](https://opencv.org/).")
