import numpy as np
import cv2 as cv
import dlib
import torch
import torch.nn.functional as F
import os
import sys # Used for path manipulation


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)


try:
    from ..estimation.models.resnet import ResNet
except ImportError:
    print("Error importing ResNet model. Make sure the script is run correctly relative to the project root.")
    print(f"Current sys.path includes: {sys.path}")
    
    try:
         sys.path.insert(0, os.path.join(current_dir, "..", "..")) 
         from ..estimation.models.resnet import ResNet
         print("Successfully imported ResNet using fallback path.")
    except ImportError:
        print("Fallback import failed. Could not find ResNet model definition.")
        exit(1)


def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    Boxes are expected in (x, y, w, h) format.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6) # Add epsilon for stability

    return iou

# Helper Function: Estimate Age
def estimate_age_for_face(face_img, model, device):
    """
    Estimates the age for a given cropped face image using the loaded model.
    Returns the estimated age (int) or None if estimation fails.
    """
    if face_img is None or face_img.size == 0:
         print("Warning: Received empty image for age estimation.")
         return None
    try:
        # Ensure image has 3 channels
        if len(face_img.shape) < 3 or face_img.shape[2] != 3:
             print(f"Warning: Image for age estimation has unexpected shape: {face_img.shape}. Skipping.")
             return None

        # Resize to model's expected input size (200x200)
        face_resized = cv.resize(face_img, (200, 200))

        # Normalize to [0, 1], convert to float, permute to CHW, add batch dim
        face_tensor = torch.tensor(face_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        face_tensor = face_tensor.to(device) # Move tensor to the correct device

        with torch.no_grad():
            output = model(face_tensor)
            # Apply offset based on UTKFace labels (Age = index + 8)
            age = output.argmax(dim=1).item() + 8
            return age
    except Exception as e:
        print(f"Age estimation failed: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed error traceback
        return None

# --- Main Execution Block ---
if __name__ == '__main__':

    # --- Configuration ---
    CASCADE_PATH = os.path.join(root_dir, 'src', 'pretrained', 'haarcascade_frontalface_default.xml')
    MODEL_WEIGHTS_PATH = os.path.join(root_dir, 'src', 'estimation', 'weights', 'resnet_new', 'model_weights.pt')
    VIDEO_SOURCE = 0 # 0 for default webcam, or path to video file
    IOU_THRESHOLD = 0.3 # Minimum IoU for matching detection to track
    CONFIDENCE_THRESHOLD = 8.0 # dlib tracker confidence threshold to keep tracking
    FRAMES_TO_LOSE_TRACK = 10 # Number of consecutive frames without update before removing track
    DETECT_EVERY_N_FRAMES = 3 # Run face detection every N frames (improves performance)
    ESTIMATE_AGE_EVERY_N_FRAMES = 15 # Re-estimate age for tracked faces every N frames


    # --- Initialization ---
    print("Loading Haar Cascade classifier...")
    if not os.path.exists(CASCADE_PATH):
        print(f"Error: Cascade file not found at {CASCADE_PATH}")
        exit(1)
    face_classifier = cv.CascadeClassifier(CASCADE_PATH)

    print("Loading Age Estimation model...")
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights file not found at {MODEL_WEIGHTS_PATH}")
        exit(1)

    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResNet() # Assuming ResNet class definition is correct
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device) # Move model to the correct device
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        exit(1)


    print("Initializing video stream...")
    stream = cv.VideoCapture(VIDEO_SOURCE, cv.CAP_DSHOW) # CAP_DSHOW often works better on Windows
    if not stream.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        exit(1)

    FRAME_WIDTH = int(stream.get(cv.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"Stream opened: {FRAME_WIDTH}x{FRAME_HEIGHT}")

    # Tracking state variables
    tracked_faces = {} # {face_id: {'tracker': dlib_tracker, 'box': (x,y,w,h), 'age': int_or_None, 'frames_since_seen': 0, 'frames_since_age_est': 0}}
    next_face_id = 1
    frame_count = 0

    #Main Processing Loop 
    while stream.isOpened():
        ret, frame = stream.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # dlib tracker works better with RGB
        frame_count += 1

        # --- 1. Predict Locations & Update Existing Trackers ---
        active_track_ids = list(tracked_faces.keys())
        predicted_boxes = {}
        ids_to_remove = []

        for face_id in active_track_ids:
            tracker = tracked_faces[face_id]['tracker']
            tracking_quality = tracker.update(frame_rgb) # Use RGB frame for dlib

            if tracking_quality >= CONFIDENCE_THRESHOLD:
                tracked_pos = tracker.get_position()
                t_x = int(tracked_pos.left())
                t_y = int(tracked_pos.top())
                t_w = int(tracked_pos.width())
                t_h = int(tracked_pos.height())

                # Basic boundary check
                t_x = max(0, t_x)
                t_y = max(0, t_y)
                t_w = min(FRAME_WIDTH - t_x, t_w)
                t_h = min(FRAME_HEIGHT - t_y, t_h)

                if t_w > 0 and t_h > 0: # Ensure valid box after clipping
                     predicted_boxes[face_id] = (t_x, t_y, t_w, t_h)
                     tracked_faces[face_id]['box'] = (t_x, t_y, t_w, t_h) # Update box with tracker prediction
                     tracked_faces[face_id]['frames_since_seen'] = 0
                     tracked_faces[face_id]['frames_since_age_est'] += 1
                else: # Tracker likely drifted off screen
                     tracked_faces[face_id]['frames_since_seen'] += 1

            else: # Tracker confidence low
                tracked_faces[face_id]['frames_since_seen'] += 1

            # Check if tracker should be removed
            if tracked_faces[face_id]['frames_since_seen'] >= FRAMES_TO_LOSE_TRACK:
                 ids_to_remove.append(face_id)

        # --- Remove lost trackers ---
        for face_id in ids_to_remove:
            print(f"Removing lost track: ID {face_id}")
            if face_id in tracked_faces:
                 del tracked_faces[face_id]
            if face_id in predicted_boxes:
                 del predicted_boxes[face_id] # Ensure consistency


        # --- 2. Detect Faces (Periodically) ---
        detected_boxes = []
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            frame_gs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Adjust parameters as needed for your conditions
            detections = face_classifier.detectMultiScale(frame_gs, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))
            detected_boxes = [(x, y, w, h) for (x, y, w, h) in detections]


        # --- 3. Match Detections to Existing Tracks (if detections were made) ---
        matched_indices = {} # {detection_idx: track_id}
        unmatched_detections = list(range(len(detected_boxes)))

        if detected_boxes and predicted_boxes: # Only match if we have both detections and active tracks
            current_track_ids = list(predicted_boxes.keys())

            # Calculate IoU for all valid pairs
            iou_matrix = np.zeros((len(detected_boxes), len(current_track_ids)))
            for i, det_box in enumerate(detected_boxes):
                for j, track_id in enumerate(current_track_ids):
                    track_box = predicted_boxes[track_id]
                    iou_matrix[i, j] = calculate_iou(det_box, track_box)

            # Simple Greedy assignment based on highest IoU first
            assigned_detections = set()
            assigned_tracks = set()

            # Get indices sorted by IoU descending
            indices = np.argsort(iou_matrix.flatten())[::-1]
            det_indices, track_indices = np.unravel_index(indices, iou_matrix.shape)

            for det_idx, track_idx in zip(det_indices, track_indices):
                if iou_matrix[det_idx, track_idx] < IOU_THRESHOLD:
                    break # Stop if IoU is below threshold

                # Check if this detection or track has already been assigned
                if det_idx not in assigned_detections and track_idx not in assigned_tracks:
                    track_id = current_track_ids[track_idx]
                    matched_indices[det_idx] = track_id
                    assigned_detections.add(det_idx)
                    assigned_tracks.add(track_idx)

            # Update the list of unmatched detections
            unmatched_detections = [i for i in range(len(detected_boxes)) if i not in assigned_detections]


        # --- 4. Update Matched Trackers ---
        temp_ids_to_remove = [] # Trackers failing re-initialization here
        for det_idx, face_id in matched_indices.items():
            box = detected_boxes[det_idx]
            (x, y, w, h) = box
            tracked_faces[face_id]['box'] = box # Update box with more accurate detection

            # Re-initialize tracker on the detection for robustness
            try:
                 dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                 tracked_faces[face_id]['tracker'].start_track(frame_rgb, dlib_rect) # Use RGB
                 tracked_faces[face_id]['frames_since_seen'] = 0 # Reset loss counter
            except Exception as e:
                 print(f"Error re-initializing tracker for ID {face_id}: {e}")
                 temp_ids_to_remove.append(face_id) # Mark for removal if re-init fails
                 continue

            # Estimate age periodically or if not estimated yet
            if tracked_faces[face_id]['age'] is None or tracked_faces[face_id]['frames_since_age_est'] >= ESTIMATE_AGE_EVERY_N_FRAMES:
                # Use slightly padded crop, ensuring boundaries
                pad_x = int(w * 0.1) # Pad 10% of width/height
                pad_y = int(h * 0.15)
                crop_x1 = max(0, x - pad_x)
                crop_y1 = max(0, y - pad_y)
                crop_x2 = min(FRAME_WIDTH, x + w + pad_x)
                crop_y2 = min(FRAME_HEIGHT, y + h + pad_y)

                if crop_x2 > crop_x1 and crop_y2 > crop_y1 :
                    face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy() # Crop from original BGR frame
                    estimated_age = estimate_age_for_face(face_crop, model, device)
                    if estimated_age is not None:
                        # Optional: Apply smoothing (e.g., Exponential Moving Average)
                        # current_age = tracked_faces[face_id]['age']
                        # alpha = 0.3 # Smoothing factor
                        # if current_age is not None:
                        #     estimated_age = int(alpha * estimated_age + (1 - alpha) * current_age)
                        tracked_faces[face_id]['age'] = estimated_age
                        tracked_faces[face_id]['frames_since_age_est'] = 0 # Reset age estimation counter
                else:
                    print(f"Warning: Invalid crop dimensions for face ID {face_id} - Skipping age estimation.")

        # Remove trackers that failed re-initialization
        for face_id in temp_ids_to_remove:
            print(f"Removing track due to re-initialization failure: ID {face_id}")
            if face_id in tracked_faces:
                 del tracked_faces[face_id]


        # --- 5. Handle Unmatched Detections (Create New Trackers) ---
        for det_idx in unmatched_detections:
            box = detected_boxes[det_idx]
            (x, y, w, h) = box

            # Filter out very small detections
            if w < 30 or h < 30:
                continue

            new_tracker = dlib.correlation_tracker()
            try:
                 dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                 new_tracker.start_track(frame_rgb, dlib_rect) # Use RGB frame
                 face_id = next_face_id
                 next_face_id += 1

                 tracked_faces[face_id] = {
                     'tracker': new_tracker,
                     'box': box,
                     'age': None,
                     'frames_since_seen': 0,
                     'frames_since_age_est': 0
                 }
                 print(f"Started tracking new face: ID {face_id}")

                 # Immediately estimate age for the new face
                 pad_x = int(w * 0.1)
                 pad_y = int(h * 0.15)
                 crop_x1 = max(0, x - pad_x)
                 crop_y1 = max(0, y - pad_y)
                 crop_x2 = min(FRAME_WIDTH, x + w + pad_x)
                 crop_y2 = min(FRAME_HEIGHT, y + h + pad_y)

                 if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                     face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy() # Crop from original BGR frame
                     estimated_age = estimate_age_for_face(face_crop, model, device)
                     if estimated_age is not None:
                        tracked_faces[face_id]['age'] = estimated_age
                 else:
                    print(f"Warning: Invalid crop dimensions for new face ID {face_id} - Skipping age estimation.")

            except Exception as e:
                 print(f"Error initializing tracker for new detection at [{x},{y},{w},{h}]: {e}")


        #Draw Bounding Boxes and Info on Output Frame
        output_frame = frame.copy() # Draw on a copy
        for face_id, data in tracked_faces.items():
            box = data['box']
            age = data['age']
            (x, y, w, h) = [int(v) for v in box] # Ensure integer coordinates

            # Draw bounding box (Green for active tracks)
            cv.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Prepare label text
            label = f"ID: {face_id}"
            if age is not None:
                label += f" | Age: {age}"
            else:
                 label += " | Age: ?" # Indicate age is unknown or pending

            # Draw label background
            (label_width, label_height), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv.rectangle(output_frame, (x, y - label_height - 10), (x + label_width, y - 10), (0, 255, 0), cv.FILLED)
            # Draw label text
            cv.putText(output_frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


        #Display
        cv.imshow("AgeNet - Multi-Face Tracking", output_frame)

        #Quit Condition
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    #Cleanup
    print("Releasing video stream and closing windows.")
    stream.release()
    cv.destroyAllWindows()