import numpy as np
import cv2 as cv
import torch
import os
import sys
import time
from collections import OrderedDict


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from src.estimation.models.resnet import ResNet
except ImportError:
    print("Error importing ResNet model. Attempting fallback path...")
    fallback_path = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if fallback_path not in sys.path:
        sys.path.insert(0, fallback_path)
    try:
         from src.estimation.models.resnet import ResNet
         print("Successfully imported ResNet using fallback path.")
    except ImportError as e:
        print(f"Fallback import failed: {e}")
        exit(1)


def calculate_iou(boxA, boxB):
    if not (isinstance(boxA, (list, tuple)) and len(boxA) == 4 and
            isinstance(boxB, (list, tuple)) and len(boxB) == 4): return 0.0
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2]) * max(0, boxA[3])
    boxBArea = max(0, boxB[2]) * max(0, boxB[3])
    denominator = float(boxAArea + boxBArea - interArea)
    iou = interArea / (denominator + 1e-6) if denominator > 1e-6 else 0.0
    return iou

def estimate_age_for_face(face_img, model, device):
    if face_img is None or face_img.size < 100: return None
    try:
        if len(face_img.shape) != 3 or face_img.shape[2] != 3: return None
        if face_img.shape[0] < 10 or face_img.shape[1] < 10: return None
        face_resized = cv.resize(face_img, (200, 200), interpolation=cv.INTER_AREA)
        face_tensor = torch.tensor(face_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        face_tensor = face_tensor.to(device)
        with torch.no_grad():
            output = model(face_tensor)
            if output is None or not hasattr(output, 'argmax'): return None
            age = output.argmax(dim=1).item() + 8
            return age
    except cv.error: return None
    except Exception: return None

if __name__ == '__main__':

    PRETRAINED_DIR = os.path.join(root_dir, "src", "pretrained")
    YUNET_MODEL_PATH = os.path.join(PRETRAINED_DIR, "face_detection_yunet_2023mar.onnx")
    MODEL_WEIGHTS_PATH_AGE = os.path.join(root_dir, 'src', 'estimation', 'weights', 'resnet_new', 'model_weights.pt')
    VIDEO_SOURCE = 0

    # Detection & Tracking Tuning
    YUNET_CONFIDENCE_THRESHOLD = 0.8 # YuNet Threshold
    IOU_MATCHING_THRESHOLD = 0.4     # Min IoU to match detection to existing track 
    FRAMES_TO_LOSE_TRACK = 7         # How many frames until an unmatched track is dropped

    # Age Estimation Tuning
    ESTIMATE_AGE_EVERY_N_UPDATES = 10
    AGE_CROP_PADDING_X = 0.15
    AGE_CROP_PADDING_Y = 0.20

    print("Loading YuNet Face Detector model...")
    if not os.path.exists(YUNET_MODEL_PATH):
        print(f"Error: YuNet model file not found at '{YUNET_MODEL_PATH}'")
        print("Please download the '.onnx' file from OpenCV Zoo and place it in the 'src/pretrained/' directory.")
        exit(1)

    # Get frame dimensions first to pass to YuNet creator
    print("Initializing video stream to get dimensions...")
    stream_init = cv.VideoCapture(VIDEO_SOURCE, cv.CAP_DSHOW)
    if not stream_init.isOpened(): print(f"Error: Could not open video source {VIDEO_SOURCE}"); exit(1)
    ret_init, frame_init = stream_init.read()
    if not ret_init or frame_init is None: print("Error: Could not read initial frame."); stream_init.release(); exit(1)
    FRAME_HEIGHT, FRAME_WIDTH = frame_init.shape[:2]
    stream_init.release() # Release after getting dimensions
    print(f"Detected Frame Size: {FRAME_WIDTH}x{FRAME_HEIGHT}")

    try:
        # Create YuNet detector instance
        face_detector = cv.FaceDetectorYN.create(
            model=YUNET_MODEL_PATH,
            config="", 
            input_size=(FRAME_WIDTH, FRAME_HEIGHT), 
            score_threshold=YUNET_CONFIDENCE_THRESHOLD, # Initial confidence threshold
            nms_threshold=0.3, # YuNet's internal NMS threshold
            top_k=5000 
        )
        print("YuNet Face Detector loaded successfully.")
    except cv.error as e:
        print(f"Error creating YuNet detector: {e}")
        exit(1)
    except Exception as e:
         print(f"An unexpected error occurred during YuNet initialization: {e}")
         exit(1)


    print("Loading Age Estimation model...")
    if not os.path.exists(MODEL_WEIGHTS_PATH_AGE): print(f"Error: Age Model weights not found at {MODEL_WEIGHTS_PATH_AGE}"); exit(1)
    device_age = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device for Age Estimation: {device_age}")
    model_age = ResNet()
    try: model_age.load_state_dict(torch.load(MODEL_WEIGHTS_PATH_AGE, map_location=device_age)); model_age.to(device_age); model_age.eval(); print("Age Estimation Model loaded successfully.")
    except Exception as e: print(f"Error loading age model state_dict: {e}"); exit(1)


    print("Initializing video stream (main)...")
    # Re-open the stream for processing
    stream = cv.VideoCapture(VIDEO_SOURCE, cv.CAP_DSHOW)
    if not stream.isOpened(): print(f"Error: Could not open video source {VIDEO_SOURCE} for main loop"); exit(1)
    # Verify dimensions again just in case
    fw_check = int(stream.get(cv.CAP_PROP_FRAME_WIDTH)); fh_check = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    if fw_check != FRAME_WIDTH or fh_check != FRAME_HEIGHT: print(f"Warning: Frame dimensions changed between init ({FRAME_WIDTH}x{FRAME_HEIGHT}) and main loop ({fw_check}x{fh_check}).")
    print(f"Stream opened: {FRAME_WIDTH}x{FRAME_HEIGHT}")

    active_tracks = OrderedDict()
    next_track_id = 1
    frame_count = 0
    start_time = time.time()

    while stream.isOpened():
        ret, frame = stream.read()
        if not ret: print("End of stream or read error."); break

        frame_count += 1
        (h, w) = frame.shape[:2] 

        #Detect Faces using YuNet (Every Frame)
        current_detections = [] # Store as list of tuples (x, y, w, h)
        try:
            # This might be needed if camera source isn't perfectly stable
            if h != FRAME_HEIGHT or w != FRAME_WIDTH:
                 face_detector.setInputSize((w, h))
                 FRAME_WIDTH, FRAME_HEIGHT = w, h 
           
            retval, faces = face_detector.detect(frame)

            if faces is not None:
                for face in faces:
                    # Extract bounding box and confidence score
                    box = face[0:4].astype(np.int32)
                    confidence = face[-1] # Confidence is the last element

                    x, y, box_w, box_h = box
                    x, y = max(0, x), max(0, y)
                    box_w, box_h = min(w - x, box_w), min(h - y, box_h) # Clamp width/height too
                    if box_w > 0 and box_h > 0: # Ensure valid dimensions after clamping
                        current_detections.append((x, y, box_w, box_h))

        except cv.error as cv_err: print(f"OpenCV error during YuNet detection: {cv_err}")
        except Exception as e: print(f"Error during YuNet detection: {e}")


        #Match Current Detections to Active Tracks
        active_track_ids = list(active_tracks.keys())
        matched_detection_indices = set()
        matched_track_ids = set()
        potential_matches = []
        previous_boxes = {track_id: active_tracks[track_id]['box'] for track_id in active_track_ids}

        if current_detections and previous_boxes:
            for i, det_box in enumerate(current_detections):
                for track_id, prev_box in previous_boxes.items():
                    iou = calculate_iou(det_box, prev_box)
                    if iou >= IOU_MATCHING_THRESHOLD: # Now 0.4
                        potential_matches.append((iou, i, track_id))
            potential_matches.sort(key=lambda x: x[0], reverse=True)
            for iou, det_idx, track_id in potential_matches:
                if det_idx not in matched_detection_indices and track_id not in matched_track_ids:
                    #Update Matched Track
                    if track_id in active_tracks:
                         active_tracks[track_id]['box'] = current_detections[det_idx]
                         active_tracks[track_id]['frames_unseen'] = 0
                         active_tracks[track_id]['updates_since_age_est'] += 1
                         matched_detection_indices.add(det_idx)
                         matched_track_ids.add(track_id)
                         #Estimate Age Periodically
                         if active_tracks[track_id]['age'] is None or \
                            active_tracks[track_id]['updates_since_age_est'] >= ESTIMATE_AGE_EVERY_N_UPDATES:
                             (x, y, w_box, h_box) = active_tracks[track_id]['box']
                             pad_w = int(w_box * AGE_CROP_PADDING_X); pad_h = int(h_box * AGE_CROP_PADDING_Y)
                             crop_x1, crop_y1 = max(0, x - pad_w), max(0, y - pad_h)
                             crop_x2, crop_y2 = min(w, x + w_box + pad_w), min(h, y + h_box + pad_h)
                             if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                                 face_crop_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                                 estimated_age = estimate_age_for_face(face_crop_bgr, model_age, device_age)
                                 if estimated_age is not None:
                                     active_tracks[track_id]['age'] = estimated_age
                                     active_tracks[track_id]['updates_since_age_est'] = 0


        #Handle Unmatched Tracks and Detections
        tracks_to_remove = []
        for track_id in active_track_ids:
            if track_id not in matched_track_ids:
                active_tracks[track_id]['frames_unseen'] += 1
                if active_tracks[track_id]['frames_unseen'] >= FRAMES_TO_LOSE_TRACK:
                    tracks_to_remove.append(track_id)
        for track_id in tracks_to_remove:
            # print(f"Removing lost track: ID {track_id}") # Less verbose
            if track_id in active_tracks: del active_tracks[track_id]

        for det_idx, box in enumerate(current_detections):
            if det_idx not in matched_detection_indices:
                track_id = next_track_id; next_track_id += 1
                active_tracks[track_id] = {'box': box, 'age': None, 'frames_unseen': 0, 'updates_since_age_est': ESTIMATE_AGE_EVERY_N_UPDATES}
                # print(f"Started new track: ID {track_id}") # Less verbose
                (x, y, w_box, h_box) = box
                pad_w = int(w_box * AGE_CROP_PADDING_X); pad_h = int(h_box * AGE_CROP_PADDING_Y)
                crop_x1, crop_y1 = max(0, x - pad_w), max(0, y - pad_h)
                crop_x2, crop_y2 = min(w, x + w_box + pad_w), min(h, y + h_box + pad_h)
                if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                    face_crop_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    estimated_age = estimate_age_for_face(face_crop_bgr, model_age, device_age)
                    if estimated_age is not None: active_tracks[track_id]['age'] = estimated_age; active_tracks[track_id]['updates_since_age_est'] = 0


       # Frames per second
        end_time = time.time(); elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

       #Bounding Boxes
        output_frame = frame.copy()
        for track_id, data in active_tracks.items():
            try:
                box = data.get('box'); age = data.get('age')
                if box is None: continue
                (x, y, w_box, h_box) = [int(v) for v in box]
                if w_box <= 0 or h_box <= 0: continue
            except (TypeError, ValueError): continue
            cv.rectangle(output_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            label = f"ID: {track_id}" + (f" | Age: {age}" if age is not None else " | Age: ?")
            (lW, lH), base = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            bgY1 = max(y - lH - 10, 0); bgY2 = max(y - 10, lH)
            cv.rectangle(output_frame, (x, bgY1), (x + lW + 4, bgY2), (0, 255, 0), cv.FILLED)
            txtY = max(y - 10 - (base // 2), lH // 2)
            cv.putText(output_frame, label, (x + 2, txtY), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv.putText(output_frame, f"FPS: {fps:.2f}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv.imshow("YuNet Tracking-by-Detection & Age Estimation", output_frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'): print("Exiting..."); break

    print("Releasing video stream and closing windows.")
    stream.release()
    cv.destroyAllWindows()