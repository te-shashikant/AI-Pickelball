import cv2
import mediapipe as mp
from ultralytics import YOLO

def process_video(input_path, output_path="output.mp4", mode="stroke"):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = fps / 6
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))

    feedbacks = set()
    current_score = 100
    flags = {'knee': False, 'arm': False, 'posture': False, 'stance': False,
             'object': False, 'back_knee': False, 'twist': False, 'toss': False, 'separation': False}

    accepted_frames = []
    rejected_frames = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = frame_index / fps
        reasons = []

        yolo_results = model.predict(source=frame, conf=0.4, verbose=False)[0]
        person_boxes = []
        object_detected = False

        for r in yolo_results.boxes.data.tolist():
            x1, y1, x2, y2 = map(int, r[:4])
            conf = r[4]
            cls = int(r[5])

            label = model.names[cls]
            if label == 'person':
                person_boxes.append((x1, y1, x2, y2))
                object_detected = True
            elif label in ['sports ball', 'paddle']:
                object_detected = True

        if not object_detected:
            reasons.append("No paddle/ball detected")
            if not flags['object']:
                feedbacks.add("Object not clearly visible (ball or paddle)")
                current_score -= 10
                flags['object'] = True

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for (x1, y1, x2, y2) in person_boxes:
            person_crop = image_rgb[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            result = pose.process(person_crop)
            if not result.pose_landmarks:
                continue

            mp_draw.draw_landmarks(
                frame[y1:y2, x1:x2],
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            lm = result.pose_landmarks.landmark

            def get(name):
                idx = mp_pose.PoseLandmark[name].value
                return {
                    "x": lm[idx].x * (x2 - x1) + x1,
                    "y": lm[idx].y * (y2 - y1) + y1
                }

            if mode == "stroke":
                knee_angle = abs(get('LEFT_HIP')['y'] - get('LEFT_KNEE')['y']) - abs(get('LEFT_KNEE')['y'] - get('LEFT_ANKLE')['y'])
                if knee_angle > 0.15:
                    reasons.append("Knee not bent enough")
                    if not flags['knee']:
                        feedbacks.add("Try bending your knees more")
                        current_score -= 25
                        flags['knee'] = True

                back_knee_angle = abs(get('RIGHT_HIP')['y'] - get('RIGHT_KNEE')['y']) - abs(get('RIGHT_KNEE')['y'] - get('RIGHT_ANKLE')['y'])
                if back_knee_angle > 0.15:
                    if not flags['back_knee']:
                        feedbacks.add("Bend your back knee slightly for balance")
                        current_score -= 10
                        flags['back_knee'] = True

                arm_angle = abs(get('RIGHT_SHOULDER')['y'] - get('RIGHT_ELBOW')['y']) + abs(get('RIGHT_ELBOW')['y'] - get('RIGHT_WRIST')['y'])
                if arm_angle < 0.2:
                    reasons.append("Arm not extended")
                    if not flags['arm']:
                        feedbacks.add("Try extending your arm fully")
                        current_score -= 25
                        flags['arm'] = True

                torso_angle = abs(((get('LEFT_SHOULDER')['y'] + get('RIGHT_SHOULDER')['y']) / 2) - ((get('LEFT_HIP')['y'] + get('RIGHT_HIP')['y']) / 2))
                if torso_angle < 0.05:
                    reasons.append("Posture too upright")
                    if not flags['posture']:
                        feedbacks.add("Try leaning forward")
                        current_score -= 25
                        flags['posture'] = True

                foot_distance = abs(get('LEFT_HEEL')['x'] - get('RIGHT_HEEL')['x'])
                if foot_distance < 0.06:
                    reasons.append("Stance too narrow")
                    if not flags['stance']:
                        feedbacks.add("Widen your stance")
                        current_score -= 25
                        flags['stance'] = True

            elif mode == "serve":
                shoulder = get('RIGHT_SHOULDER')
                wrist = get('RIGHT_WRIST')
                if wrist['y'] > shoulder['y']:
                    reasons.append("Serving arm too low")
                    if not flags['arm']:
                        feedbacks.add("Raise your serving arm higher")
                        current_score -= 25
                        flags['arm'] = True

                hip = get('RIGHT_HIP')
                if shoulder['y'] < hip['y']:
                    reasons.append("Body not leaning back")
                    if not flags['posture']:
                        feedbacks.add("Try leaning back slightly during the serve")
                        current_score -= 25
                        flags['posture'] = True

                foot_distance = abs(get('LEFT_HEEL')['x'] - get('RIGHT_HEEL')['x'])
                if foot_distance < 0.06:
                    reasons.append("Stance too narrow")
                    if not flags['stance']:
                        feedbacks.add("Widen your stance for a stable serve")
                        current_score -= 25
                        flags['stance'] = True

                elbow_angle = abs(get('RIGHT_ELBOW')['y'] - get('RIGHT_SHOULDER')['y']) + abs(get('RIGHT_ELBOW')['y'] - get('RIGHT_WRIST')['y'])
                if elbow_angle < 0.2:
                    reasons.append("Elbow not fully extended")
                    if not flags['knee']:
                        feedbacks.add("Extend your elbow fully during the toss")
                        current_score -= 25
                        flags['knee'] = True

        # Save accepted/rejected frames
        if len(accepted_frames) + len(rejected_frames) < 100 and frame_index % 3 == 0:
            if reasons:
                rejected_frames.append({"frame": frame.copy(), "reason": ", ".join(reasons), "timestamp": timestamp})
            else:
                accepted_frames.append({"frame": frame.copy(), "timestamp": timestamp})

        # 🔧 FIXED: Define score_text here so it's always available
        score_text = f"Score: {max(0, current_score)}"

        # Annotate this frame
        cv2.putText(frame, score_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 215, 0), 2, cv2.LINE_AA)

        if reasons:
            cv2.putText(frame, "Rejected", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Issues: " + ", ".join(reasons)[:100], (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Accepted", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()

    final_score = max(0, current_score)

    # Coupled holistic feedback
    overall_feedback = []
    if flags['knee'] and flags['posture']:
        overall_feedback.append("You need to improve your posture by bending your knees and leaning forward for better balance.")
    if flags['arm'] and flags['twist']:
        overall_feedback.append("Try extending your arm fully and adding torso rotation to generate more power.")
    if flags['stance'] and flags['back_knee']:
        overall_feedback.append("Widen your stance and bend your back knee to improve overall stability during your stroke.")
    if mode == "serve":
        if flags['toss'] and flags['arm']:
            overall_feedback.append("Your toss and arm position need improvement. Toss higher and raise your serving arm more for a stronger serve.")
        if flags['separation']:
            overall_feedback.append("Increase shoulder-hip separation to generate more momentum in your serve.")

    for item in overall_feedback:
        feedbacks.add("💡 " + item)

    def create_segments(frames, label):
        segments = []
        if not frames:
            return segments
        current_segment = {
            "start": frames[0]['timestamp'],
            "end": frames[0]['timestamp'],
            "frames": [frames[0]],
            "reasons": set(frames[0].get('reason', '').split(", ")) if label == "rejected" else set()
        }
        for i in range(1, len(frames)):
            current = frames[i]
            prev = frames[i - 1]
            if current['timestamp'] - prev['timestamp'] <= 0.5:
                current_segment["frames"].append(current)
                current_segment["end"] = current['timestamp']
                if label == "rejected":
                    current_segment["reasons"].update(current.get('reason', '').split(", "))
            else:
                segments.append(current_segment)
                current_segment = {
                    "start": current['timestamp'],
                    "end": current['timestamp'],
                    "frames": [current],
                    "reasons": set(current.get('reason', '').split(", ")) if label == "rejected" else set()
                }
        segments.append(current_segment)
        return segments

    accepted_segments = create_segments(accepted_frames, "accepted")
    rejected_segments = create_segments(rejected_frames, "rejected")

    return list(feedbacks), final_score, accepted_segments, rejected_segments
