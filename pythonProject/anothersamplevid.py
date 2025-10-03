import cv2
from ultralytics import YOLO
import time

# YOLO-based functions
def predict(chosen_model, img, classes=[], conf=0.4):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

# Tracker initialization
def initialize_tracker(tracker_type, frame, bbox):
    tracker = None
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")

    tracker.init(frame, bbox)
    return tracker

# Main function
if __name__ == "__main__":
    # Initialize YOLO model
    model = YOLO("yolo11x.pt")
    human_class_id = 0
    suitcase_class_id = 28
    classes_to_detect = [human_class_id, suitcase_class_id]

    # Video input and output
    video_path = r"C:\dev\litter_segmentation\movies\Blk 36-80m\DAY\Plastic of Refuse\Video\plastic.mp4"
    output_filename = r"C:\Users\user\OneDrive\Desktop\processed_video_with_tracking.mp4"
    cap = cv2.VideoCapture(video_path)

    writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 480))

    # Tracker setup
    tracker = None
    tracker_type = 'CSRT'  # Change to desired tracker type
    bbox = None  # Bounding box for tracking

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (640, 480))
        person_box = None
        suitcase_box = None

        if not tracker:
            # Perform YOLO detection
            results = predict(model, img, classes=classes_to_detect, conf=0.5)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    class_id = int(box.cls[0])

                    if class_id == human_class_id:
                        person_box = (x1, y1, x2, y2)
                        # Draw blue box for person
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(img, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
                        # Draw blue center dot
                        cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)
                    elif class_id == suitcase_class_id:
                        suitcase_box = (x1, y1, x2, y2)
                        # Draw green box for suitcase
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, "Suitcase", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                        # Draw green center dot
                        cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)

            if person_box and suitcase_box:
                # Calculate the yellow box (combined bounding box of person + suitcase)
                x1_combined = min(person_box[0], suitcase_box[0])
                y1_combined = min(person_box[1], suitcase_box[1])
                x2_combined = max(person_box[2], suitcase_box[2])
                y2_combined = max(person_box[3], suitcase_box[3])

                bbox = (x1_combined, y1_combined, x2_combined - x1_combined, y2_combined - y1_combined)
                tracker = initialize_tracker(tracker_type, img, bbox)

        if tracker:
            # Update tracker
            ok, bbox = tracker.update(img)
            if ok:
                # Draw yellow box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                center_x = (p1[0] + p2[0]) // 2
                center_y = (p1[1] + p2[1]) // 2
                cv2.rectangle(img, p1, p2, (0, 255, 255), 3, 1)
                cv2.putText(img, "Person + Suitcase", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                # Draw yellow center dot
                cv2.circle(img, (center_x, center_y), 5, (0, 255, 255), -1)
            else:
                tracker = None  # Re-initialize tracker if it fails
                cv2.putText(img, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Show FPS
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        writer.write(img)
        cv2.imshow("Detection and Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
