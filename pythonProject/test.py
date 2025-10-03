import cv2
from ultralytics import YOLO
import math

# Function to predict using YOLO model
def predict(chosen_model, img, classes=[], conf=0.4):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

# Function to calculate the Euclidean distance
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to process detection and manage tracking
def predict_and_detect(chosen_model, img, human_class_id=0, suitcase_class_id=28, handbag_class_id=26, backpack_class_id=24, classes=[], conf=0.5,
                       rectangle_thickness=2, text_thickness=1, unattended_state=False,
                       yellow_box_tracker={'person_center': None, 'suitcase_center': None, 'handbag_center': None, 'backpack_center': None}):
    results = predict(chosen_model, img, classes, conf)

    person_centers = []
    suitcase_center = None
    suitcase_box = None
    handbag_center = None
    handbag_box = None
    backpack_center = None
    backpack_box = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            class_id = int(box.cls[0])
            label = result.names[class_id]

            if class_id == human_class_id:
                person_centers.append(((center_x, center_y), (x1, y1, x2, y2)))
                color = (255, 0, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, rectangle_thickness)
                cv2.putText(img, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, text_thickness)
                cv2.circle(img, (center_x, center_y), 3, color, -1)

            elif class_id == suitcase_class_id:
                suitcase_center = (center_x, center_y)
                suitcase_box = (x1, y1, x2, y2)
                cv2.circle(img, suitcase_center, 3, (0, 255, 255), -1)  # Yellow dot for suitcase center

            elif class_id == handbag_class_id:
                handbag_center = (center_x, center_y)
                handbag_box = (x1, y1, x2, y2)
                cv2.circle(img, handbag_center, 3, (255, 165, 0), -1)  # Orange dot for handbag center

            elif class_id == backpack_class_id:
                backpack_center = (center_x, center_y)
                backpack_box = (x1, y1, x2, y2)
                cv2.circle(img, backpack_center, 3, (0, 255, 0), -1)  # Green dot for backpack center

    def draw_yellow_box_if_paired(object_center, object_box, object_label, yellow_box_tracker_key):
        if object_center:
            min_distance = float('inf')
            closest_person_center = None
            closest_person_box = None

            for person_center, person_box in person_centers:
                distance = calculate_distance(person_center, object_center)
                if distance < min_distance:
                    min_distance = distance
                    closest_person_center = person_center
                    closest_person_box = person_box

            yellow_box_tracker['person_center'] = closest_person_center
            yellow_box_tracker[yellow_box_tracker_key] = object_center

            if closest_person_box:
                x1_combined = min(closest_person_box[0], object_box[0])
                y1_combined = min(closest_person_box[1], object_box[1])
                x2_combined = max(closest_person_box[2], object_box[2])
                y2_combined = max(closest_person_box[3], object_box[3])
                cv2.rectangle(img, (x1_combined, y1_combined), (x2_combined, y2_combined), (0, 255, 255), rectangle_thickness)
                cv2.putText(img, f"Person + {object_label}", (x1_combined, y1_combined - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), text_thickness)

    draw_yellow_box_if_paired(suitcase_center, suitcase_box, "Suitcase", 'suitcase_center')
    draw_yellow_box_if_paired(handbag_center, handbag_box, "Handbag", 'handbag_center')
    draw_yellow_box_if_paired(backpack_center, backpack_box, "Backpack", 'backpack_center')

    for object_box, object_label, tracker_key in [
        (suitcase_box, "Suitcase", 'suitcase_center'),
        (handbag_box, "Handbag", 'handbag_center'),
        (backpack_box, "Backpack", 'backpack_center'),
    ]:
        if object_box:
            if yellow_box_tracker['person_center'] is None or yellow_box_tracker[tracker_key] is None:
                unattended_state = True
            else:
                unattended_state = False

            x1, y1, x2, y2 = object_box
            object_color = (0, 0, 255) if unattended_state else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), object_color, rectangle_thickness)
            cv2.putText(img, "Unattended" if unattended_state else object_label, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, object_color, text_thickness)

        if suitcase_box:
            x1, y1, x2, y2 = suitcase_box
            if unattended_state:
                suitcase_color = (0, 0, 255)  # Red for unattended suitcase
                unattended_state = True
                # Popup message for unattended suitcase
                popup_message = "Unattended Suitcase!"
                font_scale = 2
                font_thickness = 3
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Get text size
                text_size = cv2.getTextSize(popup_message, font, font_scale, font_thickness)[0]
                text_x = (img.shape[1] - text_size[0]) // 2
                text_y = (img.shape[0] + text_size[1]) // 2

                # Draw text with background rectangle
                cv2.rectangle(img, (text_x - 10, text_y - text_size[1] - 10),
                              (text_x + text_size[0] + 10, text_y + 10), (0, 0, 255), -1)
                cv2.putText(img, popup_message, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
            else:
                suitcase_color = (0, 255, 0)  # Green for attended suitcase

    return img, unattended_state

# Function to create a VideoWriter object
def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))  # Match input video's FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Main function
if __name__ == "__main__":
    model = YOLO("yolo11x.pt")
    video_path = r"C:\Users\user\Videos\S21804.mp4"
    output_filename = r"C:\Users\user\OneDrive\Desktop\output.mp4"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    writer = create_video_writer(cap, output_filename)

    human_class_id = 0
    suitcase_class_id = 28
    handbag_class_id = 26
    backpack_class_id = 24
    classes_to_detect = [human_class_id, suitcase_class_id, handbag_class_id, backpack_class_id]

    unattended_state = False
    yellow_box_tracker = {'person_center': None, 'suitcase_center': None}

    while True:
        success, img = cap.read()
        if not success:
            print("End of video or error reading frame.")
            break

        if img is not None:
            img = cv2.resize(img, (800, 500))  # Resizing to a fixed size
            img, unattended_state = predict_and_detect(model, img, human_class_id, suitcase_class_id, handbag_class_id, backpack_class_id, classes_to_detect,
                                                       conf=0.4, unattended_state=unattended_state,
                                                       yellow_box_tracker=yellow_box_tracker)

            print(f"Writing frame: {img.shape}, dtype: {img.dtype}")
            writer.write(img)
            cv2.imshow("Detection and Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")
