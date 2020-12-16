import sys
import cv2
import os
import numpy as np
sys.path.insert(0, "/Users/noorahmed/Desktop/Work/yolov5")

from detect import get_model, detect_single

from AudienceDetector import AudienceDetector

def has_face(results, face_label = 1.0, percentage_thresh = 1):
    for res in results:
        if face_label == res.cls:
            return True, res.percentage_of_screen > percentage_thresh
    return False, False

def write_to_folder(base_dataset, folder_name, frame, counts):
    cv2.imwrite(os.path.join(base_dataset, folder_name, str(counts[folder_name]).zfill(5) + ".jpg"), frame)
    counts[folder_name] += 1

def main(video_path):
    # Run the audience detector first

    # Second level will be with person detections

    # Each frame is weighted according to 3 metrics:
        # If it only has audience detection it is added to the the ad folder
        # If it has person detections with faces around 0.1 percent of the screen size
        # else if it has detected audience and also isnt included in the second cat it is added to the common set of frames

        # Folders:
            # ad (only bodies without faces)
            # critical (bodies with faces greater than 0.1 percent)
            # common (bodies with faces but less than 0.1 percent)

    cap = cv2.VideoCapture(video_path)

    initial_frame = np.zeros(tuple(map(int, (cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH), 3))), dtype=np.uint8)
    audience_detector = AudienceDetector(initial_frame, 300)

    counts = {
        "only_body":0,
        "common":0,
        "critical":0
    }

    base_dataset = os.path.join("./output_dataset", video_path.split("/")[-1].split(".")[0])

    if not os.path.exists(base_dataset):
        os.mkdir(base_dataset)

    for k in counts.keys():
        folder_path = os.path.join(base_dataset, k)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    yolo_model = get_model("../yolov5/weights/yolov5s_body_face_baseline.pt")
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if audience_detector.is_audience_present2(frame):
            results = detect_single(frame, yolo_model)
            if len(results):
                # Check size of all faces. If face is greater than 0.1 percent of the screen add it to the critical frames
                # else add it to the common folder
                has_face_flag, has_face_percent_flag = has_face(results)
                if has_face:
                    if has_face_percent_flag:
                        print("Faces found bigger than threshold")
                        write_to_folder(base_dataset, "critical", frame, counts)
                    else:
                        print("Faces found smaller than threshold")
                        write_to_folder(base_dataset, "common", frame, counts)
                else:
                    print("No faces found. Only body")
                    write_to_folder(base_dataset, "ad", frame, counts)
                    

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    assert len(sys.argv) == 2, exit(0)

    video_path = sys.argv[1]

    # import pdb; pdb.set_trace()

    main(video_path)