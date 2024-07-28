import os
import argparse
import cv2
import mediapipe as mp

def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))
    return img

def process_image(file_path, face_detection, output_dir):
    img = cv2.imread(file_path)
    if img is not None:
        img = process_img(img, face_detection)
        cv2.imwrite(os.path.join(output_dir, 'output.jpg'), img)
        print(f"Image saved to {os.path.join(output_dir, 'output.jpg')}")
    else:
        print(f"Failed to read image from {file_path}")

def process_video(file_path, face_detection, output_dir):
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read video from {file_path}")
        return

    output_video_path = os.path.join(output_dir, 'output.mp4')
    output_video = cv2.VideoWriter(output_video_path,
                                   cv2.VideoWriter_fourcc(*'MP4V'),
                                   25,
                                   (frame.shape[1], frame.shape[0]))

    while ret:
        frame = process_img(frame, face_detection)
        output_video.write(frame)
        ret, frame = cap.read()

    cap.release()
    output_video.release()
    print(f"Video saved to {output_video_path}")

def process_webcam(face_detection):
    for index in range(5):  # Try indices 0 to 4
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Using webcam index {index}")
            break
        cap.release()

    if not cap.isOpened():
        print("Failed to open webcam")
        return

    ret, frame = cap.read()
    while ret:
        frame = process_img(frame, face_detection)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='webcam', choices=['image', 'video', 'webcam'])
    parser.add_argument("--filePath", default=None)
    args = parser.parse_args()

    output_dir = './images/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        if args.mode == 'image':
            if args.filePath:
                process_image(args.filePath, face_detection, output_dir)
            else:
                print("File path is required for image mode")
        elif args.mode == 'video':
            if args.filePath:
                process_video(args.filePath, face_detection, output_dir)
            else:
                print("File path is required for video mode")
        elif args.mode == 'webcam':
            process_webcam(face_detection)

if __name__ == "__main__":
    main()
