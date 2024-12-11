import cv2
import os

def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f"frame_{count:04d}.jpg"), frame)
        count += 1

    cap.release()

def frames_to_video(input_folder, output_video_path, fps):
    images = [img for img in os.listdir(input_folder) if img.endswith(".jpg")]
    images.sort()

    if len(images) == 0:
        return

    first_image_path = os.path.join(input_folder, images[0])
    frame = cv2.imread(first_image_path)
    # frame = cv2.resize(frame, (384, 384))
    height, width, _ = frame.shape
    print(f"height: {height}, width: {width}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_file in images:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        # frame = cv2.resize(frame, (384, 384))
        out.write(frame)

    out.release()

# Example usage
video_path = 'cat.mov'
output_folder = 'outputs/cat_frames_meas'
output_video = 'output_video_meas.mp4'
fps = 30

# video_to_frames(video_path, output_folder)
frames_to_video(output_folder, output_video, fps)