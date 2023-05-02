import cv2
import numpy as np
import sys
import time

from os import listdir, makedirs, path
from time import time


def crop_eye_from_image(image: np.ndarray, 
                        eye_landmarks: np.ndarray, # single eye landmarks
                        eyebrow_landmarks: np.ndarray, # corresponding eyebrow landmarks
                        output_width: int = 32, 
                        output_height: int = 32) -> np.ndarray:
    x1 = int(min(eye_landmarks[:, 0].min(), eyebrow_landmarks[:, 0].min()))
    x2 = int(max(eye_landmarks[:, 0].max(), eyebrow_landmarks[:, 0].max()))
    y1 = int(min(eye_landmarks[:, 1].min(), eyebrow_landmarks[:, 1].min()))
    y2 = int(max(eye_landmarks[:, 1].max(), eyebrow_landmarks[:, 1].max()))
    y2 += int(0.2*(y2-y1)) # increase height to fit eyes
    cropped = image[y1:y2, x1:x2]
    resized_cropped = cv2.resize(cropped, (output_width, output_height))

    return resized_cropped

def crop_eyes_from_video_frames(video_path: str, landmarks_path: str, dest_path: str):
    makedirs(dest_path, exist_ok=True)
    # Load the video file
    cap = cv2.VideoCapture(video_path)
    try:
        # Load the npy file of face landmarks
        landmarks = np.load(landmarks_path)

        # Create a new mp4 files
        frame_number = 0
        # Iterate over the frames of the mp4 file
        while cap.isOpened():
            # Read the next frame
            ret, frame = cap.read()

            # If the frame was not read successfully, break
            if not ret:
                frame_number += 1
                break  
            

            frame_landmarks = landmarks[frame_number]
            
            left_eye_path = f"{dest_path}/left_eye_{frame_number:0>5}.png"
            left_eye_cropped_resized = crop_eye_from_image(frame, 
                                                           eye_landmarks=frame_landmarks[36:42], 
                                                           eyebrow_landmarks=frame_landmarks[17:22])
            cv2.imwrite(left_eye_path, left_eye_cropped_resized)
            
            right_eye_path = f"{dest_path}/right_eye_{frame_number:0>5}.png"
            right_eye_cropped_resized = crop_eye_from_image(frame, 
                                                            eye_landmarks=frame_landmarks[42:48], 
                                                            eyebrow_landmarks=frame_landmarks[22:27])
            cv2.imwrite(right_eye_path, right_eye_cropped_resized)

            frame_number += 1
    finally:
        # Close the video file
        cap.release()

# landmarks directory structure: landmarks > person > event > landmarks.npy; e.g. npy/id00012/_raOc3-IRsw/00112.npy
# videos directory structure: videos > person > event > video_id.mp4; e.g. npy/id00012/_raOc3-IRsw/00112.mp4
def crop_eyes_from_videos(videos_dir: str, landmarks_dir: str, eyes_dir: str, print_info : bool = False):
    """
    Crop eyes from videos that has corresponding face landmarks and save them to eyes_dir
    """
    count, err_count, start_time = 0, 0, time()
    for person in set(listdir(videos_dir)).intersection(set(listdir(landmarks_dir))):
        vevents = set(listdir(path.join(videos_dir, person)))
        levents = set(listdir(path.join(landmarks_dir, person)))
        for event in levents.intersection(vevents):
            ventries = set(entry[:-4] for entry in listdir(path.join(videos_dir, person, event)) if entry.endswith('.mp4'))
            lentries = set(entry[:-4] for entry in listdir(path.join(landmarks_dir, person, event)) if entry.endswith('.npy'))
            eyes_path = path.join(eyes_dir, person, event)
            processed_entries = set(listdir(eyes_path)) if path.exists(eyes_path) else set()
            for entry in ventries.intersection(lentries).difference(processed_entries):
                landmarks_path = path.join(landmarks_dir, person, event, f"{entry}.npy")
                video_path = path.join(videos_dir, person, event, f"{entry}.mp4")
                eyes_path = path.join(eyes_dir, person, event, entry)
                try:
                    crop_eyes_from_video_frames(video_path, landmarks_path, eyes_path)
                    count += 1
                except Exception as e:
                    print(f"Error w/ {video_path}:", e)
                    err_count += 1
                    pass
    if print_info:
        time_spent = time() - start_time
        print(f"{count} videos are processed in {time_spent:.2f} seconds (~{(count/time_spent):.2f} videos per second)")
        print(f"{err_count} videos couldn't be processed because of errors")

if __name__ == '__main__':
    videos_dir = sys.argv[1]
    landmarks_dir = sys.argv[2]
    eyes_dir = sys.argv[3]
    crop_eyes_from_videos(videos_dir, landmarks_dir, eyes_dir, print_info=True)