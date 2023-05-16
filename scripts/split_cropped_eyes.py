import os
import shutil

if __name__ == '__main__':
    prev_path = '/media/data2/eric/voxceleb2/eyes'
    new_path = '/media/data2/eric/ssldf-voxceleb2'

    left_eyes_path = os.path.join(new_path, 'left-eye-frames')
    right_eyes_path = os.path.join(new_path, 'right-eye-frames')
    if not os.path.exists(left_eyes_path):
        os.makedirs(left_eyes_path)
    if not os.path.exists(right_eyes_path):
        os.makedirs(right_eyes_path)

    people = os.listdir(prev_path)
    for person in people:
        events = os.path.join(prev_path, person)
        for event in os.listdir(events):
            videos = os.path.join(prev_path, person, event)
            for video in videos:
                images = os.path.join(prev_path, person, event, video)
                new_video_name = "_".join((person, event, video))
                left_eye_video_path = os.path.join(left_eyes_path, new_video_name)
                right_eye_video_path = os.path.join(right_eyes_path, new_video_name)
                if not os.path.exists(left_eye_video_path):
                    os.makedirs(left_eye_video_path)
                if not os.path.exists(right_eye_video_path):
                    os.makedirs(right_eye_video_path)
                for image in images:
                    image_file = os.path.join(prev_path, person, event, video, image)
                    if image[:9] == 'left_eye_':
                        left_eye_image_file = os.path.join(left_eye_video_path, image[9:])
                        shutil.copyfile(image_file, left_eye_image_file)
                    elif image[:10] == 'right_eye_':
                        right_eye_image_file = os.path.join(right_eye_video_path, image[10:])
                        shutil.copyfile(image_file, right_eye_image_file)
                    else:
                        print('There is some critical error')