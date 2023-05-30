import os
import shutil

# Define the source and destination directories
source_dir_left = "/media/data2/eric/train_test/test/left-eye-frames"
source_dir_right = "/media/data2/eric/train_test/test/right-eye-frames"
destination_dir = "/media/data2/eric/eye_pairs/test"
os.makedirs(destination_dir, exist_ok=True)


# Iterate over the subfolders in the left directory
for folder_name in os.listdir(source_dir_left):
    folder_path_l = os.path.join(source_dir_left, folder_name)
    folder_path_r = os.path.join(source_dir_right, folder_name)
    if os.path.isdir(folder_path_l):
        # Iterate over the image files in the left subfolder
        for filename in os.listdir(folder_path_l):
            if filename.endswith(".png"):
                image_id = os.path.splitext(filename)[0]
                image_folder = os.path.join(destination_dir, filename, image_id)
                os.makedirs(image_folder, exist_ok=True)
                # Move the left and right image to the image folder
                shutil.move(os.path.join(folder_path_l, filename), os.path.join(image_folder, "left.png"))
                shutil.move(os.path.join(folder_path_r, filename), os.path.join(image_folder, "right.png"))