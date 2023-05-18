import os
import shutil

def split_data_folders(left_folder, right_folder, output_folder, train_ratio, test_ratio): #val_ratio):
    # Create output folders
    sub_r = 'right-eye-frames'
    sub_l = 'left-eye-frames'
    
    
    train_folder_l = os.path.join(output_folder, 'train',sub_l)
    train_folder_r = os.path.join(output_folder, 'train',sub_r)
    test_folder_l = os.path.join(output_folder, 'test',sub_l )
    test_folder_r = os.path.join(output_folder, 'test',sub_r)

    os.makedirs(train_folder_r, exist_ok=True)
    os.makedirs(train_folder_l, exist_ok=True)
    os.makedirs(test_folder_r, exist_ok=True)
    os.makedirs(test_folder_l, exist_ok=True)

    
    #Load all the video ID
    file_list = os.listdir(left_folder)
    total_files = len(file_list)
   
    #Count and load Train and test ID
    train_count = int(total_files * train_ratio)
    test_count = int(total_files - train_count)
    
    train_data = file_list[0:train_count]
    test_data = file_list[train_count:]


    # Copy images to the train folder
    for i in range(train_count):
        
        # Get a list of image files in both left and right folders
        images_id = train_data[i]
        
        # Initialize the directory in every iteration
        train_folder_l = os.path.join(output_folder, 'train',sub_l)
        train_folder_r = os.path.join(output_folder, 'train',sub_r)
        
        #Make Folder based on Video ID
        train_folder_l = os.path.join(train_folder_l,images_id)
        train_folder_r = os.path.join(train_folder_r,images_id)
        os.makedirs(train_folder_l, exist_ok = True)
        os.makedirs(train_folder_r, exist_ok = True)
     
        #Path to that image ID
        images_r = os.path.join(right_folder,images_id)
        images_l = os.path.join(left_folder,images_id)
   
        #Load Images in that ID folder
        left_image_files = [f for f in os.listdir(images_l)]
        right_image_files = [f for f in os.listdir(images_r)]

        # Pair the images based on their indices
        image_pairs = list(zip(left_image_files, right_image_files))
        
        
        for j in range (len(image_pairs)):
    
            left_image_file, right_image_file = image_pairs[j]
            shutil.copy(os.path.join(images_l, left_image_file), os.path.join(train_folder_l, left_image_file))
            shutil.copy(os.path.join(images_r, right_image_file), os.path.join(train_folder_r, right_image_file))
            
        print("Train Folder done: ", i+1)
            
    # Copy images to the test folder
    for i in range(test_count):
        # Get a list of image files in both left and right folders
        images_id = test_data[i]
        
        test_folder_l = os.path.join(output_folder, 'test',sub_l )
        test_folder_r = os.path.join(output_folder, 'test',sub_r)
        
        #Make Folder based on Video ID
        test_folder_l = os.path.join(test_folder_l,images_id)
        test_folder_r = os.path.join(test_folder_r,images_id)
        os.makedirs(test_folder_l, exist_ok = True)
        os.makedirs(test_folder_r, exist_ok = True)
        
        #Path to that image ID
        images_r = os.path.join(right_folder,images_id)
        images_l = os.path.join(left_folder,images_id)
        
        #Load Images in that ID folder
        left_image_files = [f for f in os.listdir(images_l)]
        right_image_files = [f for f in os.listdir(images_r)]

        
        # Pair the images based on their indices
        image_pairs = list(zip(left_image_files, right_image_files))
        
        
        for j in range (len(image_pairs)):
    
            left_image_file, right_image_file = image_pairs[j]
            shutil.copy(os.path.join(images_l, left_image_file), os.path.join(train_folder_l, left_image_file))
            shutil.copy(os.path.join(images_r, right_image_file), os.path.join(train_folder_r, right_image_file))


        print("Test Folder done: ", i+1)


# Specify the left folder containing the left images
left_folder = '/media/data2/eric/ssldf_voxceleb2/left-eye-frames'

# Specify the right folder containing the right images
right_folder = '/media/data2/eric/ssldf_voxceleb2/right-eye-frames'

# Specify the output folder where the train, test, and validation folders will be created
output_folder = '/media/data2/eric/train_test'



# Specify the ratio of images for train, test, and validation (e.g., 0.7, 0.2, 0.1)
train_ratio = 0.8
test_ratio = 0.2
# val_ratio = 0.1

# Call the function to split the data
split_data_folders(left_folder, right_folder, output_folder, train_ratio, test_ratio)# val_ratio)
