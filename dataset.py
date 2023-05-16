from os import listdir, path, scandir
from PIL import Image
from random import randrange
from torch.utils.data import Dataset
from typing import List, NamedTuple


class Video:
    path: str # path to a dir that holds frames of the video as image file
    frames: List[str] # each element is a path to image file

    def __init__(self, video_dir_path: str):
        self.path = video_dir_path
        # load only image files, in alphabetical order. 
        # NOTE: frames (image files) should be named in alphabetical order, i.e frame_001, frame_002, ...
        self.frames = [path.join(self.path, entry) for entry in sorted(listdir(self.path)) 
                       if entry[-4:] in ('.jpg', '.png', '.tif', '.tiff')]

        
class TSNDataset(Dataset):
    path_to_dataset: str
    num_segments: int
    videos: List[Video]

    def __init__(self, path_to_dataset: str, num_segments: int = 8):
        self.path_to_dataset = path_to_dataset
        self.num_segments = num_segments
        
        with scandir(self.path_to_dataset) as entries:
            # load only dirs that has at least one image. Later we may change minimum number of images required
            self.videos = [video for entry in entries 
                           if entry.is_dir() and len((video := Video(entry.path)).frames)]

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int):
        video = self.videos[index] 
        images = []
        for i in self._get_random_frame_indices_from_each_segment(video):
            for j in range(self.new_length):
                start_index = min(i+j, video.num_frames-1) # next frame or last frame
                images.extend(self._load_images(video.frames[start_index]))
        
        return self.transform((images))

    
    def _get_random_frame_indices_from_each_segment(self, video: Video) -> List[int]:
        segment_length = len(video.frames) // self.num_segments
        return [randrange(i, i + segment_length) for i in range(self.num_segments)]
    
    def _get_num_segment_random_frame_indices(self, video: Video) -> List[int]:
        return [randrange(0, len(video.frames)) for __ in range(self.num_segments)]
    
    def _load_images(self, image_path: str) -> List[Image.Image]:
        rgb_image = Image.open(image_path).convert('RGB')
        
        return [rgb_image]
