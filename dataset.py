from enum import Enum
from pathlib import Path
from PIL import Image
from random import randrange
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing import List

import numpy as np


ALLOWED_IMAGE_EXTENSIONS = ('.jpg', '.png', '.tif', '.tiff')

class Modality(Enum):
    RGB = 'rgb'
    FLOW = 'flow'

class Video:
    """ 
    Video in this case is a directory that stores its frames/images. 
    Frames should be named in alphabetical order to keep the order, 
    i.e img_001=frames[0], img_001=frames[1], img_002=frames[2], ...)
    """
    path: Path
    frames: List[path]

    def __init__(self, video_path: Path):
        self.path = video_path
        self.frames = tuple(x for x in sorted(self.path.iterdir()) if x.suffix in ALLOWED_IMAGE_EXTENSIONS)

    def __getitem__(self, index: int | slice) -> Path | List[Path]:
        if isinstance(index, slice):
            return self.frames[index.start : index.stop : index.step]

        return self.frames[index]

    def __len__(self) -> int:
        return len(self.frames)
        
class VideoDataset(Dataset):
    dataset_path: Path 
    videos: List[Video]
    num_groups: int
    num_samples: int
    frames_per_group: int
    sampling_rate: int
    dense_sampling: bool
    modality: Modality
    num_consecutive_frames: int
    transform: Compose

    def __init__(self, dataset_path: Path, num_groups: int = 64, frames_per_group: int = 4, dense_sampling: bool = False, 
                 modality: Modality = Modality.RGB, num_consecutive_frames: int = 1, transform: Compose = None):
        """
        Args:
            dataset_path (Path): the file path to the root of video folder
            num_groups (int): number of frames per data sample
            frames_per_group (int): number of frames within one group
            modality (Modality): rgb or flow
            dense_sampling (bool): dense sampling in I3D
            transform (Compose): the transformer for preprocessing
        
        Argments have different meaning when dense_sampling is True:
            - num_groups ==> num of samples (sample consists of 1 frame)
            - frames_per_group ==> sampling rate
        """
        self.path = dataset_path
        # load only dirs that has at least one image. Later we may change minimum number of images required
        self.videos = [video for child in self.path.iterdir() if child.is_dir() and len(video := Video(child))]
        self.num_groups = self.num_samples = num_groups
        self.frames_per_group = self.sampling_rate = frames_per_group
        self.dense_sampling = dense_sampling
        self.modality = modality
        self.num_consecutive_frames = num_consecutive_frames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int):
        video = self.videos[index] 

        # frame_indices = self._get_random_frame_indices_from_each_segment(video) # TAMNet normal sampling
        images = []
        for i in self._get_sample_indices(video):
            for j in range(self.num_consecutive_frames):
                index = min(i+j, len(video)-1) # next frame or last frame
                images.extend(self._load_frames(video.frames[index]))

        return self.transform(images)
    
    def _get_sample_indices(self, video: Video) -> np.ndarray:
        max_frame_index = max(0, len(video) - self.num_consecutive_frames)
        
        if self.dense_sampling:
            highest_start_index = max_frame_index - self.sampling_rate * self.num_samples
            random_offset = randrange(0, highest_start_index + 1) if highest_start_index > 0 else 0
            """
            -------------------------------------------------- Full video
            x---------x---------x---------x---------x--------- random_offset = 0
            -x---------x---------x---------x---------x-------- random_offset = 1
            -----x---------x---------x---------x---------x---- random_offset = randrange(0, highest_start_index)
                 1    +    1    +    1    +    1    +    1     = sampling_size
                 <----d---->                                   = sampling_rate
            """
            return [(random_offset + i * self.sampling_rate) % len(video) 
                    for i in range(self.num_samples)]
        
        """
        num_groups = 8, frames_per_group = 4
        -xx-x--x -xx-x--x -xx-x--x -xx-x--x -xx-x--x -xx-x--x -xx-x--x -xx-x--x
        """
        total_frames = self.num_groups * self.frames_per_group
        avg_frames_per_group = max_frame_index // self.num_groups
        if avg_frames_per_group >= self.frames_per_group:
            # randomly sample f images per segement
            indices = np.arange(0, self.num_groups) * avg_frames_per_group
            indices = np.repeat(indices, repeats=self.frames_per_group)
            offsets = np.random.choice(avg_frames_per_group, self.frames_per_group, replace=False)
            offsets = np.tile(offsets, self.num_groups)
            indices = indices + offsets
        elif max_frame_index < total_frames:
            # need to sample the same images
            indices = np.random.choice(max_frame_index, total_frames)
        else:
            # sample cross all images
            indices = np.random.choice(max_frame_index, total_frames, replace=False)
        
        return np.sort(indices)

    def _get_random_frame_indices_from_each_segment(self, video: Video) -> List[int]:
        segment_length = len(video) // self.num_segments
        # for i in range()
        return [randrange(i, i + segment_length) for i in range(self.num_segments)]
    
    def _load_frames(self, video: Video, index: int) -> List[Image.Image]:
        image = Image.open(video.frames[index]).convert('RGB')
        image_copy = image.copy()
        image.close()

        return [image_copy]
