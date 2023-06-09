from enum import Enum
from pathlib import Path
from PIL import Image
from random import randrange
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing import Any, List, Optional, Tuple

import numpy as np


ALLOWED_IMAGE_EXTENSIONS = ('.jpg', '.png', '.tif', '.tiff')

class Modality(Enum):
    RGB = 'rgb'
    FLOW = 'flow'

class FrameDir:
    """Frame is a dir that stores two images: left.ext, right.ext (ext: jpg, png, tif, tiff)"""
    path: Path
    left_img_path: Optional[Path]
    right_img_path: Optional[Path]

    def __init__(self, frame_path: Path):
        self.path = frame_path
        # check if left and right images exists in the frame dir
        # check for multiple extensions
        for ext in ALLOWED_IMAGE_EXTENSIONS:
            left_img_path = self.path.joinpath(f"left.{ext}")
            right_img_path = self.path.joinpath(f"right.{ext}")
            if left_img_path.exists() and right_img_path.exists():
                self.left_img_path = left_img_path
                self.right_img_path = right_img_path
        else:
            self.left_img_path = None
            self.right_img_path = None

    def is_valid(self):
        return self.left_img_path is not None and self.right_img_path is not None
    
class VideoDir:
    """ 
    Video in this case is a directory that stores its frames as a dir (frame_001 > left.png, right.png). 
    Frames should be named in alphabetical order to keep the order
    """
    path: Path
    frames: Tuple[FrameDir]

    def __init__(self, video_path: Path):
        self.path = video_path
        self.frames = tuple(frame for x in sorted(self.path.iterdir()) 
                            if (frame := FrameDir(x)).is_valid())

    def __getitem__(self, index: int | slice) -> Path | List[Path]:
        if isinstance(index, slice):
            return self.frames[index.start : index.stop : index.step]

        return self.frames[index]

    def __len__(self) -> int:
        return len(self.frames)
        
class ImgPairDataset(Dataset):
    dataset_path: Path 
    videos: List[VideoDir]
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
        self.videos = [video for child in self.path.iterdir() if child.is_dir() and len(video := VideoDir(child))]
        self.num_groups = self.num_samples = num_groups
        self.frames_per_group = self.sampling_rate = frames_per_group
        self.dense_sampling = dense_sampling
        self.modality = modality
        self.num_consecutive_frames = num_consecutive_frames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, video_index: int) -> Tuple[Any, Any]:
        video = self.videos[video_index] 

        # frame_indices = self._get_random_frame_indices_from_each_segment(video) # TAMNet normal sampling
        left_images, right_images = [], []
        for i in self._get_sample_indices(video):
            for j in range(self.num_consecutive_frames):
                frame_index = min(i+j, len(video)-1) # next frame or last frame
                left_image, right_image = self._load_pair_images(video, frame_index)
                left_images.append(left_image)
                right_images.append(right_image)

        if self.transform is not None: 
            return self.transform(left_images), self.transform(right_images)
        
        return left_images, right_images
    
    def _get_sample_indices(self, video: VideoDir) -> np.ndarray:
        max_frame_index = max(0, len(video) - self.num_consecutive_frames)
        
        if self.dense_sampling:
            highest_start_index = max_frame_index - self.sampling_rate * self.num_samples
            random_offset = randrange(0, highest_start_index + 1) if highest_start_index > 0 else 0
            """
            -------------------------------------------------- Full video
            -----x---------x---------x---------x-------------- random_offset = randrange(0, highest_start_index)
                 1    +    1    +    1    +    1               = num_saples = 4
                 <--------->                                   = sampling_rate = 10
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

    def _get_random_frame_indices_from_each_segment(self, video: VideoDir) -> List[int]:
        segment_length = len(video) // self.num_segments
        # for i in range()
        return [randrange(i, i + segment_length) for i in range(self.num_segments)]
    
    def _load_pair_images(self, video: VideoDir, index: int) -> Tuple[Image.Image, Image.Image]:
        image = Image.open(video.frames[index].left_img_path).convert('RGB')
        left_image = image.copy()
        image.close()

        image = Image.open(video.frames[index].right_img_path).convert('RGB')
        right_image = image.copy()
        image.close()

        return left_image, right_image
