import torch
import numpy as np

from .lightglue import LightGlue
from .superpoint import SuperPoint

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.tensor(image / 255., dtype=torch.float)

def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
            for k, v in data.items()}

class LightGlueMatcher():
    def __init__(self, device = "cuda"):
        self.extractor = SuperPoint(max_num_keypoints = 2048).eval().to(device)
        self.matcher = LightGlue(features = 'superpoint').eval().to(device)
        self.device = device
    
    def extract_and_match(self, img0, img1):
        image0 = img0[..., ::-1]
        image0 = numpy_image_to_torch(image0)
        image0 = image0.to(self.device)
        image1 = img1[..., ::-1]
        image1 = numpy_image_to_torch(image1)
        image1 = image1.to(self.device)

        # Extract
        feats0 = self.extractor.extract(image0)
        feats1 = self.extractor.extract(image1)

        # Match
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        scores = matches01['scores']
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

        return points0, points1, scores