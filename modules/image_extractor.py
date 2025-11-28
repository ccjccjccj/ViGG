import cv2
import torch
from .LightGlue import LightGlueMatcher

class ImageExtractor():
    def __init__(self, device = "cuda"):
        # SuperPoint + LightGlue
        self.lightglue_matcher = LightGlueMatcher(device = device)
        self.device = device

    def SIFT(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img_gray, None)
        kp = [kp[i].pt for i in range(len(kp))]  # [kp1, kp2, ...]
        kp = torch.tensor(kp, device = self.device, dtype = torch.float32)
        des = torch.tensor(des, device = self.device, dtype = torch.float32)
        return kp, des
    
    def LightGlue(self, img0, img1):
        kp0, kp1, scores = self.lightglue_matcher.extract_and_match(img0, img1)
        dists = 1 - scores

        return kp0, kp1, dists.detach()