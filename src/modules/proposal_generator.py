import cv2
import numpy as np

class ProposalGenerator:
    def __init__(self, mode='fast'):
        self.mode = mode
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    def generate(self, image, batch_idx=0):
        self.ss.setBaseImage(image)
        if self.mode == 'fast':
            self.ss.switchToSelectiveSearchFast()
        else:
            self.ss.switchToSelectiveSearchQuality()
        rects = self.ss.process()  # (x, y, w, h)
        
        rois = []
        for (x, y, w, h) in rects:
            rois.append([batch_idx, x, y, x+w, y+h])
        return np.array(rois)
