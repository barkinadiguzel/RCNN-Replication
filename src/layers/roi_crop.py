import torch
import torch.nn as nn
import torch.nn.functional as F

class ROICrop(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size  # (H, W) patch boyutu

    def forward(self, feature_map, rois):
        crops = []
        for roi in rois:
            batch_idx, x1, y1, x2, y2 = roi
            batch_idx = int(batch_idx)
            roi_feat = feature_map[batch_idx, :, int(y1):int(y2), int(x1):int(x2)]
            roi_feat = F.adaptive_max_pool2d(roi_feat, self.output_size)
            crops.append(roi_feat)
        return torch.stack(crops, dim=0)  # [num_rois, C, H_out, W_out]
