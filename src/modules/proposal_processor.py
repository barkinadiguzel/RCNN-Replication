import torch
import torch.nn.functional as F

class ProposalProcessor:
    def __init__(self, output_size=(227, 227), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.output_size = output_size 
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)  # RGB channel mean
        self.std = torch.tensor(std).view(1, 3, 1, 1)    # RGB channel std

    def __call__(self, images, rois):
        crops = []
        for roi in rois:
            batch_idx, x1, y1, x2, y2 = roi
            batch_idx = int(batch_idx)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Crop patch
            patch = images[batch_idx, :, y1:y2, x1:x2]  # [C, h, w]
            
            # Resize to output_size
            patch = F.interpolate(patch.unsqueeze(0), size=self.output_size, mode='bilinear', align_corners=False)
            patch = patch.squeeze(0)
            
            # Normalize
            patch = (patch - self.mean) / self.std
            
            crops.append(patch)
        
        return torch.stack(crops, dim=0)
