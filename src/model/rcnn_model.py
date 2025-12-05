import torch
import torch.nn as nn
import numpy as np
from src.layers.conv_layer import ConvLayer
from src.layers.roi_crop import ROICrop
from src.layers.linear_head import LinearHead
from src.modules.feature_extractor import FeatureExtractor
from src.modules.proposal_generator import ProposalGenerator
from src.modules.proposal_processor import ProposalProcessor

class RCNNModel(nn.Module):
    def __init__(self, num_classes, roi_output_size=(227, 227)):
        super().__init__()
        # Backbone CNN to extract feature maps
        self.feature_extractor = FeatureExtractor()
        
        # Region proposal generator (Selective Search)
        self.proposal_generator = ProposalGenerator()
        
        # Proposal processor: crop → resize → normalize
        self.proposal_processor = ProposalProcessor(output_size=roi_output_size)
        
        # ROI crop on feature maps (optional, additional feature extraction)
        self.roi_crop = ROICrop(output_size=roi_output_size)
        
        # Fully connected head (classifier / SVM)
        self.linear_head = LinearHead(num_classes=num_classes)
    
    def forward(self, images):
        # 1) Extract feature maps
        feature_maps = self.feature_extractor(images)
        
        # 2) Generate proposals for each image
        proposals = []
        for i, img in enumerate(images):
            # Convert tensor to numpy for OpenCV Selective Search
            rois = self.proposal_generator.generate(img.permute(1,2,0).numpy(), batch_idx=i)
            proposals.append(rois)
        proposals = torch.from_numpy(np.vstack(proposals)).float()
        
        # 3) Crop, resize, and normalize each proposal
        patches = self.proposal_processor(images, proposals)
        
        # 4) Crop feature maps (optional)
        roi_features = self.roi_crop(feature_maps, proposals)
        
        # 5) Pass cropped features through classifier head
        outputs = self.linear_head(roi_features)
        return outputs
