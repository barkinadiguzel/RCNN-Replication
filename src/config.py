# Input / output
IMAGE_SIZE = (227, 227)       # Input size for CNN
NUM_CLASSES = 21              # e.g., 20 classes + background

# Proposal settings
PROPOSAL_MODE = 'fast'        # 'fast' or 'quality'
MAX_PROPOSALS = 2000          # Maximum proposals per image

# Backbone / feature extractor
BACKBONE_CHANNELS = [3, 64, 192, 384, 256, 256]  # Example AlexNet-like channels
BACKBONE_KERNELS = [11, 5, 3, 3, 3]
BACKBONE_STRIDES = [4, 1, 1, 1, 1]
BACKBONE_PADDINGS = [2, 2, 1, 1, 1]

# ROI / patch
ROI_OUTPUT_SIZE = (227, 227)   # Crop size for each proposal
ROI_MEAN = [0.485, 0.456, 0.406]  # RGB channel mean
ROI_STD = [0.229, 0.224, 0.225]   # RGB channel std
