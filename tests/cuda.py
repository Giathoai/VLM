import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import torch
from torchinfo import summary
from models.vit import VIT
from models.vlm import SeeMoreVLM

image_size = (8, 3, 224, 224) 

text_size = (16, 128) 
batch_size = 8
vision_encoder = VIT(embedding_dim=512, num_classes=1, num_layers=6)

summary(
    SeeMoreVLM(vision_encoder), 
    input_size=[(batch_size, 3, 224, 224), (batch_size, 128)], # Pass as a list of shapes
    dtypes=[torch.float32, torch.long],  # Important: input_ids must be Long/Int
    col_names=["input_size", "output_size", "num_params", "trainable"], 
    col_width=20
)

print(torch.version.cuda)
