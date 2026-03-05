import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import torch
from torchinfo import summary
from models.vit import VIT
from models.vlm import SeeMoreVLM

summary(SeeMoreVLM(), input_size=(16, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"], col_width=20)
print(torch.version.cuda)
