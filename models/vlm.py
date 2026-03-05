import torch
from torch import nn
from transformers import GPT2LMHeadModel

class VisionLanguageProjector(nn.Module):
    def __init__(self, vision_dim, text_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class SeeMoreVLM(nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder 
        self.language_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.projector = VisionLanguageProjector(256, self.language_model.config.hidden_size)
        
    def forward(self, images, input_ids):
        image_features = self.vision_encoder.get_features(images) 
        visual_embeds = self.projector(image_features) 
        
        text_embeds = self.language_model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        outputs = self.language_model(inputs_embeds=inputs_embeds)
        return outputs.logits