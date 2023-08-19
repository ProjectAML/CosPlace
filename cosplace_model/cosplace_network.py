
import torch
import logging
import torchvision
from torch import nn
from typing import Tuple
from cosplace_model.GRL import get_discriminator
from cosplace_model.layers import Flatten, L2Norm, GeM

# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "MobileNet_v3_small": 576,
    "MobileNet_v3_large": 960,
    "EfficientNet_b0": 1280,
    "EfficientNet_b1": 1280,
    "EfficientNet_b2": 1408,
    "EfficientNet_v2_s": 1280,
    "VGG16": 512,
}


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone : str, fc_output_dim : int, domain_adaptation=False):
        """Return a model for GeoLocalization.
        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
        """
        super().__init__()
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone, features_dim = get_backbone(backbone)
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm()
        )
        if domain_adaptation==True:
            print("with GRL")
            self.discriminator=get_discriminator(features_dim)
        else:
            print("without GRL")
            None

    def forward(self, x, grl=False):
        features = self.backbone(x)
        if grl==True:
            x=self.discriminator(features)
            return x
        return  self.aggregation(features)
    


def get_pretrained_torchvision_model(backbone_name : str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model


def get_backbone(backbone_name : str) -> Tuple[torch.nn.Module, int]:
    backbone = get_pretrained_torchvision_model(backbone_name)
    
    if backbone_name.startswith("ResNet"):
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    elif backbone_name.startswith("EfficientNet"):
        layers = list(backbone.features.children()) # Remove avg pooling and FC layer
        for layer in layers[:-2]: # freeze all the layers except the last two
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last two layers of EfficientNet, freeze the previous ones")

    elif backbone_name.startswith("MobileNet"):
        layers = list(backbone.features.children()) # Remove avg pooling and FC layer
        for layer in layers[:-2]: # freeze all the layers except the last two
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last two layers of MobileNet, freeze the previous ones")

    backbone = torch.nn.Sequential(*layers)
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim
