import timm
from .models import Paper, UNet, deeplabv3plus_resnet50
from deepblocks.network import ICTNet

def model(model_name, pretrained, num_classes):
    if model_name=='paper':
        return Paper(num_classes=num_classes)
    elif model_name == 'unet':
        return UNet(3, num_classes)
    elif model_name == 'deeplab':
        return deeplabv3plus_resnet50(num_classes=num_classes, pretrained_backbone=pretrained)
    elif model_name == 'ictnet':
        return ICTNet(in_channels=3, out_channels=num_classes, n_pool=3, growth_rate=4, n_layers_per_block=[4]*7)
    return timm.create_model(model_name, pretrained, num_classes=num_classes,  )
