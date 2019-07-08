import torch.nn as nn
from pytorch_zoo.abstract_model import EncoderDecoder

class Resnet(EncoderDecoder):
    def __init__(self, num_classes, num_channels, encoder_name):
        #self.num_channels = num_channels
        super().__init__(num_classes, num_channels, encoder_name)
        #print ("unet.py, class Resnet, self.num_channels", num_channels)
        #print ("unet.py, class Resnet, EncoderDecoder.num_channels", EncoderDecoder.num_channels)
    
    def get_encoder(self, encoder, layer, num_channels=3):        
        #print ("unet.py, encoder:", encoder)
        #print ("unet.py, encoder.num_channels:", encoder.num_channels)
        if layer == 0:
            #print ("unet.py, encoder.conv1:", encoder.conv1)
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class Resnet34_upsample(Resnet):
    def __init__(self, num_classes, num_channels=3):
        #self.num_channels = num_channels
        #print ("unet.py, class Resnet34_upsample, num_channels", num_channels)
        super().__init__(num_classes, num_channels, encoder_name='resnet34')
        #print ("unet.py, class Resnet34_upsample, num_channels", num_channels)

