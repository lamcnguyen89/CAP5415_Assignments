"""

Build Point Cloud Generator Pytorch model:

    This module creates the neural network architecture

"""
import torch
from torch import nn
from torch.nn import functional as F

# CNN block for encoder
def conv2d_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

# CNN block for decoder
def deconv2d_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

# Linear NN block
def linear_block(in_c, out_c):
    return nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.BatchNorm1d(out_c),
        nn.ReLU(),
    )

#
def pixel_bias(outViewN, outW, outH, renderDepth):
    X, Y = torch.meshgrid([torch.arange(outH), torch.arange(outW)],indexing="xy")
    X, Y = X.float(), Y.float() # [H,W]
    initTile = torch.cat([
        X.repeat([outViewN, 1, 1]), # [V,H,W]
        Y.repeat([outViewN, 1, 1]), # [V,H,W]
        torch.ones([outViewN, outH, outW]).float() * renderDepth, 
        torch.zeros([outViewN, outH, outW]).float(),
    ], dim=0) # [4V,H,W]

    return initTile.unsqueeze_(dim=0) # [1,4V,H,W]


# Create the encoder which compresses an image into features
class Encoder(nn.Module):
    """Encoder of Structure Generator"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = conv2d_block(3, 96)
        self.conv2 = conv2d_block(96, 128)
        self.conv3 = conv2d_block(128, 192)
        self.conv4 = conv2d_block(192, 256)
        self.fc1 = linear_block(4096, 2048) # After flatten
        self.fc2 = linear_block(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)

    def num_flat_features(self,x):
        size = x.size()[1:] # all dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *=s

        return num_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc1(x.contiguous().view(-1, 4096))
        x = self.fc2(x)
        x = self.fc3(x)

        return x

# Create Decoder which takes extracted features and turns to image
class Decoder(nn.Module):
    """Build Decoder"""
    def __init__(self, outViewN, outW, outH, renderDepth):
        super(Decoder, self).__init__()
        self.outViewN = outViewN

        self.relu = nn.ReLU()
        self.fc1 = linear_block(512, 1024)
        self.fc2 = linear_block(1024, 2048)
        self.fc3 = linear_block(2048, 4096)
        self.deconv1 = deconv2d_block(256, 192)
        self.deconv2 = deconv2d_block(192, 128)
        self.deconv3 = deconv2d_block(128, 96)
        self.deconv4 = deconv2d_block(96, 64)
        self.deconv5 = deconv2d_block(64, 48)
        self.pixel_conv = nn.Conv2d(48, outViewN*4, 1, stride=1, bias=False)
        self.pixel_bias = pixel_bias(outViewN, outW, outH, renderDepth)

    def forward(self, x):
        x = self.relu(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view([-1, 256, 4, 4])
        x = self.deconv1(F.interpolate(x, scale_factor=2))
        x = self.deconv2(F.interpolate(x, scale_factor=2))
        x = self.deconv3(F.interpolate(x, scale_factor=2))
        x = self.deconv4(F.interpolate(x, scale_factor=2))
        x = self.deconv5(F.interpolate(x, scale_factor=2))
        x = self.pixel_conv(x) + self.pixel_bias.to(x.device)
        XYZ, maskLogit = torch.split(
            x, [self.outViewN * 3, self.outViewN], dim=1)

        return XYZ, maskLogit


# Structure Generator combines encoder and decoder together and generates projections at different coordinates
class Structure_Generator(nn.Module):
    """Structure generator components in PCG"""

    def __init__(self, encoder=None, decoder=None,
                 outViewN=8, outW=128, outH=128, renderDepth=1.0):
        super(Structure_Generator, self).__init__()

        if encoder: self.encoder = encoder
        else: self.encoder = Encoder()

        if decoder: self.decoder = decoder
        else: self.decoder = Decoder(outViewN, outW, outH, renderDepth)

    def forward(self, x):
        latent = self.encoder(x)
        XYZ, maskLogit = self.decoder(latent)

        return XYZ, maskLogit


# TESTING
if __name__ == '__main__':
    import options
    cfg = options.get_arguments()
    encoder = Encoder()
    decoder = Decoder(cfg.outViewN, cfg.outW, cfg.outH, cfg.renderDepth)
    model = Structure_Generator()
