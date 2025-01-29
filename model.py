import torch
import torch.nn as nn
import torch.fft as fft

class DeepFreqFilt(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DeepFreqFilt, self).__init__()
        self.conv2d_1x1 = nn.Conv2d(
            in_channels=in_channels * 2, 
            out_channels=out_channels * 2, 
            kernel_size=(1, 1),
            bias=False
        )
        self.insnorm = nn.InstanceNorm2d(num_features=out_channels * 2)
        self.prelu = nn.PReLU()

        self.conv2d_7 = nn.Conv2d(
            in_channels=2, 
            out_channels=1, 
            kernel_size=(7, 7),
            stride=1, 
            padding=3, 
            padding_mode='zeros', 
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
        self.half_channels = out_channels

    def forward(self, x):
        # Fourier Transform
        fft_x = fft.rfft2(x, norm='ortho')
        fft_x = torch.cat((fft_x.real, fft_x.imag), dim=1)
        
        fft_x = self.conv2d_1x1(fft_x)
        fft_x = self.insnorm(fft_x)
        fft_x = self.prelu(fft_x)

        # Spatial attention
        x1 = torch.max(fft_x, dim=1, keepdim=True)[0]
        x2 = torch.mean(fft_x, dim=1, keepdim=True)
        x_ = torch.cat((x1, x2), dim=1)
        x_ = self.conv2d_7(x_)
        x_ = self.sigmoid(x_)
       
        output = fft_x * x_
        output_real = output[:, :self.half_channels, :, :]
        output_imag = output[:, self.half_channels:, :, :]
        output_complex = torch.complex(output_real, output_imag)

        # Inverse Fourier Transform
        x = fft.irfft2(output_complex, s=(x.shape[2], x.shape[3]), norm='ortho')

        return x

class ConvBlock_with_DFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.DFF1 = DeepFreqFilt(out_channels, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.DFF2 = DeepFreqFilt(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu1(x)
        x_ = self.DFF1(x)
        x = x + x_

        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu2(x)
        x_ = self.DFF2(x)
        x = x + x_

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu2(x)

        return x

class DAUS_net(nn.Module):
    """
    - in_channels: Number of input channels (e.g., 1 for grayscale images).
    - out_channels: Number of output channels (e.g., 1 for binary segmentation masks).
    """
    def __init__(self, in_channels, out_channels):
        super(DAUS_net, self).__init__()
        self.encoder_conv1 = ConvBlock_with_DFF(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv2 = ConvBlock_with_DFF(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv3 = ConvBlock_with_DFF(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv4 = ConvBlock_with_DFF(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv5 = ConvBlock_with_DFF(256, 512)

        self.upconv6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv6 = ConvBlock(512, 256)
        self.upconv7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv7 = ConvBlock(256, 128)
        self.upconv8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv8 = ConvBlock(128, 64)
        self.upconv9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder_conv9 = ConvBlock(64, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder_conv1(x)
        x2 = self.encoder_conv2(self.pool1(x1))
        x3 = self.encoder_conv3(self.pool2(x2))
        x4 = self.encoder_conv4(self.pool3(x3))
        x5 = self.encoder_conv5(self.pool4(x4))

        x = self.upconv6(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder_conv6(x)

        x = self.upconv7(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder_conv7(x)

        x = self.upconv8(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder_conv8(x)

        x = self.upconv9(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder_conv9(x)

        x = self.final_conv(x)
        x = torch.sigmoid(x)

        return x
