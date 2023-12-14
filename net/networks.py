import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out,inst=True, act=nn.LeakyReLU(0.2, inplace=True)):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm2d(ch_out) if inst else nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True) if act is None else act,
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm2d(ch_out) if inst else nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True) if act is None else act
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, inst=True, act=nn.LeakyReLU(0.2, inplace=True)):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm2d(ch_out) if inst else nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True) if act is None else act
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    
class conv3d_block(nn.Module):
    def __init__(self, ch_in, ch_out, inst=True, act=nn.LeakyReLU(0.2, inplace=True)):
        super(conv3d_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out) if inst else nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True) if act is None else act,
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out) if inst else nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True) if act is None else act
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv3d(nn.Module):
    def __init__(self, ch_in, ch_out, inst=True, act=nn.LeakyReLU(0.2, inplace=True)):
        super(up_conv3d, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out) if inst else nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True) if act is None else act
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Attention_block3d(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block3d, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class U_Net(nn.Module):
    def __init__(self, spat_dim=2, img_ch=3, output_ch=1, kernels=(64, 128, 256, 512, 1024), inst=True, act=nn.LeakyReLU(0.2, inplace=True), deep_supervision=False, n_ds=2):
        super(U_Net, self).__init__()

        self.spat_dim = spat_dim
        self.deep_supervision = deep_supervision
        self.n_ds = n_ds

        if spat_dim == 2:
            conv_block_class = conv_block
            up_conv_class = up_conv
            max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.Conv_1x1 = nn.Conv2d(kernels[0], output_ch, kernel_size=1, stride=1, padding=0)
        elif spat_dim == 3:
            conv_block_class = conv3d_block
            up_conv_class = up_conv3d
            max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.Conv_1x1 = nn.Conv3d(kernels[0], output_ch, kernel_size=1, stride=1, padding=0)
        else:
            raise ValueError("Unsupported number of input channels. Expected 3 or 4.")

        self.encoder_convs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        # Create encoder layers
        for i, ch_out in enumerate(kernels):
            ch_in = img_ch if i == 0 else kernels[i - 1]
            self.encoder_convs.append(conv_block_class(ch_in, ch_out, inst=inst, act=act))

        # Create decoder layers
        for i in range(len(kernels) - 1, 0, -1):
            self.up_layers.append(up_conv_class(kernels[i], kernels[i - 1], inst=inst, act=act))
            self.decoder_convs.append(conv_block_class(kernels[i-1]*2, kernels[i - 1], inst=inst, act=act))

        self.max_pool = max_pool

        if self.deep_supervision:
            self.dsv_layers = nn.ModuleList()
            for i in reversed(range(self.n_ds)):
                if self.spat_dim == 2:
                    self.dsv_layers.append(nn.Conv2d(kernels[i], output_ch, kernel_size=1))
                elif self.spat_dim == 3:
                    self.dsv_layers.append(nn.Conv3d(kernels[i], output_ch, kernel_size=1))


    def forward(self, x):
        enc_outputs = []
        # Encoding path
        for i, enc_conv in enumerate(self.encoder_convs):
            x = enc_conv(x)
            enc_outputs.append(x)
            if (i + 1) != len(self.encoder_convs):
                x = self.max_pool(x)

        # Decoding path
        dsv_outputs = []
        for i, (up, dec_conv) in enumerate(zip(self.up_layers, self.decoder_convs)):
            x = up(x)
            x = torch.cat((enc_outputs[-(i + 2)], x), dim=1)
            x = dec_conv(x)
            if self.deep_supervision and i > (len(self.up_layers) - 1 - self.n_ds):
                dsv_outputs.append(self.dsv_layers[i-len(self.up_layers)+self.n_ds](x))

        d1 = self.Conv_1x1(x)
        if self.deep_supervision:
            dsv_outs = [d1,]
            for dsv_out in dsv_outputs:
                if self.spat_dim == 3:
                    dsv_outs.append(nn.functional.interpolate(dsv_out, d1.shape[2:], mode="trilinear", align_corners=True))
                elif self.spat_dim == 2:
                    dsv_outs.append(nn.functional.interpolate(dsv_out, d1.shape[2:], mode="bilinear", align_corners=True))
            return dsv_outs


        d1 = self.Conv_1x1(x)
        return d1
                
class AttU_Net(nn.Module):
    def __init__(self, spat_dim=2, img_ch=3, output_ch=1, kernels=(64, 128, 256, 512, 1024), inst=True, act=nn.LeakyReLU(0.2, inplace=True), deep_supervision=False, n_ds=2):
        super(AttU_Net, self).__init__()
        
        self.spat_dim = spat_dim
        self.deep_supervision = deep_supervision
        self.n_ds = n_ds
        
        if spat_dim == 2:
            conv_block_class = conv_block
            up_conv_class = up_conv
            attention_block_class = Attention_block
            max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.Conv_1x1 = nn.Conv2d(kernels[0], output_ch, kernel_size=1, stride=1, padding=0)
        elif spat_dim == 3:
            conv_block_class = conv3d_block
            up_conv_class = up_conv3d
            attention_block_class = Attention_block3d
            max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.Conv_1x1 = nn.Conv3d(kernels[0], output_ch, kernel_size=1, stride=1, padding=0)
        else:
            raise ValueError("Unsupported number of input channels. Expected 3 or 4.")

        self.encoder_convs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        # Create encoder layers and attention blocks
        for i, ch_out in enumerate(kernels):
            ch_in = img_ch if i == 0 else kernels[i - 1]
            self.encoder_convs.append(conv_block_class(ch_in, ch_out, inst=inst, act=act))
            if i != len(kernels) - 1:
                self.attention_blocks.append(attention_block_class(F_g=kernels[i], F_l=kernels[i], F_int=kernels[i] // 2))

        # Create decoder layers and up layers
        for i in range(len(kernels) - 1, 0, -1):
            self.up_layers.append(up_conv_class(kernels[i], kernels[i - 1], inst=inst, act=act))
            self.decoder_convs.append(conv_block_class(kernels[i-1]*2, kernels[i - 1], inst=inst, act=act))

        self.max_pool = max_pool
        self.sigmoid = nn.Sigmoid()

        if self.deep_supervision:
            self.dsv_layers = nn.ModuleList()
            for i in reversed(range(self.n_ds)):
                if self.spat_dim == 2:
                    self.dsv_layers.append(nn.Conv2d(kernels[i], output_ch, kernel_size=1))
                elif self.spat_dim == 3:
                    self.dsv_layers.append(nn.Conv3d(kernels[i], output_ch, kernel_size=1))

    def forward(self, x):
        enc_outputs = []
        # Encoding path
        for enc_conv in self.encoder_convs:
            x = enc_conv(x)
            enc_outputs.append(x)
            if len(enc_outputs) != len(self.encoder_convs):
                x = self.max_pool(x)

        # Decoding path
        dsv_outputs = []
        for i, (up, dec_conv) in enumerate(zip(self.up_layers, self.decoder_convs)):
            x = up(x)
            if self.attention_blocks:
                x = self.attention_blocks[-(i+1)](x, enc_outputs[-(i + 2)])
            x = torch.cat((enc_outputs[-(i + 2)], x), dim=1)
            x = dec_conv(x)
            if self.deep_supervision and i > (len(self.up_layers) - 1 - self.n_ds):
                dsv_outputs.append(self.dsv_layers[i-len(self.up_layers)+self.n_ds](x))

        d1 = self.Conv_1x1(x)
        if self.deep_supervision:
            dsv_outs = [d1,]
            for dsv_out in dsv_outputs:
                if self.spat_dim == 3:
                    dsv_outs.append(nn.functional.interpolate(dsv_out, d1.shape[2:], mode="trilinear", align_corners=True))
                elif self.spat_dim == 2:
                    dsv_outs.append(nn.functional.interpolate(dsv_out, d1.shape[2:], mode="bilinear", align_corners=True))
            return dsv_outs
        else:
            return d1
