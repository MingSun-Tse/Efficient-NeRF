""" Full assembly of the parts to form the complete network """

from .unet_parts import *

'''
class UNet(nn.Module):
    def __init__(self, args, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.args = args
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        wf = args.unet.width_factor
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64 * wf) # in conv
        self.down1 = Down(64 * wf, 128 * wf) # feat: [N, 128, H//2, W//2]
        self.down2 = Down(128 * wf, 256 * wf) # feat: [N, 256, H//4, W//4]
        self.down3 = Down(256 * wf, 512 * wf) # feat: [N, 512, H//8, W//8]
        self.down4 = Down(512 * wf, 1024 * wf // factor) # feat: [N, 512, H//16, W//16]
        
        self.up4 = Up(1024 * wf, 512 * wf // factor, bilinear) # feat: [N, 256, H//8, W//8]
        self.up3 = Up(512 * wf, 256 * wf // factor, bilinear) # feat: [N, 128, H//4, W//4]
        self.up2 = Up(256 * wf, 128 * wf // factor, bilinear) # feat: [N, 64, H//2, W//2]
        self.up1 = Up(128 * wf, 64 * wf, bilinear) # feat: [N, 64, H, W]
        self.outc = OutConv(64 * wf, n_classes) # out conv

# # feature map size example of down1 ~ up1
# torch.Size([32, 128, 8, 8])
# torch.Size([32, 256, 4, 4])
# torch.Size([32, 512, 2, 2])
# torch.Size([32, 512, 1, 1])

# torch.Size([32, 256, 2, 2])
# torch.Size([32, 128, 4, 4])
# torch.Size([32, 64, 8, 8])
# torch.Size([32, 64, 16, 16])

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x3 = self.up4(x4, x3)
        x2 = self.up3(x3, x2)
        x1 = self.up2(x2, x1)
        x0 = self.up1(x1, x0)
        return self.outc(x0)
'''

class UNet(nn.Module):
    '''This UNet is a re-implementation of the above version, more flexible'''
    def __init__(self, args, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.args = args
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        
        nf, body = args.unet.base_n_filters, []
        self.inc = DoubleConv(n_channels, nf)
        # downsample part
        for i in range(args.unet.n_downsample):
            if i == args.unet.n_downsample - 1: # the last downsample layer
                body += [Down(nf, nf * 2 // factor)]
            else:
                body += [Down(nf, nf * 2)]
            nf *= 2
        
        # upsample part
        for i in range(args.unet.n_downsample):
            if i == args.unet.n_downsample - 1: # the last upsample layer
                body += [Up(nf, nf // 2, bilinear)]
            else:
                body += [Up(nf, nf // 2 // factor, bilinear)]
            nf //= 2

        self.body = nn.Sequential(*body)
        self.outc = OutConv(nf, n_classes, act=args.unet.last_act)

    def forward(self, x):
        x = self.inc(x)

        in_feats = []
        for i in range(self.args.unet.n_downsample):
            in_feats += [x]
            x = self.body[i](x)
        
        for j in range(self.args.unet.n_downsample - 1, -1, -1):
            i += 1
            x = self.body[i](x, in_feats[j])
        return self.outc(x)