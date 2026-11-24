"""
models.py
=========
  unet      : U-Net (base 32 channel) dengan CBAM opsional di skip connection
              -- IDENTIK dengan arsitektur versi lama.
  resnet18  : encoder ResNet-18 (opsional bobot SSL4EO-L MoCo) + decoder U-Net
              (CBAM opsional). Bobot dipilih per dataset:
                L7 Irish -> LANDSAT_ETM_TOA_MOCO
                L8 Biome -> LANDSAT_OLI_TIRS_TOA_MOCO
  deeplabv3plus : DeepLabV3+ (segmentation_models_pytorch) encoder ResNet-50,
              opsional bobot ImageNet. conv1 diadaptasi ke 9/11 band oleh smp
              (bobot RGB diulang & diskalakan). Backbone kompetitif untuk
              menguji apakah efek sampling tetap ada di pipeline yang lebih kuat.
  segformer : SegFormer (segmentation_models_pytorch) encoder MiT-B2 (24.7 M parameter),
              opsional bobot ImageNet (smp-hub/mit_b2.imagenet). Baseline Transformer
              (Fase 12); patch embedding pertama diadaptasi ke 9/11 band oleh smp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


# ---------------------------------------------------------------- CBAM
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        return torch.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        pooled = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return torch.sigmoid(self.conv(pooled))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(k)
        self.last_spatial_attn = None

    def forward(self, x):
        x = x * self.ca(x)
        sa = self.sa(x)
        self.last_spatial_attn = sa.detach()
        return x * sa


# ---------------------------------------------------------------- U-Net
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(cin, cout))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, cin, cskip, cout, use_cbam=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin // 2, 2, stride=2)
        self.use_cbam = use_cbam
        self.cbam = CBAM(cskip) if use_cbam else None
        self.conv = DoubleConv(cin // 2 + cskip, cout)

    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        if self.cbam is not None:
            skip = self.cbam(skip)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes=C.NUM_CLASSES, base=32, use_cbam=False):
        super().__init__()
        c = base
        self.inc = DoubleConv(in_channels, c)
        self.down1, self.down2 = Down(c, 2 * c), Down(2 * c, 4 * c)
        self.down3, self.down4 = Down(4 * c, 8 * c), Down(8 * c, 16 * c)
        self.up1 = Up(16 * c, 8 * c, 8 * c, use_cbam)
        self.up2 = Up(8 * c, 4 * c, 4 * c, use_cbam)
        self.up3 = Up(4 * c, 2 * c, 2 * c, use_cbam)
        self.up4 = Up(2 * c, c, c, use_cbam)
        self.outc = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------- ResNet-18 U-Net
def _adapt_conv1(weight: torch.Tensor, target: int) -> torch.Tensor:
    """Sesuaikan jumlah channel input conv1 (dipakai HANYA bila jumlah band beda)."""
    out_c, in_c, kh, kw = weight.shape
    if target == in_c:
        return weight
    if target < in_c:
        return weight[:, :target].clone()
    extra = weight.mean(1, keepdim=True).repeat(1, target - in_c, 1, 1)
    return torch.cat([weight, extra], 1) * (in_c / target)


class ResNet18Encoder(nn.Module):
    def __init__(self, dataset: str, in_channels: int, pretrained: bool):
        super().__init__()
        import timm
        self.load_info = {"pretrained": pretrained, "missing_keys": None,
                          "unexpected_keys": None, "conv1_adapted": False}
        if pretrained:
            from torchgeo.models import ResNet18_Weights
            weights = getattr(ResNet18_Weights, C.DATASETS[dataset]["pretrained_weights"])
            w_in = weights.meta["in_chans"]
            base = timm.create_model("resnet18", in_chans=w_in, num_classes=0)
            missing, unexpected = base.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            self.load_info.update(missing_keys=list(missing), unexpected_keys=list(unexpected),
                                  weights=str(weights), weights_in_chans=w_in)
            print(f"[ResNet18] bobot {weights} dimuat. missing={len(missing)} "
                  f"unexpected={len(unexpected)} (harus ~0, selain fc)")
            if w_in != in_channels:
                print(f"[ResNet18] PERINGATAN: bobot {w_in} band vs data {in_channels} band. "
                      "conv1 diadaptasi; PASTIKAN urutan band cocok.")
                old = base.conv1
                new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride,
                                old.padding, bias=False)
                new.weight.data = _adapt_conv1(old.weight.data, in_channels)
                base.conv1 = new
                self.load_info["conv1_adapted"] = True
        else:
            base = timm.create_model("resnet18", in_chans=in_channels, num_classes=0)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.act1)
        self.pool = base.maxpool
        self.layer1, self.layer2 = base.layer1, base.layer2
        self.layer3, self.layer4 = base.layer3, base.layer4

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4


class ResNetUNet(nn.Module):
    def __init__(self, dataset, in_channels, num_classes=C.NUM_CLASSES, use_cbam=False,
                 pretrained=True, freeze_encoder=False):
        super().__init__()
        self.encoder = ResNet18Encoder(dataset, in_channels, pretrained)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.freeze_encoder = freeze_encoder
        self.up1 = Up(512, 256, 256, use_cbam)
        self.up2 = Up(256, 128, 128, use_cbam)
        self.up3 = Up(128, 64, 64, use_cbam)
        self.up4 = Up(64, 64, 32, use_cbam)
        self.final_up = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.outc = nn.Conv2d(32, num_classes, 1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:  # BN encoder tetap pakai statistik pretrained
            self.encoder.eval()
        return self

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        d = self.up4(self.up3(self.up2(self.up1(x4, x3), x2), x1), x0)
        return self.outc(self.final_up(d))


def build_model(cfg) -> nn.Module:
    in_ch = C.DATASETS[cfg.dataset]["in_channels"]
    if cfg.arch == "unet":
        return UNet(in_ch, use_cbam=cfg.cbam)
    if cfg.arch == "resnet18":
        return ResNetUNet(cfg.dataset, in_ch, use_cbam=cfg.cbam,
                          pretrained=cfg.pretrained, freeze_encoder=cfg.freeze_encoder)
    if cfg.arch == "deeplabv3plus":
        if cfg.cbam or cfg.freeze_encoder:
            raise ValueError("deeplabv3plus tidak mendukung cbam / freeze_encoder.")
        import segmentation_models_pytorch as smp
        return smp.DeepLabV3Plus(encoder_name="resnet50",
                                 encoder_weights="imagenet" if cfg.pretrained else None,
                                 in_channels=in_ch, classes=C.NUM_CLASSES)
    if cfg.arch == "segformer":
        if cfg.cbam or cfg.freeze_encoder:
            raise ValueError("segformer tidak mendukung cbam / freeze_encoder.")
        import segmentation_models_pytorch as smp
        return smp.Segformer(encoder_name="mit_b2",
                             encoder_weights="imagenet" if cfg.pretrained else None,
                             in_channels=in_ch, classes=C.NUM_CLASSES)
    raise ValueError(f"Arsitektur tidak dikenal: {cfg.arch}")
