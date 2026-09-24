"""Three-class models. Checkpoints reconstruct offline from saved configs."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

LABELS = {0: 'other', 1: 'rice', 2: 'weed'}

class DinoSegmenter(nn.Module):
    def __init__(self, backbone, frozen=True):
        super().__init__()
        self.backbone = backbone
        self.frozen = frozen
        width = backbone.config.hidden_size
        self.projections = nn.ModuleList([nn.Conv2d(width, 64, 1) for _ in range(4)])
        self.decoder = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1),
                                     nn.GroupNorm(8, 128), nn.GELU(),
                                     nn.Conv2d(128, 64, 3, padding=1),
                                     nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 3, 1))
        self.backbone.requires_grad_(not frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, pixel_values):
        b, _, h, w = pixel_values.shape
        p = self.backbone.config.patch_size
        if h % p or w % p:
            raise ValueError('DINO input dimensions must be divisible by patch_size')
        # no_grad, rather than inference_mode: decoder needs these features for backward.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            states = self.backbone(pixel_values=pixel_values, output_hidden_states=True).hidden_states
        n = self.backbone.config.num_hidden_layers
        indices = [max(1, n // 4), max(1, n // 2), max(1, 3*n // 4), n]
        skip = 1 + self.backbone.config.num_register_tokens
        features = []
        for idx, project in zip(indices, self.projections):
            tokens = states[idx][:, skip:]
            grid = tokens.transpose(1, 2).reshape(b, -1, h // p, w // p)
            features.append(project(grid))
        x = torch.cat(features, dim=1)
        x = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=False)
        return self.decoder(x)


def build_model(kind, pretrained=True, config=None, frozen=True):
    if kind == 'student':
        net = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None,
            num_classes=3, aux_loss=False)
        return net, {'kind': kind}
    from transformers import AutoConfig, AutoModel, SegformerConfig, SegformerForSemanticSegmentation
    if kind == 'dino':
        model_id = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
        if config is None:
            backbone = AutoModel.from_pretrained(model_id)
        else:
            cfg = dict(config)
            model_type = cfg.pop('model_type')
            backbone = AutoModel.from_config(AutoConfig.for_model(model_type, **cfg))
        net = DinoSegmenter(backbone, frozen=frozen)
        return net, {'kind': kind, 'config': backbone.config.to_dict(), 'frozen': frozen,
                     'source': model_id}
    if kind == 'segformer':
        model_id = 'nvidia/segformer-b2-finetuned-ade-512-512'
        if config is None:
            net = SegformerForSemanticSegmentation.from_pretrained(
                model_id, num_labels=3, id2label=LABELS,
                label2id={v:k for k,v in LABELS.items()}, ignore_mismatched_sizes=True)
        else:
            net = SegformerForSemanticSegmentation(SegformerConfig.from_dict(config))
        return net, {'kind': kind, 'config': net.config.to_dict(), 'source': model_id}
    raise ValueError(kind)


def logits_of(model, images, size=None):
    outputs = model(images)
    if isinstance(outputs, dict):
        logits = outputs['out'] if 'out' in outputs else outputs['logits']
    elif isinstance(outputs, torch.Tensor):
        logits = outputs
    else:
        logits = outputs.logits
    return F.interpolate(logits.float(), size=size or images.shape[-2:],
                         mode='bilinear', align_corners=False)


def load_model(path, device='cpu'):
    # Only load checkpoints you generated or otherwise trust.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    spec = ckpt['model_spec']
    net, _ = build_model(spec['kind'], pretrained=False, config=spec.get('config'),
                         frozen=spec.get('frozen', True))
    net.load_state_dict(ckpt['model'])
    return net.to(device), ckpt
