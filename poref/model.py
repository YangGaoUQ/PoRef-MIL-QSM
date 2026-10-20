"""Configurable raw-image branches, axial context, and one optional FiLM."""
import copy
from contextlib import contextmanager, nullcontext
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from feature_ex.feature import FeatureExtraction
from feature_ex.crossAttention import CrossAttentionBlock
from feature_ex.film import AgeFiLM
from classifier.classifier_model import ClassifierMLP
from relabel.selfAttention import ZAxisSelfAttention


@contextmanager
def preserve_bn_buffers(module):
    saved = [(buf, buf.clone()) for name, buf in module.named_buffers() if name.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in saved:
                buffer.copy_(value)


class MultimodalEncoder(nn.Module):
    def __init__(self, modalities, fusion, slice_batch_size=4):
        super().__init__()
        self.branches = nn.ModuleList(FeatureExtraction() for _ in range(modalities))
        self.fusion = fusion
        self.slice_batch_size = slice_batch_size
        cross_blocks = max(0, modalities - 1) if fusion == "cross-attention" else 0
        self.cross = nn.ModuleList(CrossAttentionBlock() for _ in range(cross_blocks))
        self.final_cross = nn.ModuleList(CrossAttentionBlock() for _ in range(cross_blocks))
        self.norm = nn.LayerNorm(256) if cross_blocks else nn.Identity()
        self.concat = nn.Linear(modalities * 256, 256) if fusion == "concatenation" else None
        self.mlp = nn.Sequential(nn.Linear(modalities * 256, 512), nn.GELU(), nn.Linear(512, 256)) if fusion == "mlp-level" else None

    def forward(self, images):
        features = []
        for i, branch in enumerate(self.branches):
            chunks = []
            for slices in images[:, i].split(self.slice_batch_size):
                # Recompute FE activations during backward to bound slice memory.
                if self.training and torch.is_grad_enabled():
                    chunks.append(checkpoint(branch, slices, use_reentrant=False,
                                             context_fn=lambda branch=branch: (nullcontext(), preserve_bn_buffers(branch))))
                else:
                    chunks.append(branch(slices))
            features.append(torch.cat(chunks))
        if len(features) == 1:
            return features[0]
        if self.fusion == "decision-level":
            return features
        if self.fusion == "concatenation":
            return self.concat(torch.cat(features, dim=-1))
        if self.fusion == "mlp-level":
            return self.mlp(torch.cat([feat.mean(1) for feat in features], dim=-1)).unsqueeze(1)
        fused = features[0]
        for i, other in enumerate(features[1:]):
            shared = self.cross[i](fused, other, other)
            left, right = self.norm(fused + shared), self.norm(other + shared)
            fused = self.norm(self.final_cross[i](left, right, right) + left)
        return fused


class PoRefNet(nn.Module):
    def __init__(self, modalities=1, fusion="cross-attention", age_position="none", axial_chunk_size=256, slice_batch_size=4):
        super().__init__()
        self.fe1 = MultimodalEncoder(modalities, fusion, slice_batch_size)
        self.fe2 = None  # Created at round 2, after FE1 has been trained.
        self.axial = ZAxisSelfAttention(chunk_size=axial_chunk_size)
        self.classifier = ClassifierMLP()
        self.age_position = age_position
        self.film = AgeFiLM(modalities if age_position == "pre" else 256) if age_position != "none" else None
        self.current_round = 1

    def start_round(self, number):
        self.current_round = number
        if number > 1:
            if self.fe2 is None:
                # Defined initialization: FE2 starts from trained FE1, then evolves.
                self.fe2 = copy.deepcopy(self.fe1)
                self.fe2.requires_grad_(True)
            self.fe1.requires_grad_(False)
            self.fe1.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.current_round > 1:
            self.fe1.eval()
        return self

    def _classify(self, features, age):
        if self.age_position == "post":
            features = self.film(features.transpose(1, 2), age).transpose(1, 2)
        features = self.axial(features.unsqueeze(0)).squeeze(0)
        return self.classifier(features).squeeze(-1)

    def forward(self, images, age=None):
        # One complete patient sequence per call, preserving all Z context.
        if self.film is not None and age is None:
            raise ValueError("Age is required by the selected FiLM experiment")
        repeated_age = None if age is None else age.reshape(1).expand(images.shape[0])
        if self.age_position == "pre":
            images = self.film(images.squeeze(2), repeated_age).unsqueeze(2)
        encoder = self.fe1 if self.current_round == 1 else self.fe2
        features = encoder(images)
        if isinstance(features, list):
            # Shared axial module/head; decision-level fusion averages probabilities.
            probabilities = torch.stack([self._classify(f, repeated_age).sigmoid() for f in features]).mean(0)
            return torch.logit(probabilities.clamp(1e-6, 1 - 1e-6))
        return self._classify(features, repeated_age)
