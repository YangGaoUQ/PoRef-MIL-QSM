"""Figure (a) diagnosis and a separate figure (c) bag relabeler."""
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
            # Preserve the fused vector: one component per row, broadcast over
            # the pooled feature axis so that max pooling returns that vector.
            return self.mlp(torch.cat([feat.mean(1) for feat in features], dim=-1)).unsqueeze(-1).expand(-1, -1, 256)
        fused = features[0]
        for i, other in enumerate(features[1:]):
            shared = self.cross[i](fused, other, other)
            left, right = self.norm(fused + shared), self.norm(other + shared)
            fused = self.norm(self.final_cross[i](left, right, right) + left)
        return fused


class PoRefNet(nn.Module):
    """Slice diagnosis only: FEB -> max pooling -> MLP. No Z attention."""
    def __init__(self, modalities=1, fusion="cross-attention", age_position="none", axial_chunk_size=256, slice_batch_size=4):
        super().__init__()
        self.encoder = MultimodalEncoder(modalities, fusion, slice_batch_size)
        self.classifier = ClassifierMLP()
        self.age_position = age_position
        self.film = AgeFiLM(modalities if age_position == "pre" else 256) if age_position != "none" else None

    def _classify(self, features, age):
        if self.age_position == "post":
            features = self.film(features.transpose(1, 2), age).transpose(1, 2)
        return self.classifier(features).squeeze(-1)

    def forward(self, images, age=None):
        # Slices remain independent in the diagnostic path.
        if self.film is not None and age is None:
            raise ValueError("Age is required by the selected FiLM experiment")
        repeated_age = None if age is None else age.reshape(1).expand(images.shape[0])
        if self.age_position == "pre":
            images = self.film(images.squeeze(2), repeated_age).unsqueeze(2)
        features = self.encoder(images)
        if isinstance(features, list):
            # Decision-level ablation averages per-modality probabilities.
            probabilities = torch.stack([self._classify(f, repeated_age).sigmoid() for f in features]).mean(0)
            return torch.logit(probabilities.clamp(1e-6, 1 - 1e-6))
        return self._classify(features, repeated_age)


def bag_outputs(logits):
    """One patient's raw scores feed BOTH branches of figure (c)."""
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("Expected a nonempty ordered slice logit vector")
    probabilities = logits.sigmoid()
    weights = logits.softmax(dim=0)
    patient_probability = (weights * probabilities).sum()
    return {"logits": logits, "slice_probs": probabilities,
            "attention_weights": weights, "bag_probability": patient_probability}


class BagRelabelNet(PoRefNet):
    """Fresh Z attention/head, initialized from the preceding diagnostic FEB."""
    def __init__(self, modalities=1, fusion="cross-attention", age_position="none",
                 axial_chunk_size=256, slice_batch_size=4):
        super().__init__(modalities, fusion, age_position, axial_chunk_size, slice_batch_size)
        self.axial = ZAxisSelfAttention(chunk_size=axial_chunk_size)

    @classmethod
    def from_diagnosis(cls, diagnosis, axial_chunk_size=256):
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.encoder = copy.deepcopy(diagnosis.encoder)
        model.encoder.requires_grad_(True)
        model.age_position = diagnosis.age_position
        model.film = copy.deepcopy(diagnosis.film)
        if model.film is not None:
            model.film.requires_grad_(True)
        model.axial = ZAxisSelfAttention(chunk_size=axial_chunk_size)
        model.classifier = ClassifierMLP()
        return model

    def _classify(self, features, age):
        if self.age_position == "post":
            features = self.film(features.transpose(1, 2), age).transpose(1, 2)
        features = self.axial(features.unsqueeze(0)).squeeze(0)
        return self.classifier(features).squeeze(-1)

    def forward_bag(self, images, age=None):
        return bag_outputs(self(images, age))
