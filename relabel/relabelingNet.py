"""Figure (c) relabeling for a batch of single-modal patient bags."""
import copy
import torch
from torch import nn
from feature_ex.feature import FeatureExtraction
from relabel.selfAttention import ZAxisSelfAttention
from classifier.classifier_model import ClassifierMLP


class RelabelingNet(nn.Module):
    def __init__(self, threshold=0.5, feature_extractor=None):
        super().__init__()
        self.threshold = threshold
        self.feature_extractor = (copy.deepcopy(feature_extractor)
                                  if feature_extractor is not None else FeatureExtraction())
        self.feature_extractor.requires_grad_(True)
        self.z_attention = ZAxisSelfAttention()
        self.classifier = ClassifierMLP()

    def forward_bag(self, x):
        if x.ndim != 5 or x.shape[1] == 0:
            raise ValueError("Expected [patients,slices,1,256,256]")
        b, z, c, h, w = x.shape
        features = self.feature_extractor(x.reshape(b * z, c, h, w))
        features = self.z_attention(features.reshape(b, z, 256, 256))
        logits = self.classifier(features.reshape(b * z, 256, 256)).reshape(b, z)
        probabilities = logits.sigmoid()
        weights = logits.softmax(dim=1)
        return {"logits": logits, "slice_probs": probabilities,
                "attention_weights": weights,
                "bag_probability": (weights * probabilities).sum(dim=1),
                "pred_labels": (probabilities >= self.threshold).long()}

    def forward(self, x):
        result = self.forward_bag(x)
        return result["logits"].unsqueeze(-1), result["pred_labels"]

    forward_first_round = forward
    forward_later_round = forward
