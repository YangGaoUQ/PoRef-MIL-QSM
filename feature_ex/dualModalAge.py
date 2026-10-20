import torch
import torch.nn as nn

from feature_ex.feature import FeatureExtraction
from feature_ex.crossAttention import CrossAttentionBlock
from feature_ex.film import AgeFiLM
from classifier.classifier_model import ClassifierMLP


class DualModalClassifierNet(nn.Module):
    """
    双模态 + 年龄协变量分类网络

    输入:
        qsm: [B,1,256,256]
        t1 : [B,1,256,256]
        age: [B] 或 [B,1]

    输出:
        out: [B,1]
    """

    def __init__(self, age_position="post"):
        super().__init__()
        if age_position not in ("pre", "post"):
            raise ValueError("age_position must be pre or post")
        self.age_position = age_position

        self.qsm_feature = FeatureExtraction()
        self.t1_feature = FeatureExtraction()

        self.shared_cross_attention = CrossAttentionBlock()

        self.qsm_norm = nn.LayerNorm(256)
        self.t1_norm = nn.LayerNorm(256)

        self.final_cross_attention = CrossAttentionBlock()
        self.final_norm = nn.LayerNorm(256)

        # One age-conditioned FiLM, applied either before or after FE.
        self.film = AgeFiLM(2 if age_position == "pre" else 256)

        self.classifier = ClassifierMLP()

    def forward(self, qsm, t1, age):
        """
        qsm:
            [B,1,256,256]

        t1:
            [B,1,256,256]

        age:
            [B] 或 [B,1]
        """

        if self.age_position == "pre":
            modulated = self.film(torch.cat((qsm, t1), dim=1), age)
            qsm, t1 = modulated[:, :1], modulated[:, 1:]
        # 1. 单模态特征提取
        qsm_feat = self.qsm_feature(qsm)
        t1_feat = self.t1_feature(t1)

        # 2. 共享Cross-Attention特征
        shared_feat = self.shared_cross_attention(
            qsm_feat,
            t1_feat,
            t1_feat
        )

        # 3. shared_feat 同时回流给两个模态
        qsm_shared = self.qsm_norm(
            qsm_feat + shared_feat
        )

        t1_shared = self.t1_norm(
            t1_feat + shared_feat
        )

        # 4. 最终Cross-Attention融合
        fusion_feat = self.final_cross_attention(
            qsm_shared,
            t1_shared,
            t1_shared
        )

        fusion_feat = self.final_norm(
            fusion_feat + qsm_shared
        )

        # 5. Method2：Feature后、分类前加入年龄协变量
        if self.age_position == "post":
            fusion_feat = self.film(fusion_feat.transpose(1, 2), age).transpose(1, 2)

        # 6. 分类
        out = self.classifier(fusion_feat)

        return out
