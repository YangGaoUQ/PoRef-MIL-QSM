"""Full 256x256 networks on random inputs; this is not a clinical experiment."""
import torch
from poref.model import PoRefNet, BagRelabelNet
from poref.experiment import focal_loss


def main():
    torch.set_num_threads(2)
    torch.manual_seed(37)
    images = torch.randn(2, 1, 1, 256, 256)
    diagnosis = PoRefNet(slice_batch_size=1)
    logits = diagnosis(images)
    assert logits.shape == (2,)
    focal_loss(logits, torch.tensor([0., 1.])).backward()
    assert all(torch.isfinite(p.grad).all() for p in diagnosis.parameters() if p.grad is not None)
    assert diagnosis.encoder.branches[0].block1.middle.conv1x1.weight.grad.abs().sum() > 0
    diagnosis.zero_grad(set_to_none=True)
    relabeler = BagRelabelNet.from_diagnosis(diagnosis, axial_chunk_size=512)
    result = relabeler.forward_bag(images)
    loss = focal_loss(torch.logit(result["bag_probability"].clamp(1e-6, 1-1e-6)), torch.tensor(1.))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in relabeler.parameters() if p.grad is not None)
    assert relabeler.encoder.branches[0].block1.upper.block1.self_attn.in_proj_weight.grad is not None
    assert relabeler.axial.self_attn.in_proj_weight.grad is not None
    assert relabeler.classifier.mlp[0].weight.grad is not None
    diagnosis.eval()
    with torch.no_grad():
        first = diagnosis(images)[0]
        changed = images.clone()
        changed[1] += 8
        torch.testing.assert_close(first, diagnosis(changed)[0])
    print("PASS full diagnostic and relabel networks: 256x256 forward/backward, finite gradients, independent test slices")


if __name__ == "__main__":
    main()
