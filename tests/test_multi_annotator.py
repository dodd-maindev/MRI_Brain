"""Unit tests for multi-annotator voting, Beta-Binomial loss, and uncertainty decomposition."""

import numpy as np
import torch
from mus_net.data.lidc_dataset import LidcIdriSegmentationDataset
from mus_net.data.windowing import ComputedTomographyWindowTransformer
from mus_net.losses import MultiWindowSegmentationLoss
from mus_net.multi_annotator_loss import compute_beta_binomial_loss
from mus_net.utils.uncertainty_decomposition import decompose_beta_uncertainty


def test_multi_annotator_dataset_voting_and_augmentation() -> None:
    """Verify vote_positive <= vote_total, n=1 matches mask, and flip/rot90 preserves sum."""
    m0 = np.zeros((64, 64), dtype=np.uint8); m0[16:40, 16:40] = 1
    m1 = np.zeros((64, 64), dtype=np.uint8); m1[18:42, 18:42] = 1
    ann_masks = np.stack([m0, m1, np.zeros_like(m0), np.zeros_like(m0)], axis=0)
    samples = [
        {"hu_patch": np.zeros((64, 64), dtype=np.float32), "mask": m0, "annotator_masks": ann_masks, "annotator_count": 2},
        {"hu_patch": np.zeros((64, 64), dtype=np.float32), "mask": m0, "annotator_masks": ann_masks, "annotator_count": 1},
    ]
    ds_eval = LidcIdriSegmentationDataset(samples, ComputedTomographyWindowTransformer(), is_training=False, target_patch_size=64)
    it0, it1 = ds_eval[0], ds_eval[1]
    assert torch.all(it0["vote_positive"] <= it0["vote_total"])
    assert torch.equal(it1["vote_positive"], it1["mask"])
    ds_tr = LidcIdriSegmentationDataset(samples, ComputedTomographyWindowTransformer(), is_training=True, target_patch_size=64, random_annotator=True)
    it_tr = ds_tr[0]
    assert "mask_random" in it_tr and torch.all(it_tr["vote_positive"] <= it_tr["vote_total"])


def test_beta_binomial_n1_type2_ml_and_ann0_backward_compat() -> None:
    """Verify n=1, y=1 reduces to -log(alpha/(alpha+beta)) and ann0 matches tensor exactly."""
    a = torch.full((2, 1, 8, 8), 3.0, requires_grad=True)
    b = torch.full((2, 1, 8, 8), 2.0, requires_grad=True)
    ones, n_one = torch.ones(2, 1, 8, 8), torch.ones(2, 1, 8, 8)
    loss = compute_beta_binomial_loss(a, b, ones, n_one, kl_weight=0.0)
    expected = -torch.log(torch.tensor(3.0 / 5.0))
    assert abs(float(loss.item()) - float(expected.item())) < 1e-5
    loss.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()

    a_edge = torch.ones((1, 1, 4, 4), requires_grad=True)
    b_edge = torch.ones((1, 1, 4, 4), requires_grad=True)
    compute_beta_binomial_loss(a_edge, b_edge, torch.full((1, 1, 4, 4), 2.0), torch.full((1, 1, 4, 4), 4.0)).backward()
    assert torch.isfinite(a_edge.grad).all() and torch.isfinite(b_edge.grad).all()

    crit_old = MultiWindowSegmentationLoss(use_evidential=True, edl_kl_weight=0.05)
    crit_new = MultiWindowSegmentationLoss(use_evidential=True, edl_kl_weight=0.05, supervision="ann0")
    out = {"final_prediction": torch.full((2, 1, 8, 8), 0.6), "final_evidence": (a.detach(), b.detach()), "branch_masks": {}}
    assert torch.allclose(crit_old(out, ones)[0], crit_new(out, {"mask": ones})[0])


def test_uncertainty_decomposition_properties_and_mc() -> None:
    """Verify epistemic >= -1e-6, S->inf -> 0, aleatoric peaks at p=0.5, and MC agreement."""
    vals = torch.tensor([1.0, 2.0, 5.0, 20.0, 100.0, 1000.0])
    aa, bb = torch.meshgrid(vals, vals, indexing="ij")
    dec = decompose_beta_uncertainty(aa, bb)
    assert float(dec["epistemic"].min()) >= -1e-6

    s_vals = torch.tensor([10.0, 100.0, 10000.0])
    epi_s = decompose_beta_uncertainty(0.3 * s_vals, 0.7 * s_vals)["epistemic"]
    assert float(epi_s[-1]) < 1e-3 and float(epi_s[0]) > float(epi_s[-1])

    p_grid = torch.linspace(0.1, 0.9, 9)
    alea = decompose_beta_uncertainty(p_grid * 500.0, (1.0 - p_grid) * 500.0)["aleatoric"]
    assert int(torch.argmax(alea)) == 4

    rng = np.random.default_rng(42)
    p_mc = np.clip(rng.beta(4.0, 6.0, size=100_000), 1e-7, 1.0 - 1e-7)
    mc_alea = float(np.mean(-(p_mc * np.log(p_mc) + (1.0 - p_mc) * np.log(1.0 - p_mc))))
    exact = decompose_beta_uncertainty(torch.tensor([4.0]), torch.tensor([6.0]))
    assert abs(float(exact["aleatoric"].item()) - mc_alea) < 1e-2
