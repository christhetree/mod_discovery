import io
import logging
import os

import torch as tr
from PIL import Image
from matplotlib import pyplot as plt
from torch import tensor as T, nn
from tqdm import tqdm

import util
from curves import PiecewiseBezier2D
from mod_sig_distances import FirstDerivativeDistance, SecondDerivativeDistance
from paths import OUT_DIR, CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    seed = 73
    tr.manual_seed(seed)

    n_iter = 120
    # lr_acc = 1.01
    lr_acc = 1.00
    # lr_brake = 0.9
    lr_brake = 1.00

    l1_loss = nn.L1Loss()

    # loss_fn = ESRLoss()
    # loss_fn = l1_loss
    # loss_fn = nn.MSELoss()

    d1_loss = FirstDerivativeDistance(nn.L1Loss())
    d2_loss = SecondDerivativeDistance(nn.L1Loss())

    def modex_loss(y_hat: T, y) -> T:
        return l1_loss(y_hat, y) + 5.0 * d1_loss(y_hat, y) + 10.0 * d2_loss(y_hat, y)

    loss_fn = modex_loss

    # Larger batch size is worse because there are no batch-granular LR updates
    bs = 1
    n_frames = 1501

    x_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic/mod_sig_gen/mod_sig_gen__bezier_1d.yml")
    # x_mod_gen_path = os.path.join(CONFIGS_DIR, "synthetic/mod_sig_gen/mod_sig_gen__vital_curves.yml")
    x_mod_gen = util.load_class_from_yaml(x_mod_gen_path)
    curves = tr.stack([x_mod_gen(n_frames) for _ in range(bs)], dim=0)


    # ========================= Define hat hyperparams ==============================
    n_segments_hat = 12
    degree_hat = 3

    # bezier_module_hat = PiecewiseBezier(
    #     n_frames, n_segments_hat, degree_hat, is_c1_cont=False
    # )
    # si_logits = None
    # bezier_module_hat = PiecewiseBezierDiffSeg(n_frames, n_segments_hat, degree_hat)
    # si_logits = tr.rand(bs, n_segments_hat)
    # si_logits.requires_grad_()

    bezier_module_hat = PiecewiseBezier2D(
        n_frames, n_segments_hat, degree_hat, is_c1_cont=False
    )
    # ========================= Define hat hyperparams ==============================
    cp_logits_x = tr.rand(bs, n_segments_hat * degree_hat + 1)
    cp_logits_y = tr.rand(bs, n_segments_hat * degree_hat + 1)
    cp_logits_x.requires_grad_()
    cp_logits_y.requires_grad_()
    # cp_logits = tr.rand(bs, (n_segments_hat * degree_hat) + 1)
    # cp_logits.requires_grad_()

    loss_hist = []
    # curr_lr = 1e0
    curr_lr = 1e-1
    eps = 1e-5
    min_lr = eps
    curves_hat = None
    from models import Spectral2DCNN

    done = False
    frames = []

    for idx in tqdm(range(n_iter)):
        trigger_idx = 0
        if idx < trigger_idx:
            alpha = 1.0
        elif not done:
            alpha = 1.0 - ((idx - trigger_idx) / (n_iter - trigger_idx))
        cp_x = Spectral2DCNN.process_bezier_logits_x(cp_logits_x, n_segments_hat, degree_hat, alpha_linear=alpha)
        cp_y = Spectral2DCNN.process_bezier_logits_y(cp_logits_y, n_segments_hat, degree_hat, alpha_linear=alpha)
        cp = tr.stack([cp_x, cp_y], dim=-1)
        cp = cp.view(bs, 1, n_segments_hat, degree_hat + 1, 2)
        assert cp.ndim == 5
        x = bezier_module_hat(cp=cp, cp_are_logits=False)
        curves_hat = x.squeeze(1)

        loss = loss_fn(curves_hat, curves)
        loss_hist.append(loss)
        loss.backward()

        if idx > 0 and loss_hist[idx] - loss_hist[idx - 1] > -eps:
            curr_lr *= lr_brake
            done = True
            # break
        elif not done:
            with tr.no_grad():
                cp_logits_x -= curr_lr * cp_logits_x.grad * 0.2
                cp_logits_x.grad.zero_()
                cp_logits_y -= curr_lr * cp_logits_y.grad
                cp_logits_y.grad.zero_()
            curr_lr *= lr_acc
        log.info(f"loss: {loss.item():.6f}, lr: {curr_lr:.6f}")

        if curr_lr < min_lr:
            log.info(f"Reached min_lr: {min_lr}, final loss = {loss.item():.4f}")
            break

        if idx % 1 == 0:
            plt.plot(curves[0].detach().numpy(), label="target", color="black", linewidth=2.0, linestyle="--")
            plt.plot(curves_hat[0].detach().numpy(), label="hat", color="orange", linewidth=2.0)
            plt.xticks([])
            plt.yticks([])
            # plt.title(f"loss = {loss_hist[-1].item():.6f}, iter = {idx}")
            plt.ylim(-0.1, 1.1)
            plt.tight_layout()
            # plt.legend()
            # plt.show()
            # plt.savefig(os.path.join(OUT_DIR, f"curve_{idx:04d}.png"), bbox_inches="tight", dpi=300)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(Image.open(buf))
            plt.close()

    frames[0].save(os.path.join(OUT_DIR, f"curve__seed_{seed}.gif"), save_all=True, append_images=frames[1:], duration=50, loop=0)
