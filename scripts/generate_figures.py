import logging
import os

import torch as tr
from matplotlib import pyplot as plt

from cli import CustomLightningCLI
from paths import OUT_DIR, CONFIGS_DIR
from synth_modules import WavetableOsc

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    # ckpt_folder = MODELS_DIR
    ckpt_folder = os.path.join(OUT_DIR, "ckpts__wandb_runs")

    # wt_dir = os.path.join(WAVETABLES_DIR, "ableton")
    # wt_names = [f[:-3] for f in os.listdir(wt_dir) if f.endswith(".pt")]
    # filtered_wt_names = []
    # for wt_name in wt_names:
    #     if any(wt_name.startswith(n) for n in CONTINUOUS_ABLETON_WTS):
    #         filtered_wt_names.append(wt_name)
    # wt_paths = [os.path.join(wt_dir, f"{wt_name}.pt") for wt_name in filtered_wt_names]
    # wt_paths = sorted(wt_paths)
    # for idx, wt_path in enumerate(wt_paths):
    #     log.info(f"{idx}: {wt_path}")
    # wt_idx = 9

    model_paths = [
        # ("Frame", "synthetic_2/test_vital__ase__lfo_frame.yml", f"train__ase__lfo_frame/acid_ddsp_2/version_{wt_idx}/checkpoints/mss__frame_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
        # ("LPF", "synthetic_2/test_vital__ase__lfo_frame_8_hz.yml", f"train__ase__lfo_frame_8_hz/acid_ddsp_2/version_{wt_idx}/checkpoints/mss__frame_8_hz_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
        # ("Spline", "synthetic_2/test_vital__ase__lfo.yml", f"train__ase__lfo/acid_ddsp_2/version_{wt_idx}/checkpoints/mss__s24d3D_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),

        # ("Frame", "synthetic_2/train__ase__lfo_frame.yml", f"train__ase__lfo_frame/acid_ddsp_2/version_{wt_idx}/checkpoints/mss__frame_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
        # ("LPF", "synthetic_2/train__ase__lfo_frame_8_hz.yml", f"train__ase__lfo_frame_8_hz/acid_ddsp_2/version_{wt_idx}/checkpoints/mss__frame_8_hz_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
        # ("Spline", "synthetic_2/train__ase__lfo.yml", f"train__ase__lfo/acid_ddsp_2/version_{wt_idx}/checkpoints/mss__s24d3D_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),

        ("Frame", "serum_2/train__ase__sm_frame.yml", "mss__frame__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
        ("LPF", "serum_2/train__ase__sm_frame_8_hz.yml", "mss__frame_8_hz__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
        ("Spline", "serum_2/train__ase__sm.yml", "mss__s24d3D__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),

        # ("Frame", "serum_2/train__ase__sm_shan_frame.yml", "mss__shan_frame__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
        # ("LPF", "serum_2/train__ase__sm_shan_frame_8_hz.yml", "mss__shan_frame_8_hz__sm_16_1024__serum__BA_both_lfo_10__epoch_28_step_986.ckpt"),
        # ("Spline", "serum_2/train__ase__sm_shan.yml", "mss__shan_s24d3D__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),

        # ("Frame", "serum_2/train__ase__sm_ddsp_frame.yml", "mss__ddsp_frame__sm__serum__BA_both_lfo_10__epoch_29_step_510.ckpt"),
        # ("LPF", "serum_2/train__ase__sm_ddsp_frame_8_hz.yml", "mss__ddsp_frame_8_hz__sm__serum__BA_both_lfo_10__epoch_29_step_510.ckpt"),
        # ("Spline", "serum_2/train__ase__sm_ddsp.yml", "mss__ddsp_s24d3D__sm__serum__BA_both_lfo_10__epoch_29_step_510.ckpt"),
    ]

    # wt_path = wt_paths[wt_idx]
    wt_path = None
    log.info(f"Wavetable path: {wt_path}")
    seed = 42
    dataloader = None
    models = []

    for name, config_name, ckpt_name in model_paths:
        config_path = os.path.join(CONFIGS_DIR, config_name)
        ckpt_path = os.path.join(ckpt_folder, ckpt_name)
        cli = CustomLightningCLI(
            args=["-c", config_path, "--seed_everything", str(seed)],
            trainer_defaults=CustomLightningCLI.make_trainer_defaults(save_dir=OUT_DIR),
            run=False,
        )
        if dataloader is None:
            dataloader = cli.datamodule.test_dataloader()

        if wt_path is not None:
            wt = tr.load(wt_path, weights_only=True)
            wt_name = os.path.basename(wt_path)[: -len(".pt")]
            cli.model.wt_name = wt_name
            synth = cli.model.synth
            assert not synth.add_synth_module.is_trainable
            sr = synth.ac.sr
            wt_module = WavetableOsc(sr=sr, wt=wt, is_trainable=False)
            synth.register_module("add_synth_module", wt_module)
            synth_hat = cli.model.synth_hat
            # TODO(cm): WTF
            # if isinstance(synth_hat.add_synth_module, WavetableOsc) and not synth_hat.add_synth_module.is_trainable:
            wt_module_hat = WavetableOsc(sr=sr, wt=wt, is_trainable=False)
            synth_hat.register_module("add_synth_module", wt_module_hat)

        state_dict = tr.load(ckpt_path, map_location="cpu")["state_dict"]
        cli.model.load_state_dict(state_dict)
        cli.model.eval()
        models.append((name, cli.model))

    lfo_prefix = "sub"
    lfo_hat_suffix = ""
    # lfo_hat_suffix = "_inv"
    lfo_name = f"{lfo_prefix}_lfo"
    lfo_hat_name = f"{lfo_prefix}_lfo_hat{lfo_hat_suffix}"
    font_size = 32
    ax_font_size = 24

    log.info(f"Number of batches in dataloader: {len(dataloader)}")
    for batch_idx, batch in enumerate(dataloader):
        n_cols = len(models)
        fig, axs = plt.subplots(
            1, 1 + n_cols, figsize=((n_cols + 1) * 5, 5), squeeze=False, dpi=300
        )
        # fig, axs = plt.subplots(1, n_cols, figsize=((n_cols) * 5, 5), squeeze=False, dpi=300)
        spec = None

        with tr.no_grad():
            for idx, (name, model) in enumerate(models):
                out_dict = model.test_step(batch, stage="test")
                log.info(f"Loss: {out_dict['loss']}")
                if spec is None:
                    spec = out_dict["log_spec_x"]
                    ax = axs[0][0]
                    ax.imshow(
                        spec[0],
                        extent=[0, spec.size(2), 0, spec.size(1)],
                        aspect=spec.size(2) / spec.size(1),
                        origin="lower",
                        cmap="magma_r",
                        interpolation="none",
                    )
                    # ax.set_title("Audio Mel. Spec.", fontsize=font_size)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # ax.set_xlabel("Time (frames)", fontsize=ax_font_size)
                    # ax.set_ylabel("Mel. Frequency Bins", fontsize=ax_font_size)
                    ax.set_ylabel("Mod. Synth", fontsize=font_size)
                    # ax.set_ylabel("Shan et al.", fontsize=font_size)
                    # ax.set_ylabel("Engel et al.", fontsize=font_size)

                ax = axs[0][idx + 1]
                # ax = axs[0][idx]

                # lfo = out_dict[lfo_name]
                # ax.plot(lfo[0], label=lfo_name, color="black", linewidth=2.0, linestyle="--")
                # lfo_hat = out_dict[lfo_hat_name]
                # ax.plot(lfo_hat[0], label=lfo_hat_name, color="blue", linewidth=2.0)
                # ax.set_title(f"{name}", fontsize=font_size)
                # ax.set_ylim(-0.05, 1.05)
                # ax.set_xticks([])
                # ax.set_yticks([])
                # l1 = out_dict[f"{lfo_name}__l1"]
                # l1_d1 = out_dict[f"{lfo_name}__l1_d1"]
                # pcc = out_dict[f"{lfo_name}__pcc"]
                # fd = out_dict[f"{lfo_name}__fd"]
                # ax.set_xlabel(
                #     f"L1: {l1 * 10:.2f}  ∇L1: {l1_d1 * 1000:.2f}\nPCC: {pcc:.2f}  FD: {fd * 10:.2f}",
                #     fontsize=ax_font_size,
                # )

                for lfo_name in ["env_hat", "add_lfo_hat", "sub_lfo_hat"]:
                    lfo_hat = out_dict[lfo_name]
                    if "sub" in lfo_name:
                        # if "add" in lfo_name or "sub" in lfo_name:
                        lfo_hat = lfo_hat[0]
                        lfo_hat = (lfo_hat - lfo_hat.min()) / (
                            lfo_hat.max() - lfo_hat.min()
                        )
                        lfo_hat = lfo_hat.unsqueeze(0)

                    if "add" in lfo_name:
                        color = "red"
                    elif "sub" in lfo_name:
                        color = "blue"
                    else:
                        color = "orange"

                    ax.plot(lfo_hat[0], color=color, linewidth=2.0)
                    # ax.set_title(f"{name}", fontsize=font_size)
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_xticks([])
                    ax.set_yticks([])

                # if idx == 0:
                # ax.set_ylabel("Modulation Amplitude", fontsize=ax_font_size)
                # ax.set_ylabel("Mod. Synth", fontsize=font_size)
                # ax.set_ylabel("Shan et al.", fontsize=font_size)
                # ax.set_ylabel("Engel et al.", fontsize=font_size)

        # ax.legend()
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(OUT_DIR, f"{batch_idx}.pdf"), bbox_inches="tight")
        # fig.savefig(os.path.join(OUT_DIR, f"{batch_idx}__no_spec.pdf"), bbox_inches="tight")
