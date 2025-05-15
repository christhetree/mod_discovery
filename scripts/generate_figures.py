import logging
import os
import time

import torch as tr
import torchaudio
from matplotlib import pyplot as plt

from cli import CustomLightningCLI
from paths import OUT_DIR, CONFIGS_DIR, WAVETABLES_DIR, MODELS_DIR
from synth_modules import WavetableOsc
from wavetables import CONTINUOUS_ABLETON_WTS

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    # ckpt_folder = MODELS_DIR
    ckpt_folder = os.path.join(OUT_DIR, "ckpts__wandb_runs__seed_42")
    # ckpt_folder = OUT_DIR

    wt_dir = os.path.join(WAVETABLES_DIR, "ableton")
    wt_names = [f[:-3] for f in os.listdir(wt_dir) if f.endswith(".pt")]
    filtered_wt_names = []
    for wt_name in wt_names:
        if any(wt_name.startswith(n) for n in CONTINUOUS_ABLETON_WTS):
            filtered_wt_names.append(wt_name)
    wt_paths = [os.path.join(wt_dir, f"{wt_name}.pt") for wt_name in filtered_wt_names]
    wt_paths = sorted(wt_paths)
    for idx, wt_path in enumerate(wt_paths):
        log.info(f"{idx}: {wt_path}")
    wt_idx = 0
    wt_idx_mapping = {
        0: 0,
        1: 1,
        2: 2,
        3: 4,
        4: 5,
        5: 6,
        6: 7,
        7: 9,
        8: 11,
        9: 12,
    }

    model_paths = [
        ("exp_3__mod_synth__frame", "serum/train__mod_discovery__mod_synth_frame.yml", "mss__frame__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
        ("exp_3__mod_synth__lpf", "serum/train__mod_discovery__mod_synth_lpf.yml", "mss__frame_8_hz__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
        ("exp_3__mod_synth__spline", "serum/train__mod_discovery__mod_synth_spline.yml", "mss__s24d3D__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
        ("exp_3__mod_synth__gran", "serum/train__mod_discovery__mod_synth_baseline_gran.yml", "mss__frame_gran__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
        ("exp_3__mod_synth__rand_spline", "serum/train__mod_discovery__mod_synth_baseline_rand_spline.yml", "mss__s24d3D__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),

        ("exp_3__shan_et_al__frame", "serum/train__mod_discovery__shan_et_al_frame.yml", "mss__shan_frame__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
        ("exp_3__shan_et_al__lpf", "serum/train__mod_discovery__shan_et_al_lpf.yml", "mss__shan_frame_8_hz__sm_16_1024__serum__BA_both_lfo_10__epoch_28_step_986.ckpt"),
        ("exp_3__shan_et_al__spline", "serum/train__mod_discovery__shan_et_al_spline.yml", "mss__shan_s24d3D__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
        ("exp_3__shan_et_al__gran", "serum/train__mod_discovery__shan_et_al_baseline_gran.yml", "mss__shan_frame_gran__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),
        ("exp_3__shan_et_al__rand_spline", "serum/train__mod_discovery__shan_et_al_baseline_rand_spline.yml", "mss__shan_s24d3D__sm_16_1024__serum__BA_both_lfo_10__epoch_29_step_1020.ckpt"),

        ("exp_3__engel_et_al__frame", "serum/train__mod_discovery__engel_et_al_frame.yml", "mss__ddsp_frame__sm__serum__BA_both_lfo_10__epoch_29_step_510.ckpt"),
        ("exp_3__engel_et_al__lpf", "serum/train__mod_discovery__engel_et_al_lpf.yml", "mss__ddsp_frame_8_hz__sm__serum__BA_both_lfo_10__epoch_29_step_510.ckpt"),
        ("exp_3__engel_et_al__spline", "serum/train__mod_discovery__engel_et_al_spline.yml", "mss__ddsp_s24d3D__sm__serum__BA_both_lfo_10__epoch_29_step_510.ckpt"),
        ("exp_3__engel_et_al__gran", "serum/train__mod_discovery__engel_et_al_baseline_gran.yml", "mss__ddsp_frame_gran__sm__serum__BA_both_lfo_10__epoch_29_step_510.ckpt"),
        ("exp_3__engel_et_al__rand_spline", "serum/train__mod_discovery__engel_et_al_baseline_rand_spline.yml", "mss__ddsp_s24d3D__sm__serum__BA_both_lfo_10__epoch_29_step_510.ckpt"),

        # ("exp_1__frame", f"synthetic/train__mod_extraction__frame.yml", f"train__mod_ex__frame/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__frame_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
        # ("exp_1__lpf", f"synthetic/train__mod_extraction__lpf.yml", f"train__mod_ex__lpf/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__frame_8_hz_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
        # ("exp_1__spline", f"synthetic/train__mod_extraction__spline.yml", f"train__mod_ex__spline/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__s24d3D_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
        # ("exp_1__rand_spline", f"synthetic/train__mod_extraction__baseline_rand_spline.yml", f"train__mod_ex__spline/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__s24d3D_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),

        # ("exp_1__frame", f"synthetic/test_vital_curves__mod_extraction__frame.yml", f"train__mod_ex__frame/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__frame_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
        # ("exp_1__lpf", f"synthetic/test_vital_curves__mod_extraction__lpf.yml", f"train__mod_ex__lpf/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__frame_8_hz_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
        # ("exp_1__spline", f"synthetic/test_vital_curves__mod_extraction__spline.yml", f"train__mod_ex__spline/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__s24d3D_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),
        # ("exp_1__rand_spline", f"synthetic/test_vital_curves__mod_extraction__baseline_rand_spline.yml", f"train__mod_ex__spline/acid_ddsp_2/version_{wt_idx_mapping[wt_idx]}/checkpoints/mss__s24d3D_nn__lfo__ase__ableton_13__epoch_29_step_600.ckpt"),

        # ("exp_2__oracle", f"synthetic/train__mod_discovery__baseline_oracle.yml", f"train__mod_discovery__baseline_oracle/mod_discovery/version_{wt_idx}/checkpoints/mss__oracle__sm_16_1024__ase__ableton_10__epoch_29_step_1200.ckpt"),
        # ("exp_2__frame", f"synthetic/train__mod_discovery__frame.yml", f"train__mod_discovery__frame/mod_discovery/version_{wt_idx}/checkpoints/mss__frame__sm_16_1024__ase__ableton_10__epoch_29_step_1200.ckpt"),
        # # ("exp_2__frame", f"synthetic/train__mod_discovery__frame.yml", f"train__mod_discovery__frame/mod_discovery/version_{wt_idx}/checkpoints/mss__frame__sm_16_1024__ase__ableton_10__epoch_28_step_1160.ckpt"),
        # ("exp_2__lpf", f"synthetic/train__mod_discovery__lpf.yml", f"train__mod_discovery__lpf/mod_discovery/version_{wt_idx}/checkpoints/mss__frame_8_hz__sm_16_1024__ase__ableton_10__epoch_29_step_1200.ckpt"),
        # # ("exp_2__lpf", f"synthetic/train__mod_discovery__lpf.yml", f"train__mod_discovery__lpf/mod_discovery/version_{wt_idx}/checkpoints/mss__frame_8_hz__sm_16_1024__ase__ableton_10__epoch_28_step_1160.ckpt"),
        # ("exp_2__spline", f"synthetic/train__mod_discovery__spline.yml", f"train__mod_discovery__spline/mod_discovery/version_{wt_idx}/checkpoints/mss__s24d3__sm_16_1024__ase__ableton_10__epoch_29_step_1200.ckpt"),
        # # ("exp_2__spline", f"synthetic/train__mod_discovery__spline.yml", f"train__mod_discovery__spline/mod_discovery/version_{wt_idx}/checkpoints/mss__s24d3__sm_16_1024__ase__ableton_10__epoch_27_step_1120.ckpt"),
        # ("exp_2__rand_spline", f"synthetic/train__mod_discovery__baseline_rand_spline.yml", f"train__mod_discovery__spline/mod_discovery/version_{wt_idx}/checkpoints/mss__s24d3__sm_16_1024__ase__ableton_10__epoch_29_step_1200.ckpt"),
        # # ("exp_2__rand_spline", f"synthetic/train__mod_discovery__baseline_rand_spline.yml", f"train__mod_discovery__spline/mod_discovery/version_{wt_idx}/checkpoints/mss__s24d3__sm_16_1024__ase__ableton_10__epoch_27_step_1120.ckpt"),
    ]

    # wt_path = wt_paths[wt_idx]
    wt_path = None
    log.info(f"Wavetable path: {wt_path}")
    # seed = 0
    seed = 42
    dataloader = None
    models = []

    for name, config_name, ckpt_name in model_paths:
        config_path = os.path.join(CONFIGS_DIR, config_name)
        ckpt_path = os.path.join(ckpt_folder, ckpt_name)
        log.info(f"Ckpt path: {ckpt_path}")
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
    sr = 48000
    dpi = 50
    n_batches = 1
    linewidth = 3.0

    log.info(f"Number of batches in dataloader: {len(dataloader)}")
    for batch_idx, batch in enumerate(dataloader):
        # if batch_idx == n_batches:
        #     break

        # n_cols = len(models)
        # fig, axs = plt.subplots(
        #     1, 1 + n_cols, figsize=((n_cols + 1) * 5, 5), squeeze=False, dpi=dpi
        # )
        # fig, axs = plt.subplots(1, n_cols, figsize=((n_cols) * 5, 5), squeeze=False, dpi=dpi)
        spec = None

        with tr.no_grad():
            for idx, (name, model) in enumerate(models):
                out_dict = model.test_step(batch, stage="test")
                # time.sleep(1.0)
                time.sleep(0.5)
                log.info(f"Loss: {out_dict['loss']}")
                if spec is None:
                    spec = out_dict["log_spec_x"]
                    assert spec.size(0) == 1
                    # ax = axs[0][0]
                    fig, ax = plt.subplots(1, 1, dpi=dpi)
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
                    # ax.set_ylabel("Mod. Synth", fontsize=font_size)
                    # ax.set_ylabel("Shan et al.", fontsize=font_size)
                    # ax.set_ylabel("Engel et al.", fontsize=font_size)
                    fig.tight_layout()
                    # fig.savefig(os.path.join(OUT_DIR, f"{name}__spec__{wt_idx}_{batch_idx}.png"), bbox_inches="tight")
                    # fig.savefig(os.path.join(OUT_DIR, f"{name}__spec__{batch_idx}.png"), bbox_inches="tight")
                    plt.close(fig)

                    # audio = out_dict["x"][0]
                    # torchaudio.save(os.path.join(OUT_DIR, f"{name}__audio_{wt_idx}_{batch_idx}.wav"), audio, sr)
                    # torchaudio.save(os.path.join(OUT_DIR, f"{name}__audio__{wt_idx}_{batch_idx}.mp3"), audio, sr)
                    # torchaudio.save(os.path.join(OUT_DIR, f"{name}__audio__{batch_idx}.mp3"), audio, sr)

                audio_hat = out_dict["x_hat"][0]
                # torchaudio.save(os.path.join(OUT_DIR, f"{name}__audio_hat_{wt_idx}_{batch_idx}.wav"), audio_hat, sr)
                # torchaudio.save(os.path.join(OUT_DIR, f"{name}__audio_hat__{wt_idx}_{batch_idx}.mp3"), audio_hat, sr)
                # torchaudio.save(os.path.join(OUT_DIR, f"{name}__audio_hat__{batch_idx}.mp3"), audio_hat, sr)

                fig, ax = plt.subplots(1, 1, dpi=dpi)
                spec_hat = out_dict["log_spec_x_hat"]
                ax.imshow(
                    spec_hat[0],
                    extent=[0, spec_hat.size(2), 0, spec_hat.size(1)],
                    aspect=spec_hat.size(2) / spec_hat.size(1),
                    origin="lower",
                    cmap="magma_r",
                    interpolation="none",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout()
                # fig.savefig(os.path.join(OUT_DIR, f"{name}__spec_hat__{wt_idx}_{batch_idx}.png"), bbox_inches="tight")
                # fig.savefig(os.path.join(OUT_DIR, f"{name}__spec_hat__{batch_idx}.png"), bbox_inches="tight")
                plt.close(fig)

                # ax = axs[0][idx + 1]
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

                # for lfo_name in ["env", "add_lfo", "sub_lfo"]:
                #     if "add" in lfo_name:
                #         color = "red"
                #     elif "sub" in lfo_name:
                #         color = "blue"
                #     else:
                #         color = "orange"
                #     lfo = out_dict[lfo_name][0]
                #     lfo_hat = out_dict[f"{lfo_name}_hat"][0]
                #     # lfo_hat_inv = out_dict[f"{lfo_name}_hat_inv"][0]
                #     # lfo_hat_inv_all = out_dict[f"{lfo_name}_hat_inv_all"][0]
                #     fig, ax = plt.subplots(1, 1, dpi=dpi)
                #     ax.plot(lfo, color="black", linewidth=linewidth, linestyle="--")
                #     ax.plot(lfo_hat, color=color, linewidth=linewidth)
                #     # ax.plot(lfo_hat_inv, color=color, linewidth=linewidth, linestyle=":")
                #     # ax.plot(lfo_hat_inv_all, color=color, linewidth=linewidth)
                #     ax.set_ylim(-0.05, 1.05)
                #     ax.set_xticks([])
                #     ax.set_yticks([])
                #     fig.tight_layout()
                #     fig.savefig(os.path.join(OUT_DIR, f"{name}__{lfo_name}__{wt_idx}_{batch_idx}.svg"), bbox_inches="tight")
                #     plt.close(fig)

                if not name.endswith("__gran"):
                    fig, ax = plt.subplots(1, 1, dpi=dpi)
                    for lfo_name in ["env_hat", "add_lfo_hat", "sub_lfo_hat"]:
                        lfo_hat = out_dict[lfo_name]
                        # if "sub" in lfo_name:
                        if "add" in lfo_name or "sub" in lfo_name:
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

                        ax.plot(lfo_hat[0], color=color, linewidth=linewidth)
                        # ax.set_title(f"{name}", fontsize=font_size)
                        ax.set_ylim(-0.05, 1.05)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    fig.tight_layout()
                    # fig.savefig(os.path.join(OUT_DIR, f"{name}__ase__{batch_idx}.svg"), bbox_inches="tight")
                    plt.close(fig)

                for ad_name in ["rms", "sc", "sb", "sf"]:
                    if f"{ad_name}__audio_feature" in out_dict:
                        audio_feature = out_dict[f"{ad_name}__audio_feature"]
                        audio_feature = audio_feature[:, 32:-32]
                        lfo_inv_all = out_dict[f"{ad_name}__lfo_inv_all"]
                        lfo_inv_all = lfo_inv_all[:, 32:-32]
                        fig, ax = plt.subplots(1, 1, dpi=dpi)
                        ax.plot(audio_feature[0], color="black", linewidth=linewidth, linestyle="--")
                        ax.plot(lfo_inv_all[0], color="magenta", linewidth=linewidth)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        fig.tight_layout()
                        # fig.savefig(os.path.join(OUT_DIR, f"{name}__{ad_name}__{batch_idx}.svg"), bbox_inches="tight")
                        fig.savefig(os.path.join(OUT_DIR, f"{name}__{ad_name}_lpf_trim_32__{batch_idx}.svg"), bbox_inches="tight")
                        plt.close(fig)

                # if idx == 0:
                # ax.set_ylabel("Modulation Amplitude", fontsize=ax_font_size)
                # ax.set_ylabel("Mod. Synth", fontsize=font_size)
                # ax.set_ylabel("Shan et al.", fontsize=font_size)
                # ax.set_ylabel("Engel et al.", fontsize=font_size)

        # ax.legend()
        # fig.tight_layout()
        # fig.show()
        # fig.savefig(os.path.join(OUT_DIR, f"{batch_idx}.pdf"), bbox_inches="tight")
        # fig.savefig(os.path.join(OUT_DIR, f"{batch_idx}__no_spec.pdf"), bbox_inches="tight")
