"""
Description:
Compute reconstruction-quality metrics for a
DiffusionModel_2D_FWIGuidedSamplingSynthetic run on OpenFWI FlatVel-A:

  - Velocity-model space: relative L2 error ||v_pred - v_true|| / ||v_true||
    for every posterior sample, the posterior mean, and the initial
    (Gaussian-smoothed) model used to start guidance -- as a baseline.
  - Data space: relative L2 error ||d_pred - d_true|| / ||d_true||, where
    d_* comes from a fixed, deterministic (one source per grid column,
    non-simultaneous) Deepwave forward model of v_true vs. the posterior
    mean and the initial model. This uses a plain, reproducible acquisition
    rather than the random simultaneous-source batches used during
    guidance, since it's meant for a stable evaluation number, not training.

Reads args.json + samples.npy from the target run folder (both already
written by DiffusionModel_2D_FWIGuidedSamplingSynthetic.py) and re-derives
v_true/v_init the same way that script does, including the OpenFWI
orientation fix.

Run as:
python Metrics_FlatVelA.py --results_folder ../results/DiffusionModel_2D_FWIGuidedSamplingSynthetic_<timestamp>
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import deepwave
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.append('../src/')
from guidedfwi.openfwi import load_test_sample
from guidedfwi.plots import plot_modulus

# Comment out if LateX is not present
plt.style.use('../asset/plots.mplstyle')

# Acquisition constants mirroring the hardcoded values inside
# p_sample_loop_with_fwi_guidance (DiffusionModel_2D_FWIGuidedSamplingSynthetic.py).
# These aren't stored in args.json since they aren't CLI-exposed there.
NT = 2000
DT = 0.004


def relative_l2(pred, true):
    """||pred - true||_2 / ||true||_2"""
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def forward_model_all_shots(vp, dz, dx, freq, device):
    """
    Single-shot (one source per grid column, no simultaneous-source
    batching) Deepwave forward model of the full line, used only for
    evaluation here -- a fixed, deterministic acquisition, unlike the
    random simultaneous-source batches used during FWI guidance.

    Parameters
    ----------
    vp : ndarray (nz, nx)
        Velocity model in m/s.

    Returns
    -------
    data : ndarray (nx, nt, nx)
        Shot gathers (one per source position).
    """
    nz, nx = vp.shape
    vp_t = torch.from_numpy(vp).float().to(device)

    source_locations = torch.zeros(nx, 1, 2)
    source_locations[:, 0, 0] = dx
    source_locations[:, 0, 1] = torch.arange(nx) * dx
    source_locations[:, :, 0] /= dz
    source_locations[:, :, 1] /= dx

    receiver_locations = torch.zeros(nx, nx, 2)
    receiver_locations[:, :, 0] = dx
    receiver_locations[:, :, 1] = torch.arange(nx) * dx
    receiver_locations[:, :, 0] /= dz
    receiver_locations[:, :, 1] /= dx

    wavelet = deepwave.wavelets.ricker(freq, NT, DT, 1 / freq).to(device)
    source_amplitudes = wavelet.repeat(nx, 1, 1)

    data = deepwave.scalar(
        vp_t,
        grid_spacing=[dz, dx],
        dt=DT,
        source_amplitudes=source_amplitudes.to(device),
        source_locations=source_locations.to(device),
        receiver_locations=receiver_locations.to(device),
        accuracy=8,
        pml_freq=freq,
        pml_width=[0, 10, 10, 10],
    )[-1]

    return data.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_folder", type=str, required=True,
        help="DiffusionModel_2D_FWIGuidedSamplingSynthetic_<timestamp> folder to read (must contain args.json and samples.npy).",
    )
    parser.add_argument(
        "--compute_data_metric", type=str, default='y',
        help="Also forward-model true/init/mean with Deepwave and report the data-space relative L2 error (y/n).",
    )
    parser.add_argument(
        "--per_sample_data_metric", type=str, default='n',
        help="Also compute the data-space metric for every individual posterior sample, not just the mean (y/n). Expensive: one extra Deepwave run per sample.",
    )
    parser.add_argument(
        "--data_frequency", type=float, default=None,
        help="Override the source frequency used for the data-space metric (default: the --frequency the run itself used).",
    )
    args = parser.parse_args()

    with open(os.path.join(args.results_folder, "args.json")) as f:
        run_args = json.load(f)

    if run_args.get("velocity_type") != "flatvel_a":
        raise ValueError(
            f"This script only supports velocity_type='flatvel_a' runs (got "
            f"'{run_args.get('velocity_type')}'). Extend it if you need other velocity types."
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ##################################################################
    # Ground truth, initial model, and posterior samples
    ##################################################################

    openfwi_sample = load_test_sample(run_args["openfwi_root"], run_args["test_file"], run_args["test_sample"])
    # DiffusionModel_2D_FWIGuidedSamplingSynthetic.py's flatvel_a branch
    # pre-transposes vp_raw, but the shared pipeline downstream applies its
    # own .T when building the vp_true *tensor* -- the two cancel out, so
    # the actual vp_true used by the network (and saved into samples.npy)
    # is the OpenFWI array with no net transform. Do NOT transpose here.
    v_true = openfwi_sample["velocity"].numpy()[0]

    sigma = run_args.get("sigma", [10, 10])
    v_init = gaussian_filter(v_true, sigma)

    samples_all = np.load(os.path.join(args.results_folder, "samples.npy"))
    vp_samples = samples_all[:, 0]  # (num_samples, nz, nx), physical m/s (vp channel)

    if vp_samples.shape[1:] != v_true.shape:
        raise ValueError(
            f"samples.npy per-sample shape {vp_samples.shape[1:]} does not match "
            f"v_true shape {v_true.shape} (can happen if --resize_model was used)."
        )

    vp_mean = vp_samples.mean(axis=0)

    ##################################################################
    # Velocity-space relative L2 error
    ##################################################################

    vel_error_init = relative_l2(v_init, v_true)
    vel_error_mean = relative_l2(vp_mean, v_true)
    vel_error_samples = [relative_l2(vp_samples[i], v_true) for i in range(vp_samples.shape[0])]

    print(f"Relative L2 error (velocity) - initial model : {vel_error_init:.4f}")
    print(f"Relative L2 error (velocity) - posterior mean: {vel_error_mean:.4f}")
    print(f"Relative L2 error (velocity) - per sample    : {[f'{e:.4f}' for e in vel_error_samples]}")

    metrics = {
        "results_folder": args.results_folder,
        "test_file": run_args["test_file"],
        "test_sample": run_args["test_sample"],
        "num_samples": vp_samples.shape[0],
        "velocity_relative_l2": {
            "initial_model": vel_error_init,
            "posterior_mean": vel_error_mean,
            "per_sample": vel_error_samples,
        },
    }

    ##################################################################
    # Data-space relative L2 error (optional, forward-modelling based)
    ##################################################################

    data_error_samples = None

    if args.compute_data_metric == 'y':
        freq = args.data_frequency if args.data_frequency is not None else run_args["frequency"]
        dz, dx = run_args["openfwi_dz"], run_args["openfwi_dx"]

        d_true = forward_model_all_shots(v_true, dz, dx, freq, device)
        d_init = forward_model_all_shots(v_init, dz, dx, freq, device)
        d_mean = forward_model_all_shots(vp_mean, dz, dx, freq, device)

        data_error_init = relative_l2(d_init, d_true)
        data_error_mean = relative_l2(d_mean, d_true)

        print(f"Relative L2 error (data) - initial model : {data_error_init:.4f}")
        print(f"Relative L2 error (data) - posterior mean: {data_error_mean:.4f}")

        metrics["data_relative_l2"] = {
            "initial_model": data_error_init,
            "posterior_mean": data_error_mean,
        }

        if args.per_sample_data_metric == 'y':
            data_error_samples = []
            for i in range(vp_samples.shape[0]):
                d_i = forward_model_all_shots(vp_samples[i], dz, dx, freq, device)
                data_error_samples.append(relative_l2(d_i, d_true))
            print(f"Relative L2 error (data) - per sample     : {[f'{e:.4f}' for e in data_error_samples]}")
            metrics["data_relative_l2"]["per_sample"] = data_error_samples

    ##################################################################
    # Save metrics (json + csv)
    ##################################################################

    with open(os.path.join(args.results_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(args.results_folder, "metrics.csv"), "w") as f:
        f.write("sample,velocity_relative_l2,data_relative_l2\n")
        for i, e in enumerate(vel_error_samples):
            d = data_error_samples[i] if data_error_samples is not None else ""
            f.write(f"{i},{e},{d}\n")
        data_mean = metrics.get("data_relative_l2", {}).get("posterior_mean", "")
        data_init = metrics.get("data_relative_l2", {}).get("initial_model", "")
        f.write(f"mean,{vel_error_mean},{data_mean}\n")
        f.write(f"initial,{vel_error_init},{data_init}\n")

    ##################################################################
    # Plots
    ##################################################################

    # Velocity relative L2 error per sample, with mean/initial reference lines
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(np.arange(len(vel_error_samples)), vel_error_samples, color='#0C5DA5', label='Posterior samples')
    ax.axhline(vel_error_mean, color='#FF2C00', linestyle='--', label='Posterior mean')
    ax.axhline(vel_error_init, color='#474747', linestyle=':', label='Initial model')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Relative L2 error (velocity)')
    ax.legend()
    fig.savefig(os.path.join(args.results_folder, 'metrics_velocity_error.pdf'))
    plt.close(fig)

    # Data relative L2 error (initial vs. mean, and per-sample if computed)
    if args.compute_data_metric == 'y':
        labels = ['Initial', 'Posterior mean']
        values = [data_error_init, data_error_mean]
        if data_error_samples is not None:
            labels += [f'Sample {i}' for i in range(len(data_error_samples))]
            values += data_error_samples

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(labels, values, color='#00B945')
        ax.set_ylabel('Relative L2 error (data)')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        fig.savefig(os.path.join(args.results_folder, 'metrics_data_error.pdf'))
        plt.close(fig)

    # Spatial error map of the posterior mean vs. the true model
    extent = [
        0, v_true.shape[1] * run_args["openfwi_dx"] / 1e3, v_true.shape[0] * run_args["openfwi_dz"] / 1e3, 0
    ]
    plot_modulus(
        np.abs(vp_mean - v_true) / 1e3,
        aspect='auto', cmap='Reds',
        extent=extent,
        fig_name=os.path.join(args.results_folder, 'metrics_error_map.pdf'),
    )

    print(f"Saved metrics.json, metrics.csv and plots to {args.results_folder}")


if __name__ == "__main__":
    main()
