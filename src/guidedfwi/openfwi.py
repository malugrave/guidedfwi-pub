"""
I/O utilities for the OpenFWI FlatVel-A benchmark.

Expected directory layout (as shipped by OpenFWI, not modified here):

    <root>/model/model1.npy ... model60.npy   shape (500, 1, 70, 70), m/s
    <root>/data/data1.npy  ... data60.npy      shape (500, 5, 1000, 70)

Files 1-55 are reserved for training the diffusion prior; files 56-60 are
held out exclusively for test/inference. Sample i inside model{n}.npy
always corresponds to sample i inside data{n}.npy.

This module only replaces *data reading* for the original proprietary
velocity volumes (`np.fromfile(...)` on SEAM/SEG/BP binaries). It does not
touch the diffusion architecture, training loop, guidance loop or sampler.
"""

from pathlib import Path

import numpy as np
import torch

VELOCITY_SHAPE = (1, 70, 70)
SEISMIC_SHAPE = (5, 1000, 70)
SAMPLES_PER_FILE = 500

TRAIN_FILES = tuple(range(1, 56))   # model1..model55 / data1..data55
TEST_FILES = tuple(range(56, 61))   # model56..model60 / data56..data60


def _model_path(root, file_number):
    return Path(root) / "model" / f"model{file_number}.npy"


def _data_path(root, file_number):
    return Path(root) / "data" / f"data{file_number}.npy"


def _check_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"OpenFWI file not found: {path}")


def _check_finite(array, name):
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf values.")


def _print_diagnostics(context, files, velocity, seismic=None):
    print(f"[OpenFWI/FlatVel-A] {context}")
    print(f"  Arquivos selecionados      : {list(files)}")
    print(f"  Numero total de amostras   : {velocity.shape[0]}")
    print(f"  Formato dos modelos        : {tuple(velocity.shape)}")
    print(f"  dtype (modelos)            : {velocity.dtype}")
    print(f"  Min/Max velocidade (m/s)   : {float(velocity.min()):.2f} / {float(velocity.max()):.2f}")
    if seismic is not None:
        print(f"  Formato dos dados sismicos : {tuple(seismic.shape)}")
        print(f"  dtype (dados sismicos)     : {seismic.dtype}")
        print(f"  Min/Max amplitude sismica  : {float(seismic.min()):.4g} / {float(seismic.max()):.4g}")


def global_index_to_file_sample(global_index, file_numbers, samples_per_file=SAMPLES_PER_FILE):
    """Map a global dataset index to (file_number, sample_index within that file).

    global_index = 0   -> (file_numbers[0], 0)
    global_index = 499 -> (file_numbers[0], 499)
    global_index = 500 -> (file_numbers[1], 0)
    """
    n_total = len(file_numbers) * samples_per_file
    if not 0 <= global_index < n_total:
        raise IndexError(f"global_index {global_index} out of range [0, {n_total})")
    file_position = global_index // samples_per_file
    sample_index = global_index % samples_per_file
    return file_numbers[file_position], sample_index


def load_velocity_file(root, file_number, mmap=True):
    """Load only model{file_number}.npy.

    Used for training the diffusion prior, which never needs the seismic
    gathers -- the corresponding data{file_number}.npy is never opened.
    """
    path = _model_path(root, file_number)
    _check_exists(path)

    velocity = np.load(path, mmap_mode="r" if mmap else None)
    if velocity.shape[1:] != VELOCITY_SHAPE:
        raise ValueError(f"{path} has per-sample shape {velocity.shape[1:]}, expected {VELOCITY_SHAPE}")

    velocity = np.array(velocity, dtype=np.float32)  # materialize just this one file
    _check_finite(velocity, str(path))
    return velocity


def build_training_velocity_tensor(root, file_numbers=TRAIN_FILES):
    """Concatenate the vp samples from `file_numbers` (default: the 55
    training files) into a single float32 tensor of shape (N, 1, 70, 70).

    Only the requested files are ever opened -- by default the 5 held-out
    test files (56-60) are never touched, and the seismic gathers are never
    read at all.
    """
    for n in file_numbers:
        if n in TEST_FILES:
            raise ValueError(
                f"model{n}.npy belongs to the held-out test range {TEST_FILES} "
                f"and must not be used for training."
            )

    chunks = [load_velocity_file(root, n, mmap=True) for n in file_numbers]
    velocity_all = np.concatenate(chunks, axis=0)
    _check_finite(velocity_all, "concatenated training velocity models")
    _print_diagnostics("treinamento do prior de difusao (apenas velocidade)", file_numbers, velocity_all)

    return torch.from_numpy(velocity_all).float()


def load_pair_file(root, file_number, mmap=True):
    """Load model{n}.npy and data{n}.npy together, verifying that the
    sample-by-sample correspondence between velocity and seismic data holds."""
    model_path, data_path = _model_path(root, file_number), _data_path(root, file_number)
    _check_exists(model_path)
    _check_exists(data_path)

    velocity = np.array(np.load(model_path, mmap_mode="r" if mmap else None), dtype=np.float32)
    seismic = np.array(np.load(data_path, mmap_mode="r" if mmap else None), dtype=np.float32)

    assert velocity.shape[0] == seismic.shape[0], (
        f"model{file_number}.npy has {velocity.shape[0]} samples but "
        f"data{file_number}.npy has {seismic.shape[0]}"
    )
    assert velocity.shape[1:] == VELOCITY_SHAPE, f"Unexpected velocity shape {velocity.shape}"
    assert seismic.shape[1:] == SEISMIC_SHAPE, f"Unexpected seismic shape {seismic.shape}"

    _check_finite(velocity, str(model_path))
    _check_finite(seismic, str(data_path))

    return velocity, seismic


def load_test_sample(root, test_file, test_sample):
    """Load a single (velocity, seismic) pair from the held-out test files (56-60).

    `velocity` (v_true) is intended only for evaluating the reconstruction;
    `seismic` (d_obs) is the recorded/observed data for that sample.
    """
    if test_file not in TEST_FILES:
        raise ValueError(
            f"test_file={test_file} is not in the held-out range {TEST_FILES}; "
            f"files 1-55 are reserved for training and must not be mixed in here."
        )
    if not (0 <= test_sample < SAMPLES_PER_FILE):
        raise IndexError(f"test_sample={test_sample} out of range [0, {SAMPLES_PER_FILE})")

    model_path, data_path = _model_path(root, test_file), _data_path(root, test_file)
    _check_exists(model_path)
    _check_exists(data_path)

    velocity_all = np.load(model_path, mmap_mode="r")
    seismic_all = np.load(data_path, mmap_mode="r")

    assert velocity_all.shape[0] == seismic_all.shape[0], (
        f"model{test_file}.npy and data{test_file}.npy have different sample counts"
    )
    assert velocity_all.shape[1:] == VELOCITY_SHAPE, f"Unexpected velocity shape {velocity_all.shape}"
    assert seismic_all.shape[1:] == SEISMIC_SHAPE, f"Unexpected seismic shape {seismic_all.shape}"

    v_true = np.array(velocity_all[test_sample], dtype=np.float32)
    d_obs = np.array(seismic_all[test_sample], dtype=np.float32)

    _check_finite(v_true, f"model{test_file}.npy[{test_sample}]")
    _check_finite(d_obs, f"data{test_file}.npy[{test_sample}]")

    _print_diagnostics(
        f"amostra de teste/inferencia (arquivo {test_file}, indice {test_sample})",
        [test_file], v_true[None], d_obs[None],
    )

    return {
        "velocity": torch.from_numpy(v_true).float(),
        "seismic": torch.from_numpy(d_obs).float(),
        "file_index": test_file,
        "sample_index": test_sample,
    }
