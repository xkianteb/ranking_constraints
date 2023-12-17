import random

import numpy as np
import yaml
from pathlib import Path

FULL_TO_SHORT = {
    "MC": "mc",
    "SC": "sc",
    "SCHinge": "schinge",
    "OracleController": "oracle",
    "PC": "pc",
    "PCHinge": "pchinge",
    "BaseController": "base",
}

SHORT_TO_FULL = {
    "mc": "MC",
    "sc": "SC",
    "schinge": "SCHinge",
    "oracle": "OracleController",
    "pc": "PC",
    "pchinge": "PCHinge",
    "base": "BaseController",
}


def exposure_fn(rk, DATASET):
    if DATASET == "toy":
        return np.array([1, 0.5, 0])
    elif DATASET == "movie_len_top_10":
        print("DATASET == 'movie_len_top_10'")
        e = 1 / rk
        e[16:] = 0
        return e
    else:
        return 1 / rk


def dcg_util_fn(rk, DATASET):
    if DATASET == "toy":
        return np.array([1, 0.2, 0])
    elif DATASET == "movie_len_top_10":
        dcg = 1 / np.log2(rk + 1)
        dcg[16:] = 0
        return dcg
    elif DATASET == "early_and_late@5":
        dcg = 1 / np.log2(rk + 1)
        dcg[5:] = 0
        return dcg
    else:
        dcg = 1 / np.log2(rk + 1)
        return dcg


def cost_fn(c, k, m):
    return c / np.power(2, k * np.arange(m))


def to_permutation_matrix(permutation):
    """
    Convert a permutation to a permutation matrix.
    """
    n = len(permutation)
    P = np.zeros((n, n))
    for i, j in enumerate(permutation):
        P[j, i] = 1
    return P

# In practice, we decompose the biostat matrix into a combination of permutaiton matrices
# def bvn_decomp(U):
#    U = tf.convert_to_tensor(U[np.newaxis, :, :].astype(np.float32))
#    p, c = bvn.bvn(U, 10)
#    return list(p.numpy()[0]), list(c.numpy()[0])


def get_train_relevance(N, EXP_DIR):
    D = np.load(EXP_DIR / "train.npy")[:, :N]
    return D


def get_test_relevance(N, EXP_DIR):
    R = np.load(EXP_DIR / "test.npy")[:, :N]
    return R


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_conf(configs):
    arg_dict = configs.__dict__

    filename = "dev.npy" if configs.dev else "test.npy"
    estimate = np.loadtxt(configs.EXP_DIR / f"exposure_estimate_{configs.N}.txt")
    R = np.load(configs.EXP_DIR / filename)[:, : configs.N]
    with open(Path(configs.conf), "r") as f:
        config = yaml.safe_load(f)

    groups, _ = config["groups"], config["targets"]
    if configs.targets is None:
        targets = config["targets"]
    else:
        arg_dict["targets"] = targets
    assert len(groups) == len(targets)

    M = np.zeros((len(groups), configs.N), dtype=np.int32)
    E = np.zeros((len(groups),), dtype=float)
    for gid, group in enumerate(groups):
        for idx in group:
            M[gid, idx] = 1
            E[gid] += estimate[idx]
    delta = np.multiply(E, np.array(targets))
    print(f"E: {E}")
    print(f"delta: {delta}")
    T = R.shape[0]
    m = M.shape[0]

    arg_dict["T"] = T
    arg_dict["R"] = R
    arg_dict["M"] = M
    arg_dict["delta"] = delta
    arg_dict["m"] = m
    return configs
