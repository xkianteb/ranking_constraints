import sys
import numpy as np
from tqdm import tqdm
from loguru import logger

#from config import dataConfig as dc
#from utils import exposure_fn, dcg_util_fn
from ranking_constraints.config import create_parser, parse_args, parse_dataset_args

if __name__ == "__main__":
    #dataset = sys.argv[1]
    #dc.init_config(dataset)
    configs = create_parser()
    configs = parse_dataset_args(configs)
    configs = parse_args(configs)

    #EXP_DIR, N = dc.EXP_DIR, dc.N
    EXP_DIR, N = configs['EXP_DIR'], configs['N']
    logger.info(f"Processing {dataset} in {EXP_DIR} with N = {N}...")
    R = np.load(EXP_DIR / "test.npy")[:, :N]
    T = R.shape[0]
    exposure = np.zeros(N)
    v = exposure_fn(np.arange(N) + 1)
    u = dcg_util_fn(np.arange(N) + 1)
    dcg = []
    for tau, r in tqdm(enumerate(R)):
        ranking = list(reversed(list(np.argsort(r))))
        rank = np.zeros(N, dtype=np.int32)
        rank[ranking] = np.arange(N)
        exposure += v[rank]
        dcg.append(u.dot(r[ranking]))
    np.savetxt(EXP_DIR / f"exposure_estimate_{N}.txt", exposure, fmt="%s")
    print("Utility w/o intervention: {}".format(np.mean(dcg)))
