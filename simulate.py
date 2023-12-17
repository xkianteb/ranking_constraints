import os

import argparse
from pathlib import Path
from loguru import logger
import pandas as pd

#from config import dataConfig as dc
#import controller as ctrl
#from simulator import Simulator
#from constraint_ranking.utils import init_seed, load_conf #, SHORT_TO_FULL

from constraint_ranking.config import create_parser, parse_args, parse_dataset_args
from constraint_ranking.simulator import Simulator
from constraint_ranking import controller as ctrl

if __name__ == "__main__":
    configs = create_parser()
    configs = parse_dataset_args(configs)
    configs = parse_args(configs)

    logger.level("DEBUG")
    logger.info(f"T={configs.T}, N={configs.N}, delta={configs.delta}")

    output_dir = f"{os.getcwd()}/results/{configs.dataset}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ctrl_class = getattr(ctrl, configs.controller)
    controller = ctrl_class(configs)
    simulator = Simulator(configs, controller, test=True, tqdm_disable=False)
    state, utility, obs = simulator.simulate(configs.R)

    intermediate_metrics = simulator.intermediate_metrics
    df = pd.DataFrame(intermediate_metrics)
    if hasattr(configs, 'metrics_file_name'):
        df.to_pickle(f'{output_dir}/{configs.metrics_file_name}.pkl')

    metrics = simulator.get_metrics(configs.delta)

    columns = list(metrics.keys())
    values = [str(metrics[c]) for c in columns]
    header = ",".join(columns)
    print(header)
    row = ",".join(values)
    print(row)
