from pathlib import Path
import yaml

import argparse
from pathlib import Path

from ranking_constraints.utils import init_seed, load_conf, SHORT_TO_FULL


def create_parser():
    parser = argparse.ArgumentParser(
        description="Simulate the online ranking system"
        "under differenct control policies",
        argument_default=argparse.SUPPRESS
    )
    parser.add_argument("--conf", type=str, help="config file", required=True)
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--ctrl", type=str, choices=list(SHORT_TO_FULL.keys()), help="controller", required=True)
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--lmbda", type=float, help="lambda used in P-Controller")
    parser.add_argument("--gamma", type=float, help="gamma used in BP-Controller")
    parser.add_argument("--output_dir", type=str, help="Output of mterics")
    parser.add_argument("--c", nargs="+", help="C in weighted objective", type=float)
    parser.add_argument("--b", type=int, help="B offline in PC with bootstrap")
    parser.add_argument("--bo", type=int, help="B online in PC with bootstrap")
    parser.add_argument(
        "--relevance_type",
        type=str,
        choices=["offline_relevance", "online_relevance", "sequence_relevance"],
        default=None
    )
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--init", choices=["one", "zero"], help="init value")
    parser.add_argument("--metrics_file_name", type=str, default=None)
    parser.add_argument("--targets", nargs="+", type=float, default=None)
    parser.add_argument("--dev", action="store_true", default=False)
    parser.add_argument("--hinge_min", type=float)
    parser.add_argument("--shuffle_bootstraps", type=str)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--beta", type=float)
    configs = parser.parse_args()
    return configs 

def parse_dataset_args(configs):
    # Parse Dataset yaml
    if configs.dataset:
        with open(Path(__file__).parent.parent / "configs" / f"config.{configs.dataset}.yml", "r") as f:
            config = yaml.safe_load(f)
        arg_dict = configs.__dict__
        arg_dict['DATASET'] = configs.dataset
        arg_dict['DATA_DIR'] = Path(config["dataset"]["raw"]).expanduser()
        arg_dict['EXP_DIR'] = Path(config["dataset"]["exp"]).expanduser()
        arg_dict['EXP_DIR'].mkdir(exist_ok=True)
        arg_dict['N'] = config["sample"].get("N", None)
        arg_dict['K'] = config["sample"].get("K", None)
        arg_dict['MAX_UTIL'] = config.get("max_util", None)
        arg_dict['IS_TEMPORAL'] = config.get("is_temporal", False)
    return configs 


def parse_args(configs):
    seed = configs.seed if hasattr(configs, 'seed') else 0
    init_seed(seed)

    configs = load_conf(configs)
    arg_dict = configs.__dict__
    arg_dict["controller"] = SHORT_TO_FULL[configs.ctrl]
    return configs
