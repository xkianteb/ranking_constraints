import sys

import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str)
parser.add_argument("--task", type=str)
args = parser.parse_args()


def main() -> int:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(3, 1),
        gridspec_kw={"hspace": 0.2, "wspace": 0.6},
        sharex=True,
    )

    markevery_dict = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 1,
    }
    marker_dict = {
        0: "D",
        1: "P",
        2: "s",
        3: "o",
        4: "P",
        5: "h",
        6: 6,
        7: 11,
        8: 10,
        9: 9,
        10: 7,
        11: "D",
    }

    label = {
        0: "oracle",
        2: "mc",
        3: "sc",
        7: "pc",
    }

    plot_label = {
        0: r"oracle",
        2: r"mc",
        3: r"sc",
        7: r"pc",
    }
    init_ca = ["zero"]

    for col in [0, 2, 3, 7]:
        x = [0.01, 0.1, 1.0, 10.0, 100.0]
        hyparams = [
            (hinge_min, task)
            for hinge_min in [0.0,]
            for task in ["movie_len_top_10", "shuffled_movie_len_top_10"]
        ]
        for (hinge_min, task) in hyparams:
            y_dcg = []
            y_weighted_objective = []
            y_unsat_mean = []
            y_unsat_sum = []
            y_time = []

            cs = ["0.01_0.01", "0.1_0.1", "1._1.", "10._10.", "100._100."]
            if task == "kuai":
                r_type = "online_relevance"
                lr_sweep = [0.1, 0.01, 0.001, 0.0001]
                b_sweep = [20, 50]
            elif task == "early_and_late":
                r_type = "online_relevance"
                lr_sweep = [100.0, 10.0, 1.0, 0.1, 0.01]
                b_sweep = [1]
            elif task == "zf_tv":
                r_type = "online_relevance"
                lr_sweep = [10.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
                b_sweep = [20, 40, 50]
                cs = ["0.01", "0.1", "1.", "10.", "100."]
            elif task == "movie_len_top_10" or task == "shuffled_movie_len_top_10":
                r_type = "online_relevance"
                lr_sweep = [10.0, 1.0, 0.1, 0.01]
                b_sweep = [20, 50]

            for c in cs:
                if label[col] == "mc":
                    args.algorithm = label[col]
                    params = [
                        ("0.0001", c, init_, b, bo, r_type, hinge_min, "None", "None")
                        for b in [1]
                        for bo in [1]
                        for init_ in ["None"]
                    ]
                elif label[col] == "oracle":
                    args.algorithm = label[col]
                    params = [
                        ("0.0001", c, init_, b, bo, r_type, "None", "None", "None")
                        for b in [1]
                        for bo in [1]
                        for init_ in ["None"]
                    ]
                elif label[col] in ["pc"]:
                    args.algorithm = label[col]
                    params = [
                        (lr, c, init_, b, bo, "offline_relevance", "None", beta, eps)
                        for b in b_sweep
                        for bo in b_sweep
                        for init_ in ["zero"]
                        for beta in [0.5]  # , 0.9, 0.98]
                        for eps in ["1e-05", "1e-08"]
                        for lr in lr_sweep
                    ]
                elif label[col] in ["sc"]:
                    args.algorithm = label[col]
                    params = [
                        (lr, c, init_, b, bo, r_type, "None", beta, eps)
                        for b in [1]
                        for bo in [1]
                        for init_ in init_ca  # ['zero']
                        for beta in [0.5, 0.9, 0.98]
                        for eps in ["1e-05", "1e-08"]
                        for lr in lr_sweep
                    ]
                else:
                    raise Exception(f"Unknow alg: {label[col]}")

                weighted_objective = -np.inf
                file = None

                for param in params:
                    (lr, c, init_, b, bo, r_type, hinge_min, beta, eps) = param
                    name = f"{args.algorithm}_" + "_".join([str(x) for x in param])
                    name = name.replace("/", "_")

                    try:
                        df = pd.read_pickle(f"results/{task}/{name}.pkl")
                    except:
                        print(f"[Error]: file results/{task}/{name}.pkl not found.")

                    if df.iloc[-1]["Weighted Objective"] > weighted_objective:
                        weighted_objective = df.iloc[-1]["Weighted Objective"]
                        file = f"results/{task}/{name}.pkl"

                if file == None:
                    print(f"{args.algorithm} -- unsatisfied")
                    for param in params:
                        (lr, c, init_, b, bo, r_type) = param
                        name = f"{args.algorithm}_" + "_".join([str(x) for x in param])
                        name = name.replace("/", "_")

                        df = pd.read_pickle(f"results/{task}/{name}.pkl")
                        if (
                            df.iloc[-1]["Unsatisfaction Mean"] > unsatisfaction_mean
                            and df.iloc[-1]["Unsatisfaction Std"] < unsatisfaction_std
                        ):
                            unsatisfaction_mean = df.iloc[-1]["Unsatisfaction Mean"]
                            unsatisfaction_std = df.iloc[-1]["Unsatisfaction Std"]
                            file = f"results/{task}/{name}.pkl"
                else:
                    print(
                        f'Best {args.algorithm} file: {file.split("/")[-1]} {weighted_objective}'
                    )

                df = pd.read_pickle(file)

                y_dcg.append(df["Sum DCG"].iloc[-1])
                y_weighted_objective.append(df["Weighted Objective"].iloc[-1])
                y_unsat_mean.append(df["Unsatisfaction Mean"].iloc[-1])
                y_unsat_sum.append(df["Unsatisfaction Sum"].iloc[-1])
                try:
                    y_time.append(df["elapsed_time"].iloc[-1] / 60.0)
                except:
                    import pdb

                    pdb.set_trace()

            markersize = 3
            # Plot DCG
            if task == "movie_len_top_10":
                ax_idx = 0
                axes[0].set_title(
                    f"last.fm (temporal)", fontsize="xx-small", rotation=0
                )
                axes[0].set_ylabel("objective", fontsize="xx-small")

            # Plot Unsatisfaction Sum
            elif task == "shuffled_movie_len_top_10":
                ax_idx = 1
                axes[1].set_title(
                    f"last.fm (non-temporal)", fontsize="xx-small", rotation=0
                )

            if ax_idx == 0:
                axes[ax_idx].plot(
                    x,
                    y_weighted_objective,
                    marker=marker_dict[col],
                    mfc="w",
                    markevery=markevery_dict[col],
                    markersize=markersize,
                    label=f"{plot_label[col]}",
                )
            else:
                axes[ax_idx].plot(
                    x,
                    y_weighted_objective,
                    marker=marker_dict[col],
                    mfc="w",
                    markevery=markevery_dict[col],
                    markersize=markersize,
                )
            axes[ax_idx].set_xscale("log")
            axes[ax_idx].tick_params(axis="both", which="major", labelsize="xx-small")
            axes[ax_idx].tick_params(axis="both", which="minor", labelsize="xx-small")
            axes[ax_idx].set_xlabel(r"cost vector $\phi$", fontsize="xx-small")

    for j in range(2):
        axes[j].margins(0.15)
        axes[j].set_xticks([0.01, 1, 100])
        axes[j].set_xscale("log")
        axes[j].tick_params(axis="both", which="major", labelsize="xx-small")
        axes[j].tick_params(axis="both", which="minor", labelsize="xx-small")

    lines_labels = [ax.get_legend_handles_labels() for idx, ax in enumerate(fig.axes)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        fancybox=False,
        ncol=2,
        frameon=False,
        fontsize="x-small",
    )

    fig.tight_layout()
    fig.savefig(
        f"main_temporal_plot.pdf", format="pdf", bbox_inches="tight", pad_inches=0
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())  # next section explains the use of sys.exit
