import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import itertools
from matplotlib.ticker import ScalarFormatter
import json


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"


yScalarFormatter = ScalarFormatterClass(useMathText=True)


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def main() -> int:
    ctrls = ["oracle", "base", "mc", "sc", "pc"]
    ctrl_to_best_params = json.load(open("ctrl_to_best_params.json"))
    c = "1._1."
    fig, axes = plt.subplots(
        nrows=2, ncols=5, figsize=(13, 4), sharex=True, sharey="row"
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    hor = 6
    data = np.zeros((4, 6))
    data[0] = np.ones(hor)
    data[1] = np.ones(hor) * 0.95
    data[2] = [-1, 0.5, -1, 0.5, -1, 0.5]
    data[3] = [0.5, -1, 0.5, -1, 0.5, -1]
    x = np.arange(400)

    # Base
    dfs = [pd.read_pickle(ctrl_to_best_params[ctrl][c]) for ctrl in ctrls]
    target1 = [dfs[0]["delta"].to_list()[0][0]] * 400
    target2 = [dfs[0]["delta"].to_list()[0][1]] * 400
    for i in range(5):
        ctrl = ctrls[i]
        axes[0][i].set_title(ctrl, fontsize="small")
        df = dfs[i]
        # Base DCG
        axes[0][i].plot(x, dfs[0]["Mean DCG"], color="lightgrey", linestyle="--")
        axes[0][i].plot(x, df["Mean DCG"], color="black")
        axes[0][i].axis("on")
        axes[0][i].set_ylabel("utility", fontsize="small")

        axes[1][i].plot(
            x,
            target1,
            marker="o",
            mfc="w",
            color="lightgrey",
            linestyle="--",
            label="category-1 target",
            markevery=50,
        )
        axes[1][i].plot(
            x,
            target2,
            marker="X",
            mfc="w",
            color="lightgrey",
            linestyle="--",
            label="category-2 target",
            markevery=50,
        )

        estimate = df["state"].to_list()
        estimate1 = [estimate[j][0] for j in x]
        estimate2 = [estimate[j][1] for j in x]
        axes[1][i].plot(
            x,
            estimate1,
            marker="o",
            mfc="w",
            color="black",
            label="category-1 estimate",
            markevery=50,
        )
        axes[1][i].plot(
            x,
            estimate2,
            marker="X",
            mfc="w",
            color="black",
            label="category-2 estimate",
            markevery=50,
        )

        axes[1][i].set_ylim([-1, 200])
        axes[1][i].set_yticks([0, 50, 100, 150, 200])
        axes[1][i].set_ylabel("exposure", fontsize="small")
        axes[1][i].set_xticks([0, 150, 300, 450])
        labels = [item.get_text() for item in axes[1][i].get_xticklabels()]
        axes[1][i].set_xlabel("queries", fontsize="small")

    lines_labels = [
        ax.get_legend_handles_labels() for idx, ax in enumerate(fig.axes) if idx == 5
    ]
    for idx, ax in enumerate(fig.axes):
        print(idx, ax.get_legend_handles_labels())
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        fancybox=False,
        ncol=4,
        frameon=False,
        fontsize="small",
    )

    fig.align_ylabels(axes[:, 0])

    fig.tight_layout()
    fig.savefig(f"counter_example.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # next section explains the use of sys.exit
