import numpy as np
from tqdm import tqdm
import time


class Simulator(object):
    def __init__(self, configs, controller, test=False, tqdm_disable=True):
        self.controller = controller
        self.state = None  # Exposure for every constraint
        self.obs = None  # Exposure for every doc
        self.utility = None
        self.m = configs.m
        self.N = configs.N
        self.MAX_UTIL = configs.MAX_UTIL
        self.tqdm_disable = tqdm_disable
        self.C = np.array(configs.c)
        self.delta = configs.delta
        self.intermediate_metrics = []
        self.previous_state = 0.0
        self.states = []

    def init_state(self):
        self.state = np.zeros(self.m)
        self.obs = np.zeros(self.N)
        self.utility = []
        self.lambda_ = []

    def simulate(self, R):
        self.init_state()
        self.st = time.time()

        for tau, r in tqdm(enumerate(R), disable=self.tqdm_disable):
            self.step(r, tau)
            self.et = time.time()

            metrics = self.get_metrics(self.delta, tau=tau)
            self.intermediate_metrics.append(metrics)
            self.states.append(self.state.copy())
        return self.state, self.utility, self.obs

    def step(self, r, tau):
        self.state, util, obs, exposure = self.controller.act(self.state, r, tau)
        self.utility.append(util)
        self.obs = self.obs + obs
        self.exposure = exposure

    def get_metrics(self, delta, tau=None):
        unsatisfaction = np.clip(delta - self.state, 0, None) / delta
        unsatisfaction[unsatisfaction < 1e-7] = 0.0

        weighted = (
            np.sum(self.utility.copy())
            - np.clip(delta - self.state, 0, None).copy().sum()
        )
        weighted_objective = (
            np.sum(self.utility.copy())
            - self.C @ np.clip(delta - self.state, 0, None).copy()
        )
        return {
            "reward_minus_cost": weighted,
            "Weighted Objective": weighted_objective,
            "Mean DCG": np.mean(self.utility.copy()),
            "Sum DCG": np.sum(self.utility.copy()),
            "Sacrifice Ratio": (self.MAX_UTIL - np.mean(self.utility.copy()))
            / self.MAX_UTIL,
            "Unsatisfaction Mean": np.mean(unsatisfaction.copy()),
            "Unsatisfaction Std": np.std(unsatisfaction.copy()),
            "Unsatisfaction Max": np.amax(unsatisfaction.copy()),
            "Unsatisfaction Sum": np.clip(delta - self.state, 0, None).copy().sum(),
            "Unsatisfaction Diff": np.clip(delta - self.state, 0, None).copy(),
            "C * Unsatisfaction Sum": self.C
            @ np.clip(delta - self.state, 0, None).copy(),
            "delta": delta.copy(),
            "state": self.state.copy(),
            "elapsed_time": (self.et - self.st),
            "tau": 0 if tau == None else (tau + 1) / self.controller.T * delta.copy(),
            "per_step_exposure": self.exposure.copy(),
        }

    def get_states(self):
        return self.states
