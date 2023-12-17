import random

import torch

import cvxpy as cp
import numpy as np
from loguru import logger
from sinkhorn_knopp import sinkhorn_knopp as skp
from collections import defaultdict
from constraint_ranking.utils import (
    dcg_util_fn,
    exposure_fn,
    get_train_relevance,
    get_test_relevance,
)

TOLERANCE = np.finfo(float).eps * 10.0


class BaseController(object):
    def __init__(self, configs):
        self.T, self.N = configs.T, configs.N
        self.M = configs.M
        self.m = configs.M.shape[0]
        self.delta = configs.delta
        self.u = dcg_util_fn(np.arange(configs.N) + 1, configs.DATASET)
        self.v = exposure_fn(np.arange(configs.N) + 1, configs.DATASET)
        self.C = getattr(configs, "c", [1.0] * configs.M.shape[0])

        # CP class variables
        self.realtime_delta = configs.delta
        self.realtime_T = configs.T
        self.delta_ = configs.delta
        self.T_ = configs.T

    def get_observation(self, ranking):
        rank = np.zeros(self.N, dtype=np.int32)
        rank[ranking] = np.arange(self.N)
        return self.v[rank]

    def get_utility(self, ranking, r):
        return self.u.dot(r[ranking])

    # (state, r, tau)
    def get_action(self, state, r, tau):
        ranking_ = np.argsort(r)[::-1]
        return ranking_

    def act(self, state, r, tau):
        ranking = self.get_action(state, r, tau)
        obs = self.get_observation(ranking)
        util = self.get_utility(ranking, r)
        state += self.M.dot(obs)
        return state, util, obs, self.M.dot(obs)


class BasePController(BaseController):
    def __init__(self, configs):
        self.lmbda = configs.lmbda
        self.C = getattr(configs, "c", [1.0] * configs.M.shape[0])
        super().__init__(configs)

    def get_action(self, state, r, tau):
        # No intervention
        ranking_ = np.argsort(r)[::-1]
        obs_ = self.get_observation(ranking_)
        state_ = state + self.M.dot(obs_)
        setpoint = (tau + 1) / self.T * self.delta

        group_error = (
            self.lmbda * np.clip(setpoint - state_, 0, None) / np.sum(self.M, axis=1)
        )
        item_error = self.M.T.dot(group_error)

        s = r + item_error
        ranking = np.argsort(s)[::-1]
        return ranking


class BiostochasticController(BaseController):
    def solve(self):
        pass

    # TODO: remove the below function and enable the next function
    def act(self, state, r, tau):
        U = self.solve(state, r, tau)

        obs = U @ self.v
        util = r.T @ U @ self.u
        exposure = self.M @ U @ self.v
        state += self.M.dot(obs)
        return state, util, obs, exposure

    def get_action(self, state, r):
        U = self.solve(state, r)
        permutations, coefficients = bvn_decomp(U)
        permutation = random.choices(permutations, weights=coefficients)[0]
        _, ranking = np.nonzero(permutation.T)
        return ranking


class MyopicController(BiostochasticController):
    pass


class MC(MyopicController):
    def __init__(self, configs):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.hinge_min = getattr(configs, "hinge_min", 0.0)
        super().__init__(configs)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        U = cp.Variable((self.N, self.N), nonneg=True)
        C = cp.Constant(self.C)

        target_ = cp.Constant((tau + 1) / self.T * self.delta)
        state_ = cp.Constant(state) + M @ U @ v
        obj = r.T @ U @ u - C @ cp.maximum(
            [self.hinge_min] * self.M.shape[0], (target_ - state_)
        )

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones, ones @ U == ones]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)


class StationaryController(BiostochasticController):
    def __init__(self, configs):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.hinge_min = getattr(configs, "hinge_min", 0.0)
        self.beta = getattr(configs, "beta", 0.5)
        self.eps = getattr(configs, "eps", 1e-5)
        super().__init__(configs)

        self.C = np.array(getattr(configs, "c", np.ones(self.m)))

        self.lr = getattr(configs, "lr", 0.01)
        self.init = getattr(configs, "init", "one")
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        if self.init == "zero":
            self.lmbda = torch.zeros((self.m,), requires_grad=True)
        elif self.init == "one":
            self.lmbda = torch.ones((self.m,), requires_grad=True)
        self.lmbda_opt = torch.optim.Adam(
            [self.lmbda], lr=self.lr, betas=(self.beta, 0.999), eps=self.eps
        )
        self.delta_ = configs.delta

    def act(self, state, r, tau):
        U = self.solve(state, r, tau)

        obs = U @ self.v
        exposure = self.M @ U @ self.v
        util = r.T @ U @ self.u
        state += self.M.dot(obs)

        self.solve_lmbda(self.M @ obs, tau)
        return state, util, obs, exposure

    def solve_lmbda(self, state, tau):
        target_ = (1 / self.T) * self.delta_
        g = target_ - state

        # Gradient Ascent (-1)
        self.lmbda.sum().backward()
        self.lmbda.grad = torch.tensor(-1 * g).float()
        self.lmbda_opt.step()

        with torch.no_grad():
            min_ = torch.zeros(self.m)
            max_ = torch.tensor(self.C)
            self.lmbda.data = torch.clamp(self.lmbda.data, min=min_, max=max_).float()


class SC(StationaryController):
    def __init__(self, configs):
        super().__init__(configs)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        C = cp.Constant(self.C)
        U = cp.Variable((self.N, self.N), nonneg=True)

        # Clip lmabda
        lmbda = self.lmbda.detach().numpy()
        lmbda = cp.Constant(lmbda)

        state_ = M @ U @ v
        obj = (r.T @ U @ u) + (lmbda @ state_)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones, ones @ U == ones]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)


class SCHinge(StationaryController):
    def __init__(self, configs):
        super().__init__(configs)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        C = cp.Constant(self.C)
        U = cp.Variable((self.N, self.N), nonneg=True)

        # Clip lmabda
        lmbda = self.lmbda.detach().numpy()
        lmbda = cp.Constant(lmbda)

        state_ = M @ U @ v
        target_ = cp.Constant((1.0 / self.T) * self.delta_)

        obj = (r.T @ U @ u) - lmbda @ cp.maximum(
            [self.hinge_min] * self.M.shape[0], (target_ - state_)
        )

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones, ones @ U == ones]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)


class OracleController(BiostochasticController):
    def __init__(self, configs):
        super().__init__(configs)

        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.R = get_test_relevance(configs.N, configs.EXP_DIR)
        self.T_ = configs.T
        logger.info("Learning optimal U for each query...")

        Us = []
        constraints = []
        ones = cp.Constant(np.ones((self.N,)))
        for _ in range(self.T_):
            U = cp.Variable((self.N, self.N), nonneg=True)
            constraints.append(U @ ones == ones)
            constraints.append(ones @ U == ones)
            Us.append(U)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        R = cp.Constant(self.R)
        u = cp.Constant(self.u)
        target_ = cp.Constant(self.delta)
        C = cp.Constant(self.C)

        utility = 0
        state = 0
        for i in range(self.T_):
            utility += R[i].T @ Us[i] @ u
            state += M @ Us[i] @ v
        group_error_ = cp.maximum(np.zeros(self.M.shape[0]), (target_ - state))

        obj = utility - C @ group_error_
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        self.optimal_Us = [U.value for U in Us]

    def solve(self, state, r, tau):
        return self.sk.fit(self.optimal_Us[tau])


class PredictiveController(BiostochasticController):
    def __init__(self, configs):
        super().__init__(configs)
        self.samples_ = None

        self.B = getattr(configs, "b", 100)
        self.B_online = getattr(configs, "bo", self.B)

        self.relevance_type = getattr(configs, "relevance_type", None)
        self.hinge_min = getattr(configs, "hinge_min", 0.0)
        self.count = 0

        self.lr = getattr(configs, "lr", 0.01)
        self.init = getattr(configs, "init", "zero")
        self.beta = getattr(configs, "beta", 0.5)
        self.eps = getattr(configs, "eps", 1e-5)
        self.C = np.array(getattr(configs, "c", np.ones(self.m)))
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)

        self.shuffle_bootstraps = getattr(configs, "shuffle_bootstraps", "true")

        if self.relevance_type == "offline_relevance":
            self.R = get_train_relevance(configs.N, configs.EXP_DIR)
        elif self.relevance_type == "online_relevance":
            self.R = get_test_relevance(configs.N, configs.EXP_DIR)
        else:
            raise Exception("Unknown relevance type")
        self.NUM_OFFLINE = self.R.shape[0]
        self.num_samples_per_step = self.NUM_OFFLINE // self.T_
        self.online_to_sampled_offline = defaultdict(set)

        assert self.NUM_OFFLINE > 0
        self.queries = set()
        self.Us_offline = self.learn_offline()

        if self.init == "zero":
            self.lmbda = torch.zeros((self.B_online, self.m), requires_grad=True)
        elif self.init == "one":
            self.lmbda = torch.ones((self.B_online, self.m), requires_grad=True)
        self.lmbda_opt = torch.optim.Adam(
            [self.lmbda], lr=self.lr, betas=(self.beta, 0.999), eps=self.eps
        )

        logger.info("Init offline Us...")

    def bootstrap_online(self, tau):
        if self.shuffle_bootstraps == "true":
            bs = np.random.choice(list(self.queries), self.T_ - tau - 1, replace=False)
        else:
            bs = []
            for i in range(tau + 1, self.T_):
                idx = np.random.choice(list(self.online_to_sampled_offline[i]))
                bs.append(idx)
            bs = np.array(bs)
        return bs

    def learn_offline(self):
        def bootstrap_offline():
            if self.shuffle_bootstraps == "true":
                bs = np.random.choice(self.NUM_OFFLINE, self.T_, replace=False)
                self.queries |= set(list(bs))
            elif self.shuffle_bootstraps == "false":
                bs = []
                for i in range(self.T_):
                    idx = np.random.randint(
                        i * self.num_samples_per_step,
                        (i + 1) * self.num_samples_per_step,
                    )
                    self.online_to_sampled_offline[i].add(idx)
                    bs.append(idx)
                bs = np.array(bs)
            return bs

        samples = [bootstrap_offline() for _ in range(self.B)]
        Us = []
        constraints = []
        ones = cp.Constant(np.ones((self.N,)))

        for _ in range(self.NUM_OFFLINE):
            U = cp.Variable((self.N, self.N), nonneg=True)
            constraints.append(U @ ones == ones)
            constraints.append(ones @ U == ones)
            Us.append(U)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        R = cp.Constant(self.R)
        u = cp.Constant(self.u)
        target_ = cp.Constant(self.delta_)
        C = cp.Constant(self.C)

        utility = 0
        group_error_ = 0
        for i in range(self.B):
            state = 0
            for _, r_index in enumerate(samples[i]):
                state += M @ Us[r_index] @ v
                utility += R[r_index].T @ Us[r_index] @ u

            group_error_ += C @ cp.maximum([0.0] * self.M.shape[0], (target_ - state))

        obj = utility - group_error_
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return [U.value for U in Us]

    def act(self, state, r, tau):
        self.samples_ = [self.bootstrap_online(tau) for _ in range(self.B_online)]
        U = self.solve(state.copy(), r, tau)
        self.solve_lmbda(state.copy(), tau, U)

        obs = U @ self.v
        exposure = self.M @ U @ self.v
        util = r.T @ U @ self.u
        state += self.M.dot(obs)

        return state, util, obs, exposure

    def solve_lmbda(self, state, tau, U):
        samples = self.samples_.copy()
        target_ = self.delta_

        violation_ = []
        for i in range(self.B_online):
            state_ = state.copy()
            for _, r_index in enumerate(samples[i]):
                state_ += self.M @ self.Us_offline[r_index] @ self.v
            state_ += self.M @ U @ self.v
            violation_.append((target_ - state_))

        g = np.stack(violation_)

        # Gradient Ascent (-1)
        self.lmbda.sum().backward()
        self.lmbda.grad = torch.tensor(-1 * g).float()
        self.lmbda_opt.step()

        with torch.no_grad():
            min_ = torch.zeros(self.m)
            max_ = torch.tensor(self.C)
            self.lmbda.data = torch.clamp(self.lmbda.data, min=min_, max=max_).float()


class PC(PredictiveController):
    def __init__(self, configs):
        super().__init__(configs)
        assert self.relevance_type == "offline_relevance"
        assert self.B >= 1

    def solve(self, state, r, tau):
        U = cp.Variable((self.N, self.N), nonneg=True)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)

        lmbda = self.lmbda.detach().numpy()
        lmbda = cp.Constant(lmbda)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [
            U @ ones == ones,
            ones @ U == ones,
        ]

        utility = r.T @ U @ u
        state_ = M @ U @ v

        exposure = 0
        for i in range(self.B_online):
            exposure += lmbda[i] @ state_

        obj = utility + exposure / self.B_online
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)


class PCHinge(PredictiveController):
    def __init__(self, configs):
        super().__init__(configs)
        assert self.relevance_type == "offline_relevance"
        assert self.B >= 1

    def solve(self, state, r, tau):
        samples = self.samples_.copy()

        U = cp.Variable((self.N, self.N), nonneg=True)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        R = cp.Constant(self.R)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        target_ = cp.Constant(self.delta_)

        # Clip lmabda
        lmbda = self.lmbda.detach().numpy()
        lmbda = cp.Constant(lmbda)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [
            U @ ones == ones,
            ones @ U == ones,
        ]

        utility = r.T @ U @ u

        exposure = 0
        for i in range(self.B_online):
            state_ = cp.Constant(state.copy())
            for _, r_index in enumerate(samples[0]):
                state_ += M @ cp.Constant(self.Us_offline[r_index]) @ v
            state_ += M @ U @ v
            exposure += lmbda[i] @ cp.maximum(
                [self.hinge_min] * self.M.shape[0], (target_ - state_)
            )

        obj = utility - exposure / self.B_online
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)
