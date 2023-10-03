# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve as sparse_solve

# from .math_utils import *
from math_utils import *
import sys
from functools import partial


class Variable:
    def __init__(self, arg, param_slice):
        self.arg = arg
        self.param_slice = param_slice
        self.backup_arg = None

    def backup(self):
        self.backup_arg = self.arg.data()

    def restore(self):
        self.arg.set_data(self.backup_arg)


class BlockDB:
    def __init__(self):
        self.blocks = {}
        self.tot_data_size = 0
        self.tot_param_size = 0

    def add(self, arg):
        uid = id(arg)  # python object id
        if uid not in self.blocks:
            plen = param_len(arg)
            pslice = slice(self.tot_param_size, self.tot_param_size + plen)
            self.blocks[uid] = Variable(arg, pslice)
            self.tot_param_size += plen
            return pslice
        return self.blocks[uid].param_slice


class Cost:
    def __init__(self, func, args, param_slices, elim=True):
        self.func = func
        self.args = args
        self.param_slices = param_slices
        self.elim = elim


class Optimizer:
    MOODS = {
        "excited": "ヽ(^@^)ノ",
        "worried": " (~@~メ) ",
        "crying": " (╥﹏╥)  ",
        "shrug": "¯\\_(ツ)_/¯",
        "impatient": "(ﾟ@ﾟ;≡;ﾟ@ﾟ)",
    }

    def __init__(self):

        self.mult_damping = 3e-2
        # self.add_damping = 1e-4

        self.elims = BlockDB()
        self.survs = BlockDB()
        self.costs = []
        # amir added
        self.initial_cameras_locations = []

    # add a residual, first argument is Schur elimination
    def add_elim_cost(self, func, *args):
        param_slices = [
            self.elims.add(args[0]),
            *(self.survs.add(arg) for arg in args[1:]),
        ]
        self.costs.append(Cost(func, args, param_slices, elim=True))

    # add a residual, no Schur elimination parameter
    def add_cost(self, func, *args):
        param_slices = [self.survs.add(arg) for arg in args]
        self.costs.append(Cost(func, args, param_slices, elim=False))

    # Create matrices for blocks of the Hessian, to apply Schur elimination
    # trick. Schur elimination allows to project eliminate a subset of parameters
    # from the linear system, the math is explained below. From optimization point
    # of view we will be computing the step on cameras solving a reduced
    # optimization problem: we turn the quadratic model of the cost on
    # points+cameras to a quadratic model only on cameras, assuming the points are
    # going to be placed to their best possible position. After solving on cameras
    # we go back to points and compute the step on points.

    # This process is also called marginalization (this is probability jargon, if
    # the Hessian is the precision =covariance^{-1} matrix we marginalize away some
    # variables, and we get to work with the precision matrix of the subset - on
    # covariance side it's easy as we would just get a block of the matrix, since
    # we are working with the inverse we must perform a small computation to the
    # "inverse of the block of the inverse"). It can be useful in general to
    # understand how the constraints map to the subset of parameters, for instance
    # we can get a camera-camera edge marginalizing away the points and some neighboring
    # cameras (be aware that inaccuracies arise if we marginalize away stuff multiple
    # times... so this cannot be used to simplify a complex problem without any loss)
    #
    # We want to solve H * x = b, b is the gradient and H the (damped) Hessian. Splitting
    # the camera (X, surv) and point (P, elim) parts we want to solve
    #  |  A   B |   | deltaX |   | errorX |
    #  |        | * |        | = |        |
    #  | B^T  D |   | deltaP |   | errorP |
    #
    # The key observation is that D is block-diagonal (and in real optimizers it's
    # kept as a list of blocks), so it's very easy to invert, inverting individual
    # blocks (inversion of a dense matrix is O(n^3), so many small blocks are harmless
    # compared to a large dense matrix)
    #
    # The above equation can be rewritten as
    # | A * deltaX + B * deltaP = errorX
    # | B^T * deltaX + D * deltaP = errorP
    # Since we can invert D we can explicitate deltaP from the second:
    #   deltaP = D^{-1}(errorP - B^T * deltaX)
    # and replacing this value of deltaP in the first:
    #   A * deltaX + B * D^{-1}(errorP - B^T * deltaX) = errorX
    # The only unknown is deltaX! therefore we can collect it on one side:
    #   (A - B * D^{-1} * B^T) * deltaX = errorX - B * D^{-1} * errorP
    # The matrix S = (A - B * D^{-1} * B^T) is the Schur complement.
    def setup(self):
        self.deltaP = np.ndarray(self.elims.tot_param_size, dtype=np.float64)
        self.deltaX = np.ndarray(self.survs.tot_param_size, dtype=np.float64)
        self.errorP = np.ndarray(self.elims.tot_param_size, dtype=np.float64)
        self.errorX = np.ndarray(self.survs.tot_param_size, dtype=np.float64)
        self.A = lil_matrix((self.survs.tot_param_size, self.survs.tot_param_size))
        self.D = lil_matrix((self.elims.tot_param_size, self.elims.tot_param_size))
        self.B = lil_matrix((self.survs.tot_param_size, self.elims.tot_param_size))

        # Costs is everything we have to find. Num_points*Num_camersa
        for res in self.costs:
            for i, s_i in enumerate(res.param_slices):
                iSurv = i >= (1 if res.elim else 0)
                for j in range(i + 1):  # j = 0,1,..,i
                    s_j = res.param_slices[j]
                    jSurv = j >= (1 if res.elim else 0)
                    if jSurv:
                        # TODO: use solver just requiring lower half
                        self.A[s_i, s_j] = 1.0
                        self.A[s_j, s_i] = 1.0
                    elif iSurv:
                        self.B[s_i, s_j] = 1.0
                    else:
                        self.D[s_i, s_j] = 1.0
        self.A = self.A.tocsr()
        self.D = self.D.tocsr()
        self.invD = self.D.copy()
        self.B = self.B.tocsr()

    def compute_cost(self):

        cost = 0
        for res in self.costs:
            err = res.func(*res.args)
            if isinstance(err, Vec):
                cost += err.squared_norm() * 0.5  # cost contribution
            else:  # assume plain number
                cost += err * err * 0.5

        return cost

    def compute_gradient(self):
        self.errorP.fill(0)
        self.errorX.fill(0)

        cost = 0
        for res in self.costs:
            err, pders = Jet.compute_first_order(res.func, *res.args)
            cost += (err.T @ err)[0, 0] * 0.5  # cost contribution

            for i, s_i in enumerate(res.param_slices):
                iSurv = i >= (1 if res.elim else 0)
                # gradient contribution
                (self.errorX if iSurv else self.errorP)[s_i] += (
                    err.T @ pders[i]
                ).reshape(-1)

        return cost

    def compute_gradient_hessian(self, calc_hessian_flag=True):
        self.errorP.fill(0)
        self.errorX.fill(0)
        self.A.data.fill(0)
        self.B.data.fill(0)
        self.D.data.fill(0)

        cost = 0
        for res in self.costs:

            err, pders = Jet.compute_first_order(res.func, *res.args)

            cost += (err.T @ err)[0, 0] * 0.5  # cost contribution

            for i, s_i in enumerate(res.param_slices):
                iSurv = i >= (1 if res.elim else 0)
                # gradient contribution
                (self.errorX if iSurv else self.errorP)[s_i] += (
                    err.T @ pders[i]
                ).reshape(-1)
                if calc_hessian_flag is True:
                    for j in range(i + 1):  # j = 0,1,..,i
                        s_j = res.param_slices[j]
                        jSurv = j >= (1 if res.elim else 0)
                        block = pders[i].T @ pders[j]  # reshape(-1, 1)
                        if jSurv:
                            # TODO: use solver just requiring lower half
                            self.A[s_i, s_j] += block
                            if i != j:
                                self.A[s_j, s_i] += block.T
                        elif iSurv:
                            self.B[s_i, s_j] += block
                        else:
                            self.D[s_i, s_j] += block

        return cost

    def optimize(self):

        self.setup()

        print(
            "Baby-Optimizer is up and toddling its baby optimization-steps!\n"
            "\n"
            '   ,=""=,\n'
            "  c , _,{\n"
            "  /\\  @ )                 __\n"
            " /  ^~~^\\          <=.,__/ '}=\n"
            "(_/ ,, ,,)          \_ _>_/~\n"
            " ~\\_(/-\\)'-,_,_,_,-'(_)-(_)\n"
        )

        # main loop
        iteration = 0
        last_problematic_iteration = 0
        last_improvement = 0
        while True:

            cost = self.compute_gradient_hessian()

            # apply damping to D, compute invD inverting one block at a time
            for arg_info in self.elims.blocks.values():
                s = arg_info.param_slice
                D_block = self.D[s, s]
                D_damped_block = D_block.todense() + np.diag(
                    D_block.diagonal() * self.mult_damping  # + self.add_damping
                )
                self.D[s, s] = D_damped_block
                self.invD[s, s] = np.linalg.inv(D_damped_block)

            # apply damping to A
            self.A.setdiag(
                self.A.diagonal() * (1 + self.mult_damping)  # + self.add_damping
            )

            # compute Schur complement
            S = self.A - (self.B @ (self.invD @ self.B.T))

            # compute error for reduced linear system
            reducedErrorX = self.errorX - self.B @ (self.invD @ self.errorP)
            self.deltaX = sparse_solve(S, reducedErrorX)
            self.deltaP = self.invD @ (self.errorP - self.B.T @ self.deltaX)

            # expected cost improvement, according to the quadratic model
            expected_cost_improvement = 0.5 * (
                np.dot(self.deltaX, self.errorX) + np.dot(self.deltaP, self.errorP)
            )

            for b in self.survs.blocks.values():
                b.backup()
                apply_parametrized_step(b.arg, -self.deltaX[b.param_slice])

            for b in self.elims.blocks.values():
                b.backup()
                apply_parametrized_step(b.arg, -self.deltaP[b.param_slice])

            new_cost = self.compute_gradient()
            new_expected_cost_improvement = 0.5 * (
                np.dot(self.deltaX, self.errorX) + np.dot(self.deltaP, self.errorP)
            )

            cost_improvement = cost - new_cost
            improvement_ratio = cost_improvement / expected_cost_improvement

            if new_cost > cost:  # cost got worse? that's BAD!
                self.mult_damping = self.mult_damping * 2
                # self.add_damping = self.add_damping * 3
                mood = Optimizer.MOODS["crying"]
                last_problematic_iteration = iteration

                # restore previous state
                for b in self.survs.blocks.values():
                    b.restore()

                for b in self.elims.blocks.values():
                    b.restore()

            else:  # cost improved? check how well it matches with quadratic model

                # at least some degree of agreement with quadratic model
                if improvement_ratio > 0.4:
                    self.mult_damping = self.mult_damping * 0.5
                    # self.add_damping = self.add_damping * 0.7
                    mood = Optimizer.MOODS["excited"]
                else:
                    self.mult_damping = self.mult_damping * 2
                    # self.add_damping = self.add_damping * 1.5
                    mood = Optimizer.MOODS["worried"]

            # print status
            rel_change_percentage = (new_cost - cost) * 100.0 / cost
            print(
                f"{mood} [{iteration}]"
                + f" {cost:.03e} -> {new_cost:.03e} ({rel_change_percentage:.02f}%)"
                + f"  q-rel-\u03b4: {improvement_ratio:.02f}"
                + f"  \u03bb: {self.mult_damping:.02e}"
            )

            # check stopping condition (they are all sloppy...)
            if new_cost < cost * 0.999:
                last_improvement = iteration

            if (
                iteration >= last_improvement + 3
                and iteration <= last_problematic_iteration + 2
            ) or new_cost < 1e-15:
                mood = Optimizer.MOODS["shrug"]
                print(f"{mood} converged or no improvement")
                break

            iteration += 1
