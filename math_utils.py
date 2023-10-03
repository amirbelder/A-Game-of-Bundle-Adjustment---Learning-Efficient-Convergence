# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math

import numpy as np

# simple vector class
class Vec:
    def __init__(self, *values):
        if len(values) == 1 and isinstance(
            values[0], (Quat, Vec, list, tuple, np.ndarray)
        ):
            self.values = list(values[0])
        else:
            self.values = list(values)

    def __add__(self, other):
        assert len(self.values) == len(other)
        return Vec(*[v + other[i] for i, v in enumerate(self.values)])

    def __iadd__(self, other):
        for i in range(len(self.values)):
            self.values[i] += other[i]
        return self

    def __radd__(self, other):
        assert len(self.values) == len(other)
        return Vec(*[v + other[i] for i, v in enumerate(self.values)])

    def __sub__(self, other):
        assert len(self.values) == len(other)
        return Vec(*[v - other[i] for i, v in enumerate(self.values)])

    def __isub__(self, other):
        for i in range(len(self.values)):
            self.values[i] -= other[i]
        return self

    def __rsub__(self, other):
        assert len(self.values) == len(other)
        return Vec(*[other[i] - v for i, v in enumerate(self.values)])

    def __mul__(self, other):
        return Vec(*[v * other for v in self.values])

    def __imul__(self, other):
        for i in range(len(self.values)):
            self.values[i] *= other
        return self

    def __rmul__(self, other):
        return Vec(*[v * other for v in self.values])

    def __truediv__(self, other):
        return Vec(*[v / other for v in self.values])

    def __itruediv__(self, other):
        for i in range(len(self.values)):
            self.values[i] /= other
        return self

    def __neg__(self):
        return Vec(*[-v for v in self.values])

    def __repr__(self):
        return "Vec" + str(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def __iter__(self):
        return self.values.__iter__()

    def data(self):
        return list(self.values)

    def set_data(self, values):
        self.values = list(values)

    def dot(self, other):
        assert len(self.values) == len(other)
        retv = self[0] * other[0]
        for i in range(1, len(self.values)):
            retv += self[i] * other[i]
        return retv

    def squared_norm(self):
        retv = self[0] * self[0]
        for i in range(1, len(self.values)):
            retv += self[i] * self[i]
        return retv

    def norm(self):
        return sqrt(self.squared_norm())

    def normalized(self):
        return self / self.norm()

    def normalize(self):
        self.values = self.normalized().values

    def cross(self, other):
        assert len(self.values) == 3
        assert len(other.values) == 3
        return Vec(
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        )


# quaternion vector class
# Amir - this means Rotation and Translation
class Quat:
    def __init__(self, *values, normalize=False):
        if len(values) == 4:
            self.w, self.x, self.y, self.z = values
        elif (
            len(values) == 1
            and isinstance(values[0], (Quat, Vec, list, tuple, np.ndarray))
            and len(values[0]) == 4
        ):
            self.w, self.x, self.y, self.z = values[0]
        elif (
            len(values) == 2
            and isinstance(values[1], (Vec, list, tuple, np.ndarray))
            and len(values[1]) == 3
        ):
            self.w, (self.x, self.y, self.z) = values[0], values[1]
        else:
            raise Exception(
                "could not construct Quat from {0}, len={1}".format(values, len(values))
            )
        if normalize:
            self.normalize()

    def __repr__(self):
        return "Quat" + str([self.w, self.x, self.y, self.z])

    def __iter__(self):
        return [self.w, self.x, self.y, self.z].__iter__()

    def __getitem__(self, key):
        return getattr(self, "wxyz"[key])

    def __len__(self):
        return 4

    def __param_len__(self):
        return 3

    def __parametrize__(self, g):
        return Vec(
            Vec(-self.x, self.w, self.z, -self.y).dot(g),
            Vec(-self.y, -self.z, self.w, self.x).dot(g),
            Vec(-self.z, self.y, -self.x, self.w).dot(g),
        )

    def __apply_parametrized_step__(self, dg):
        delta = (
            dg[0] * Vec(-self.x, self.w, self.z, -self.y)
            + dg[1] * Vec(-self.y, -self.z, self.w, self.x)
            + dg[2] * Vec(-self.z, self.y, -self.x, self.w)
        )
        self.w, self.x, self.y, self.z = (self.coeffs() + delta).normalized()

    def coeffs(self):
        return Vec(self.w, self.x, self.y, self.z)

    def data(self):
        return [self.w, self.x, self.y, self.z]

    def set_data(self, values):
        self.w, self.x, self.y, self.z = values

    def normalize(self):
        self.w, self.x, self.y, self.z = self.coeffs().normalized()

    def vec(self):
        return Vec(self.x, self.y, self.z)

    def __call__(self, v):
        uv = 2.0 * self.vec().cross(v)
        return v + self.w * uv + self.vec().cross(uv)


# simple jet evaluation class
class Jet:
    def __init__(self, value: float, pders: np.ndarray):
        self.value = value
        self.pders = pders

    @staticmethod
    def var(value: float, index: int):
        pders = np.zeros(index + 1)
        pders[index] = 1.0
        return Jet(value, pders)

    def block(values: np.array, start_index: int):
        return Vec(*[Jet.var(v, start_index + i) for i, v in enumerate(values)])

    def __repr__(self):
        return "{0}:{1}".format(self.value, list(self.pders))

    @staticmethod
    def ensure_pders_same_size(a, b):
        sa = len(a.pders)
        sb = len(b.pders)
        if sa == sb:
            return
        if sa < sb:
            # orig
            # a.pders.resize(sb)
            # mine
            a.pders.resize(sb, refcheck=False)
        else:
            # orig
            # b.pders.resize(sa)
            # mine
            b.pders.resize(sa, refcheck=False)

    def __add__(self, other):
        if isinstance(other, (float, int)):
            return Jet(self.value + other, self.pders)
        if not isinstance(other, Jet):
            return NotImplemented
        Jet.ensure_pders_same_size(self, other)
        return Jet(self.value + other.value, self.pders + other.pders)

    def __radd__(self, other):
        if isinstance(other, (float, int)):
            return Jet(self.value + other, self.pders)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return Jet(self.value - other, self.pders)
        if not isinstance(other, Jet):
            return NotImplemented
        Jet.ensure_pders_same_size(self, other)
        return Jet(self.value - other.value, self.pders - other.pders)

    def __rsub__(self, other):
        if isinstance(other, (float, int)):
            return Jet(other - self.value, -self.pders)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Jet(self.value * other, self.pders * other)
        if not isinstance(other, Jet):
            return NotImplemented
        Jet.ensure_pders_same_size(self, other)
        return Jet(
            self.value * other.value,
            self.pders * other.value + other.pders * self.value,
        )

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return Jet(self.value * other, self.pders * other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return Jet(self.value / other, self.pders / other)
        if not isinstance(other, Jet):
            return NotImplemented
        Jet.ensure_pders_same_size(self, other)
        inv_other_v = 1.0 / other.value
        return Jet(
            self.value * inv_other_v,
            self.pders * inv_other_v
            - other.pders * (self.value * inv_other_v * inv_other_v),
        )

    def __neg__(self):
        return Jet(-self.value, -self.pders)

    def sqrt(self):
        sqrt_val = math.sqrt(self.value)
        return Jet(sqrt_val, self.pders * (0.5 / sqrt_val))

    def asin(self):
        asin_val = math.asin(self.value)
        factor = 1.0 / math.sqrt(1.0 - self.value * self.value)
        return Jet(asin_val, self.pders * factor)

    def acos(self):
        acos_val = math.acos(self.value)
        factor = -1.0 / math.sqrt(1.0 - self.value * self.value)
        return Jet(acos_val, self.pders * factor)

    # this function computes the given function AND the derivative with respect to
    # the arguments - it assumes the function uses the derivatives arithmetics or
    # Vec/Quat types above that can accept Jet fields, and invoked the function
    # with arguments replaced with "Jet" arguments, and extracts from the
    @staticmethod
    def compute_first_order(function, *argv):
        first_order_args, indices, lengths = [], [], []
        start_index = 0
        """if not isinstance(argv[2][0],int):
            new_argv = (argv[0], argv[1], Vec(argv[2][0]))
            argv = new_argv"""
        for arg in argv:
            indices.append(start_index)
            if isinstance(arg, (Quat, Vec, np.ndarray, list, tuple)):
                first_order_args.append(Jet.block(arg, start_index))
                start_index += len(arg)
                lengths.append(len(arg))
            else:
                first_order_args.append(Jet.var(arg, start_index))
                start_index += 1
                lengths.append(1)
        result = Vec(function(*first_order_args))
        pders = np.zeros((len(result), start_index))
        for j, r in enumerate(result):
            #try:
                pders[j, 0 : len(r.pders)] = r.pders
            #except:
            #    continue
        retv = (
            np.array([r.value for r in result]).reshape(-1, 1),
            [
                np.array(
                    [
                        parametrize(arg, pders[j, i : (i + l)])
                        for j in range(len(result))
                    ]
                )
                for arg, i, l in zip(argv, indices, lengths)
            ],
        )
        return retv


def sqrt(x):
    if isinstance(x, Jet):
        return x.sqrt()
    return np.sqrt(x)


def asin(x):
    if isinstance(x, Jet):
        return x.asin()
    return np.asin(x)


def acos(x):
    if isinstance(x, Jet):
        return x.acos()
    return np.acos(x)


def param_len(x):
    return x.__param_len__() if hasattr(x, "__param_len__") else len(x)


def parametrize(x, dx):
    return np.array(x.__parametrize__(dx)) if hasattr(x, "__parametrize__") else dx


def apply_parametrized_step(x, dx):
    if hasattr(x, "__apply_parametrized_step__"):
        x.__apply_parametrized_step__(dx)
    else:
        x += dx


# simple test
if __name__ == "__main__":

    def my_func(xy, zw, q):
        return xy.dot(zw) + sqrt(q)

    retv = Jet.compute_first_order(my_func, [0.1, 0.2], [4, 5], 7)
    print(retv)
