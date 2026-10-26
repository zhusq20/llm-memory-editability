"""Minimal MIT-licensed NDR generators, with sample-class names disambiguated.

Source: https://github.com/RobertCsordas/ndr
Commit: da20530dfb4336deddfbe5e79d62e72d1dc2580e
Original random generation and arithmetic behavior are preserved.

MIT License

Copyright (c) 2021 Róbert Csordás

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


# Original dataset/listops.py SHA256:
# 57506c0991129879ccfdfc5931ed7280e9ce0258795859b49ae33e7af1256931
@dataclasses.dataclass
class ListOpsSample:
    input: str | list[int]
    output: int
    depth: int
    length: int
    max_dependency_depth: int

    def dup(self, **overwrite_args):
        d = dataclasses.asdict(self)
        d.update(overwrite_args)
        return self.__class__(**d)


def get_selector_op(
    selector: Callable[[list[ListOpsSample]], int],
) -> Callable[[list[ListOpsSample]], ListOpsSample]:
    def op(x):
        y = selector(x)
        return x[y].dup(
            max_dependency_depth=x[y].max_dependency_depth + 1, depth=max(a.depth for a in x) + 1
        )

    return op


def sum_op(x: list[ListOpsSample]) -> ListOpsSample:
    return ListOpsSample(
        "",
        sum(a.output for a in x) % 10,
        max(a.depth for a in x) + 1,
        -1,
        max(a.max_dependency_depth for a in x) + 1,
    )


def median_op(x: list[ListOpsSample]) -> ListOpsSample:
    indices = np.argsort([a.output for a in x])
    selected = (
        [indices[len(x) // 2]]
        if len(x) % 2 == 1
        else [indices[len(x) // 2 - 1], indices[len(x) // 2]]
    )
    res = int(sum(x[i].output for i in selected) / len(selected))
    return ListOpsSample(
        "",
        res,
        max(a.depth for a in x) + 1,
        -1,
        max(x[i].max_dependency_depth for i in selected) + 1,
    )


class ListOps:
    OPS = [
        ("MIN", get_selector_op(lambda x: min(range(len(x)), key=lambda i: x[i].output))),
        ("MAX", get_selector_op(lambda x: max(range(len(x)), key=lambda i: x[i].output))),
        ("MED", median_op),
        ("SM", sum_op),
    ]

    def get_one(self, seed) -> ListOpsSample:
        def sample_subop(args: list[ListOpsSample]) -> ListOpsSample:
            op = self.OPS[seed.randint(len(self.OPS))]
            res = op[1](args)
            res.input = f"[{op[0]} {' '.join(a.input for a in args)} ]"
            res.length = 3 + sum(a.length for a in args)
            return res

        def gen_args(max_args: int, max_depth: int) -> list[ListOpsSample]:
            n_args = seed.randint(2, max_args + 1) if max_args + 1 != 2 else 2
            return [get_subtree(max_depth) for _ in range(n_args)]

        def get_subtree(max_depth: int) -> ListOpsSample:
            if max_depth == 1 or seed.rand() > self.p_op:
                val = seed.randint(10)
                return ListOpsSample(str(val), val, 1, 1, 1)
            else:
                return sample_subop(gen_args(self.max_args, max_depth - 1))

        res = get_subtree(self.depth[1])
        while res.depth < self.depth[0]:
            args = gen_args(self.max_args - 1, self.depth[1] - 1)
            args.insert(seed.randint(len(args)), res)
            res = sample_subop(args)
        return res


# Original dataset/simple_arithmetic.py SHA256:
# f83e7c7aa549947b6effe5152d985e8b8be045bc082ae53b68b42938a7501502
@dataclass
class ArithmeticSample:
    input: str
    output: int
    depth: int


class SimpleArithmetics:
    OPS = [("+", lambda a, b: a + b), ("*", lambda a, b: a * b)]

    def format_op(self, a1, a2, op):
        return f"{a1} {a2}{op}" if self.rpn else f"({a1}{op}{a2})"

    def get_one(self, seed, min_depth: int, max_depth: int) -> ArithmeticSample:
        def sample_single_op(a1: ArithmeticSample, a2: ArithmeticSample) -> ArithmeticSample:
            op = self.OPS[seed.randint(0, len(self.OPS))]
            expr = self.format_op(a1.input, a2.input, op[0])
            return ArithmeticSample(
                expr, op[1](a1.output, a2.output) % self.maxval, max(a1.depth, a2.depth) + 1
            )

        def get(d: int) -> ArithmeticSample:
            if d >= max_depth or seed.rand() > self.p_subop:
                res = seed.randint(0, self.maxval)
                assert res < self.maxval
                return ArithmeticSample(str(res), res, 1)
            else:
                a1 = get(d + 1)
                a2 = get(d + 1)
                return sample_single_op(a1, a2)

        s = get(1)
        while s.depth < min_depth:
            other = get(2)
            s, other = (s, other) if seed.rand() < 0 else (other, s)
            s = sample_single_op(s, other)
        return s

    def generate_basic_set(self) -> list[ArithmeticSample]:
        res = []
        for op in self.OPS:
            for a1 in range(self.maxval):
                for a2 in range(self.maxval):
                    res.append(
                        ArithmeticSample(
                            self.format_op(a1, a2, op[0]), op[1](a1, a2) % self.maxval, 1
                        )
                    )
        return res
