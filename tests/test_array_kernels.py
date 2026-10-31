import math
import unittest

import matlib
from debug.errors import ShapeError, ValidationError
from matlib._array import broadcasting, linalg, manipulation, ops, stats
from matlib._list import (
    broadcasting as list_broadcasting,
)
from matlib._list import (
    linalg as list_linalg,
)
from matlib._list import (
    manipulation as list_manipulation,
)
from matlib._list import (
    ops as list_ops,
)
from matlib._list import (
    stats as list_stats,
)
from structs.array.array import Array


class ArrayKernelTests(unittest.TestCase):
    def test_unary_kernels_return_arrays(self):
        array = Array([[1.0, 4.0], [9.0, 16.0]])

        result = ops.sqrt(array)

        self.assertIsInstance(result, Array)
        self.assertEqual(result.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(ops.square(Array([2, 3])).tolist(), [4, 9])
        self.assertEqual(ops.abs(Array([-2, 3])).tolist(), [2, 3])
        self.assertEqual(ops.clip(Array([-2, 0, 5]), 0, 3).tolist(), [0, 0, 3])
        self.assertEqual(ops.sign(Array([-2, 0, 5])).tolist(), [-1, 0, 1])
        self.assertEqual(ops.round(Array([1.25, 1.35]), 1).tolist(), [1.2, 1.4])
        self.assertAlmostEqual(ops.exp(Array([0.0])).tolist()[0], 1.0)
        self.assertEqual(ops._apply_unary(2.0, math.sqrt), math.sqrt(2.0))

    def test_unary_kernels_validate_domains(self):
        with self.assertRaises(ValidationError):
            ops.log(Array([1.0, 0.0]))
        with self.assertRaises(ValidationError):
            ops.log(Array([-1.0]))
        with self.assertRaises(ValidationError):
            ops.sqrt(Array([4.0, -1.0]))

    def test_broadcasting_kernels(self):
        matrix = Array([[1, 2, 3], [4, 5, 6]])
        row = Array([10, 20, 30])

        result = broadcasting.elementwise(matrix, row, lambda x, y: x + y)
        self.assertEqual(result.tolist(), [[11, 22, 33], [14, 25, 36]])

        scalar = broadcasting.elementwise(matrix, 2, lambda x, y: x * y)
        self.assertEqual(scalar.tolist(), [[2, 4, 6], [8, 10, 12]])

        result = broadcasting.elementwise(Array([1, 2]), Array([[10], [20]]), lambda x, y: x * y)
        self.assertEqual(result.tolist(), [[10, 20], [20, 40]])

        left, right, squeeze = broadcasting.broadcast(Array([1, 2]), Array([3, 4]))
        self.assertEqual(left.shape, (1, 2))
        self.assertEqual(right.shape, (2, 1))
        self.assertEqual(squeeze, "scalar")

        with self.assertRaises(ValueError):
            broadcasting.elementwise(Array([1, 2]), Array([1, 2, 3]), lambda x, y: x + y)

    def test_linalg_kernels(self):
        a = Array([[1, 2, 3], [4, 5, 6]])
        b = Array([[1, 2], [3, 4], [5, 6]])

        self.assertEqual(linalg.matmul(a, b).tolist(), [[22, 28], [49, 64]])
        self.assertEqual(linalg.matvec(a, Array([1, 2, 3])).tolist(), [14, 32])
        self.assertEqual(linalg.outer(Array([1, 2]), Array([3, 4])).tolist(), [[3, 4], [6, 8]])
        self.assertEqual(linalg.transpose(a).tolist(), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(
            linalg.permute(Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), (2, 0, 1)).tolist(),
            [[[1, 3], [5, 7]], [[2, 4], [6, 8]]],
        )
        self.assertEqual(linalg.tensordot(a, b, axes=1).tolist(), [[22, 28], [49, 64]])
        self.assertEqual(linalg.tensordot(a, a, axes=([0, 1], [0, 1])), 91)
        self.assertEqual(linalg.det(Array([[1, 2], [3, 4]])), -2)
        self.assertEqual(linalg.inverse(Array([[1, 2], [3, 4]])).tolist(), [[-2.0, 1.0], [1.5, -0.5]])
        self.assertEqual(linalg.solve(Array([[2, 1], [1, 3]]), Array([3, 5])).tolist(), [0.8, 1.4])
        self.assertEqual(linalg.cross(Array([1, 0, 0]), Array([0, 1, 0])).tolist(), [0, 0, 1])
        self.assertEqual(linalg.cholesky(Array([[4, 2], [2, 3]])).tolist(), [[2.0, 0], [1.0, math.sqrt(2)]])

    def test_transpose_and_permute_are_buffer_views(self):
        a = Array([[1, 2, 3], [4, 5, 6]])

        transposed = a.T
        self.assertIs(transposed.data, a.data)
        self.assertEqual(transposed.shape, (3, 2))
        self.assertEqual(transposed.strides, (1, 3))
        self.assertEqual(transposed.tolist(), [[1, 4], [2, 5], [3, 6]])

        a[0, 1] = 20
        self.assertEqual(transposed[1, 0], 20)
        transposed[1, 0] = 50
        self.assertEqual(a[0, 1], 50)

        cube = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        permuted = linalg.permute(cube, (2, 0, 1))
        self.assertIs(permuted.data, cube.data)
        self.assertEqual(permuted.tolist(), [[[1, 3], [5, 7]], [[2, 4], [6, 8]]])

    def test_linalg_validation(self):
        with self.assertRaises(ShapeError):
            linalg.det(Array([[1, 2, 3], [4, 5, 6]]))
        with self.assertRaises(ShapeError):
            linalg.inverse(Array([[1, 2, 3], [4, 5, 6]]))
        with self.assertRaises(ShapeError):
            linalg.inverse(Array([[1, 2], [2, 4]]))
        with self.assertRaises(ShapeError):
            linalg.matmul(Array([[1, 2]]), Array([[1, 2]]))
        with self.assertRaises(ShapeError):
            linalg.matvec(Array([[1, 2], [3, 4]]), Array([1]))
        with self.assertRaises(ShapeError):
            linalg.cross(Array([1, 2]), Array([3, 4]))
        with self.assertRaises(ShapeError):
            linalg.cholesky(Array([[1, 2], [2, 1]]))
        with self.assertRaises(ShapeError):
            linalg.permute(Array([[1, 2]]), (0, 0))

    def test_stats_reductions(self):
        array = Array([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(stats.sum(array), 21)
        self.assertEqual(stats.mean(array), 3.5)
        self.assertAlmostEqual(stats.variance(array), 35 / 12)
        self.assertAlmostEqual(stats.std(array), math.sqrt(35 / 12))
        self.assertEqual(stats.max(array), 6)
        self.assertEqual(stats.min(array), 1)
        self.assertEqual(stats.prod(array), 720)
        self.assertEqual(stats.argmax(array), 5)
        self.assertEqual(stats.argmin(array), 0)
        self.assertEqual(stats.count_nonzero(array), 6)
        self.assertAlmostEqual(stats.norm(array), math.sqrt(91))

        self.assertEqual(stats.sum_axis(array, 0).tolist(), [5, 7, 9])
        self.assertEqual(stats.sum_axis(array, 1).tolist(), [6, 15])
        self.assertEqual(stats.sum_axis(array, 1, keepdims=True).tolist(), [[6], [15]])
        self.assertEqual(stats.mean_axis(array, 0).tolist(), [2.5, 3.5, 4.5])
        self.assertEqual(stats.max_axis(array, 1).tolist(), [3, 6])
        self.assertEqual(stats.min_axis(array, 0).tolist(), [1, 2, 3])
        self.assertEqual(stats.argmax_axis(array, 1).tolist(), [2, 2])
        self.assertEqual(stats.count_nonzero_axis(Array([[0, 1], [2, 0]]), 0).tolist(), [1, 1])
        self.assertEqual(stats.prod_axis(array, 0).tolist(), [4, 10, 18])
        self.assertEqual(stats.sum_axis(array), 21)
        self.assertEqual(stats.sum_axis(array, axis=None, keepdims=True).tolist(), [[21]])

    def test_stats_validation(self):
        with self.assertRaises(ValidationError):
            stats.mean([])
        with self.assertRaises(ValidationError):
            stats.variance([])
        with self.assertRaises(ValidationError):
            stats.std([])
        with self.assertRaises(ValidationError):
            stats.max([])
        with self.assertRaises(ValidationError):
            stats.min([])
        with self.assertRaises(ValidationError):
            stats.argmax([])
        with self.assertRaises(ValidationError):
            stats.argmin([])

    def test_manipulation_kernels(self):
        a = Array([[1, 2, 3], [4, 5, 6]])
        b = Array([[10, 20, 30], [40, 50, 60]])

        self.assertEqual(
            manipulation.concatenate([a, b], axis=0).tolist(),
            [[1, 2, 3], [4, 5, 6], [10, 20, 30], [40, 50, 60]],
        )
        self.assertEqual(
            manipulation.concatenate([a, b], axis=1).tolist(),
            [[1, 2, 3, 10, 20, 30], [4, 5, 6, 40, 50, 60]],
        )
        self.assertEqual(
            manipulation.stack([a, b], axis=0).shape,
            (2, 2, 3),
        )
        self.assertEqual(
            manipulation.stack([a, b], axis=1).tolist()[0][0],
            [1, 2, 3],
        )
        self.assertEqual(manipulation.flip(a, 0).tolist(), [[4, 5, 6], [1, 2, 3]])
        self.assertEqual(manipulation.flip(a, 1).tolist(), [[3, 2, 1], [6, 5, 4]])

        parts = manipulation.split(a, 3, axis=1)
        self.assertEqual([part.tolist() for part in parts], [[[1], [4]], [[2], [5]], [[3], [6]]])
        self.assertEqual(
            [part.tolist() for part in manipulation.split(a, [1], axis=0)],
            [[[1, 2, 3]], [[4, 5, 6]]],
        )

        rows = manipulation.unbind(a, axis=0)
        self.assertEqual([row.tolist() for row in rows], [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(manipulation.unbind(Array([1, 2, 3]), axis=0), [1, 2, 3])

    def test_manipulation_validation(self):
        a = Array([[1, 2], [3, 4]])
        with self.assertRaises(ValidationError):
            manipulation.concatenate([], axis=0)
        with self.assertRaises(ValidationError):
            manipulation.stack([], axis=0)
        with self.assertRaises(ShapeError):
            manipulation.stack([a, Array([[1, 2, 3]])], axis=0)
        with self.assertRaises(ShapeError):
            manipulation.concatenate([a, Array([[1, 2, 3]])], axis=1)
        with self.assertRaises(ValidationError):
            manipulation.split(a, 0, axis=0)
        with self.assertRaises(ShapeError):
            manipulation.split(Array([1, 2, 3]), 2, axis=0)

    def test_array_integration(self):
        a = Array([[1, 2, 3], [4, 5, 6]])
        b = Array([[10, 20, 30], [40, 50, 60]])

        self.assertEqual((a + b).tolist(), [[11, 22, 33], [44, 55, 66]])
        self.assertEqual((a + Array([10, 20, 30])).tolist(), [[11, 22, 33], [14, 25, 36]])
        self.assertEqual((a * 2).tolist(), [[2, 4, 6], [8, 10, 12]])
        self.assertEqual((a - 1).tolist(), [[0, 1, 2], [3, 4, 5]])
        self.assertEqual((a / 2).tolist(), [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
        self.assertTrue((a > 3).tolist()[1][0])

        in_place = Array([[1, 2], [3, 4]])
        in_place += Array([[10, 20], [30, 40]])
        self.assertEqual(in_place.tolist(), [[11, 22], [33, 44]])

        self.assertEqual(a.sum(axis=0).tolist(), [5, 7, 9])
        self.assertEqual(a.mean(axis=1).tolist(), [2.0, 5.0])
        self.assertEqual(a.max(axis=0).tolist(), [4, 5, 6])
        self.assertAlmostEqual(a.std(), math.sqrt(35 / 12))
        self.assertEqual(a.T.tolist(), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(a.reshape(3, 2).tolist(), [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(a.flatten().tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(a.matmul(Array([[1, 0], [0, 1], [1, 1]])).tolist(), [[4, 5], [10, 11]])
        self.assertEqual(Array.concatenate([a, b], axis=0).tolist()[2], [10, 20, 30])
        self.assertEqual([part.tolist() for part in a.split(2, axis=0)], [[[1, 2, 3]], [[4, 5, 6]]])

        assigned = Array([[1, 2], [3, 4]])
        assigned.data = [[5, 6], [7, 8]]
        self.assertEqual(assigned.tolist(), [[5, 6], [7, 8]])

    def test_routed_matlib_functions(self):
        array = Array([[1.0, 4.0], [9.0, 16.0]])

        exp_result = matlib.exp(array)
        self.assertIsInstance(exp_result, Array)
        self.assertAlmostEqual(exp_result.tolist()[0][0], math.e)

        self.assertEqual(matlib.sin(Array([0.0])).tolist(), [0.0])
        self.assertEqual(matlib.sqrt(array).tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(matlib.matmul(array, Array([[1, 0], [0, 1]])).tolist(), array.tolist())
        self.assertEqual(matlib.transpose(Array([[1, 2], [3, 4]])).tolist(), [[1, 3], [2, 4]])
        self.assertEqual(matlib.outer(Array([1, 2]), Array([3, 4])).tolist(), [[3, 4], [6, 8]])
        self.assertEqual(matlib.det(Array([[1, 2], [3, 4]])), -2)
        self.assertEqual(matlib.sum(Array([1, 2, 3])), 6)
        self.assertEqual(matlib.mean(Array([1, 2, 3])), 2)
        self.assertEqual(matlib.max(Array([1, 2, 3])), 3)
        self.assertEqual(matlib.sum_axis(Array([[1, 2], [3, 4]]), 0).tolist(), [4, 6])
        self.assertEqual(matlib.flip(Array([1, 2, 3]), 0).tolist(), [3, 2, 1])

        concatenated = matlib.concatenate([Array([[1, 2]]), Array([[3, 4]])], axis=0)
        self.assertIsInstance(concatenated, Array)
        self.assertEqual(concatenated.tolist(), [[1, 2], [3, 4]])

        stacked = matlib.stack([Array([1, 2]), Array([3, 4])], axis=0)
        self.assertIsInstance(stacked, Array)
        self.assertEqual(stacked.tolist(), [[1, 2], [3, 4]])

    def test_parity_with_list_kernels(self):
        nested = [[1.0, 4.0], [9.0, 16.0]]
        array = Array(nested)

        for name, args in {
            "exp": (nested,),
            "sin": (nested,),
            "sqrt": (nested,),
            "abs": ([[-1, 2], [-3, 4]],),
            "clip": (nested, 2, 9),
            "sign": ([[-1, 0, 2]],),
            "round": ([1.25, 1.35], 1),
        }.items():
            list_result = getattr(list_ops, name)(*[Array(arg) if isinstance(arg, Array) else arg for arg in args])
            array_result = getattr(ops, name)(*[Array(arg) if isinstance(arg, list) else arg for arg in args])
            self.assertEqual(Array(list_result).tolist(), array_result.tolist(), name)

        list_result = list_broadcasting.elementwise(
            [[1, 2], [3, 4]], [[10, 20], [30, 40]], lambda x, y: x * y
        )
        array_result = broadcasting.elementwise(
            Array([[1, 2], [3, 4]]), Array([[10, 20], [30, 40]]), lambda x, y: x * y
        )
        self.assertEqual(Array(list_result).tolist(), array_result.tolist())

        list_result = list_linalg.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        array_result = linalg.matmul(Array([[1, 2], [3, 4]]), Array([[5, 6], [7, 8]]))
        self.assertEqual(Array(list_result).tolist(), array_result.tolist())

        self.assertEqual(
            list_linalg.det(Array([[1, 2], [3, 4]])),
            linalg.det(Array([[1, 2], [3, 4]])),
        )
        self.assertEqual(
            Array(list_linalg.transpose(Array([[1, 2], [3, 4]]))).tolist(),
            linalg.transpose(Array([[1, 2], [3, 4]])).tolist(),
        )

        self.assertEqual(list_stats.sum([1, 2, 3, 4]), stats.sum(Array([[1, 2], [3, 4]])))
        self.assertEqual(list_stats.mean([1, 2, 3]), stats.mean(Array([1, 2, 3])))
        self.assertAlmostEqual(
            list_stats.variance([1, 2, 3]), stats.variance(Array([1, 2, 3]))
        )

        list_result = list_manipulation.concatenate(
            [Array([[1], [2]]), Array([[3], [4]])], axis=0
        )
        array_result = manipulation.concatenate(
            [Array([[1], [2]]), Array([[3], [4]])], axis=0
        )
        self.assertEqual(Array(list_result).tolist(), array_result.tolist())

        list_result = list_manipulation.stack(
            [Array([[1], [2]]), Array([[3], [4]])], axis=1
        )
        array_result = manipulation.stack(
            [Array([[1], [2]]), Array([[3], [4]])], axis=1
        )
        self.assertEqual(Array(list_result).tolist(), array_result.tolist())

        list_result = list_manipulation.flip(Array([[1, 2], [3, 4]]), axis=1)
        array_result = manipulation.flip(Array([[1, 2], [3, 4]]), axis=1)
        self.assertEqual(Array(list_result).tolist(), array_result.tolist())

        list_result = list_manipulation.split(Array([[1, 2], [3, 4]]), 2, axis=1)
        array_result = manipulation.split(Array([[1, 2], [3, 4]]), 2, axis=1)
        self.assertEqual(
            [Array(part).tolist() for part in list_result],
            [part.tolist() for part in array_result],
        )

        list_result = list_manipulation.unbind(Array([[1, 2], [3, 4]]), axis=0)
        array_result = manipulation.unbind(Array([[1, 2], [3, 4]]), axis=0)
        self.assertEqual(
            [Array(part).tolist() if isinstance(part, list) else part for part in list_result],
            [part.tolist() if isinstance(part, Array) else part for part in array_result],
        )




    def test_view_edge_cases(self):
        a = Array([[1, 2, 3], [4, 5, 6]])

        flipped = manipulation.flip(a, 1)
        self.assertIs(flipped.data, a.data)
        self.assertEqual(flipped.strides[1], -1)
        self.assertEqual(flipped.tolist(), [[3, 2, 1], [6, 5, 4]])

        reshaped = a.reshape(3, 2)
        self.assertEqual(reshaped.tolist(), [[1, 2], [3, 4], [5, 6]])
        squeezed = Array([[[1], [2]]]).squeeze()
        self.assertEqual(squeezed.shape, (2,))
        unsqueezed = Array([1, 2]).unsqueeze(0)
        self.assertEqual(unsqueezed.shape, (1, 2))
        broadcast = Array([1, 2]).broadcast_to((2, 2))
        self.assertEqual(broadcast.tolist(), [[1, 2], [1, 2]])

        split = a.split(2, axis=0)
        self.assertIs(split[1].data, a.data)
        unbound = a.unbind(axis=0)
        self.assertIs(unbound[1].data, a.data)

        empty = Array([]).reshape(0, 2)
        self.assertEqual(empty.shape, (0, 2))
        self.assertEqual(empty.tolist(), [])

if __name__ == "__main__":

    unittest.main()
