import math
import unittest

from structs.array.array import Array


class ArrayTests(unittest.TestCase):
    def test_scalar_indexing_returns_scalar(self):
        arr = Array([[1, 2], [3, 4]])
        self.assertEqual(arr[0, 1], 2)
        self.assertEqual(arr[:, 1].data, [2, 4])

    def test_setitem_updates_nested_data(self):
        arr = Array([[1, 2], [3, 4]])
        arr[0, 0] = 99
        self.assertEqual(arr.data, [[99, 2], [3, 4]])

    def test_statistics_methods(self):
        arr = Array([[1, 2], [3, 4]])
        self.assertEqual(arr.sum(), 10)
        self.assertAlmostEqual(arr.mean(), 2.5)
        self.assertAlmostEqual(arr.std(), math.sqrt(1.25))

    def test_linear_algebra_helpers(self):
        mat = Array([[1, 2], [3, 4]])
        vec = Array([5, 6])
        self.assertEqual(mat.matvec(vec).data, [17, 39])
        self.assertEqual(mat.matmul(Array([[5, 6], [7, 8]])).data, [[19, 22], [43, 50]])
        self.assertEqual(Array([1, 2]).outer(Array([3, 4])).data, [[3, 4], [6, 8]])
        self.assertEqual(Array([1, 2, 3]).dot(Array([4, 5, 6])), 32)
        outer_result = Array([-0.76]).outer(Array([1, 2, 3])).data
        self.assertEqual(len(outer_result), 1)
        self.assertAlmostEqual(outer_result[0][0], -0.76)
        self.assertAlmostEqual(outer_result[0][1], -1.52)
        self.assertAlmostEqual(outer_result[0][2], -2.28)
        self.assertEqual(mat.T.data, [[1, 3], [2, 4]])


if __name__ == "__main__":
    unittest.main()
