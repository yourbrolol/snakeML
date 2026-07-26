import math
import unittest

from structs.array.array import Array

from debug.errors import ShapeError, ValidationError

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

    def test_generator_inputs_are_materialized(self):
        arr = Array([[0, 1], [1, 0]] for _ in range(3))
        self.assertEqual(arr.data, [[[0, 1], [1, 0]], [[0, 1], [1, 0]], [[0, 1], [1, 0]]])
        self.assertEqual(arr.shape, (3, 2, 2))

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

    def test_extended_array_helpers(self):
        arr = Array([[1, 2], [3, 4]])
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.ndim, 2)
        self.assertEqual(arr.size, 4)
        self.assertEqual(arr.dtype, "int")
        self.assertEqual(arr.tolist(), [[1, 2], [3, 4]])
        self.assertEqual(arr.item(3), 4)
        self.assertEqual(arr[[0, 1]].data, [[1, 2], [3, 4]])
        self.assertEqual(arr[[True, False]].data, [[1, 2]])
        self.assertEqual(arr.expand_dims(0).shape, (1, 2, 2))
        self.assertEqual(arr.transpose().data, [[1, 3], [2, 4]])
        self.assertEqual(arr.broadcast_to((3, 2, 2)).shape, (3, 2, 2))
        self.assertEqual(arr.vstack(Array([[5, 6]])).data, [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(arr.hstack(Array([[5, 6]])).data, [[1, 2, 5, 6], [3, 4, 5, 6]])
        self.assertEqual(arr.var(), 1.25)
        self.assertEqual(arr.min(), 1)
        self.assertEqual(arr.max(), 4)
        self.assertEqual(arr.argmax(), 3)
        self.assertEqual(arr.count_nonzero(), 4)
        self.assertEqual(arr.clip(0, 2).data, [[1, 2], [2, 2]])
        self.assertEqual(arr.sign().data, [[1, 1], [1, 1]])
    
    def test_array_battle(self):
        # ===== Constructors =====
        self.assertEqual(Array([]).shape, (0,))
        self.assertEqual(Array([[[1]]]).shape, (1, 1, 1))
        self.assertEqual(Array(range(5)).data, [0, 1, 2, 3, 4])

        # ===== Basic indexing =====
        a = Array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

        self.assertEqual(a[0, 0], 1)
        self.assertEqual(a[-1, -1], 9)
        self.assertEqual(a[1].data, [4, 5, 6])
        self.assertEqual(a[:, 1].data, [2, 5, 8])
        self.assertEqual(a[1:, :2].data, [[4, 5], [7, 8]])
        self.assertEqual(a[::-1].data,
                        [[7, 8, 9],
                        [4, 5, 6],
                        [1, 2, 3]])

        # ===== Boolean / fancy indexing =====
        self.assertEqual(
            a[[True, False, True]].data,
            [[1, 2, 3], [7, 8, 9]]
        )

        self.assertEqual(
            a[[2, 0]].data,
            [[7, 8, 9], [1, 2, 3]]
        )

        # ===== Assignment =====
        b = Array([[1, 2], [3, 4]])
        b[0, 1] = 999
        self.assertEqual(b.data, [[1, 999], [3, 4]])

        # ===== Statistics =====
        c = Array([-3, -2, 0, 2, 3])

        self.assertEqual(c.sum(), 0)
        self.assertEqual(c.mean(), 0)
        self.assertEqual(c.min(), -3)
        self.assertEqual(c.max(), 3)
        self.assertEqual(c.argmax(), 4)
        self.assertEqual(c.count_nonzero(), 4)

        # ===== Clip / sign =====
        self.assertEqual(
            c.clip(-1, 1).data,
            [-1, -1, 0, 1, 1]
        )

        self.assertEqual(
            c.sign().data,
            [-1, -1, 0, 1, 1]
        )

        # ===== Transpose =====
        self.assertEqual(
            a.T.data,
            [[1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]]
        )

        # ===== Matrix multiplication =====
        left = Array([[1, 2],
                    [3, 4]])

        right = Array([[5, 6],
                    [7, 8]])

        self.assertEqual(
            left.matmul(right).data,
            [[19, 22],
            [43, 50]]
        )

        # ===== Matrix-vector =====
        self.assertEqual(
            left.matvec(Array([1, 2])).data,
            [5, 11]
        )

        # ===== Dot =====
        self.assertEqual(
            Array([1, 2, 3]).dot(Array([4, 5, 6])),
            32
        )

        # ===== Outer =====
        self.assertEqual(
            Array([1, 2]).outer(Array([3, 4])).data,
            [[3, 4],
            [6, 8]]
        )

        # ===== Broadcasting =====
        self.assertEqual(
            Array([1, 2]).broadcast_to((3, 2)).shape,
            (3, 2)
        )

        # ===== Stack =====
        self.assertEqual(
            Array([[1], [2]]).hstack(Array([[3], [4]])).data,
            [[1, 3],
            [2, 4]]
        )

        self.assertEqual(
            Array([[1, 2]]).vstack(Array([[3, 4]])).data,
            [[1, 2],
            [3, 4]]
        )

        # ===== Shape metadata =====
        self.assertEqual(a.ndim, 2)
        self.assertEqual(a.size, 9)
        self.assertEqual(a.shape, (3, 3))

        # ===== tolist =====
        self.assertEqual(
            a.tolist(),
            [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]
        )

        # ===== item =====
        self.assertEqual(a.item(8), 9)

        # ===== Expand dims =====
        self.assertEqual(
            Array([1, 2, 3]).expand_dims(0).shape,
            (1, 3)
        )

        self.assertEqual(
            Array([1, 2, 3]).expand_dims(1).shape,
            (3, 1)
        )

        # ===== Zero / one element =====
        self.assertEqual(Array([42]).sum(), 42)
        self.assertEqual(Array([0]).count_nonzero(), 0)

        # ===== Floating point =====
        f = Array([0.1, 0.2, 0.3])
        self.assertAlmostEqual(f.sum(), 0.6, places=7)

        # ===== Large transpose =====
        big = Array([[i * 10 + j for j in range(10)] for i in range(10)])
        self.assertEqual(big.T.shape, (10, 10))
        self.assertEqual(big.T[0, 9], 90)
        self.assertEqual(big.T[9, 0], 9)
        
        a = Array([[[[1]]]])
        self.assertEqual(a.shape, (1, 1, 1, 1))
        self.assertEqual(a.ndim, 4)
        self.assertEqual(a[0,0,0,0], 1)
        
        a = Array([1,2,3,4,5])
        self.assertEqual(a[::-1].data, [5,4,3,2,1])
        self.assertEqual(a[::2].data, [1,3,5])
        self.assertEqual(a[1::2].data, [2,4])
        self.assertEqual(a[-3:].data, [3,4,5])
        
        a = Array([])
        self.assertEqual(a.shape, (0,))
        self.assertEqual(a.size, 0)

        with self.assertRaises(ValidationError):
            a.mean()

        with self.assertRaises(ValidationError):
            a.max()
        
        a = Array([-5,-2,7,-9])
        self.assertEqual(a.min(), -9)
        self.assertEqual(a.max(), 7)
        self.assertEqual(a.argmax(), 2)
        
        a = Array([0.1,0.2,0.3])
        self.assertAlmostEqual(a.sum(), 0.6)
        self.assertAlmostEqual(a.mean(), 0.2)
        
        I = Array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])
        A = Array([
            [3,5,7],
            [2,4,6],
            [8,1,9]
        ])
        self.assertEqual(A.matmul(I).data, A.data)
        self.assertEqual(I.matmul(A).data, A.data)
        
        a = Array([
            [1,2,3],
            [4,5,6]
        ])
        self.assertEqual(a.T.T.data, a.data)
        
        a = Array([1,2,3])
        self.assertEqual(a.dot(a), 14)
        
        a = Array([1,2,3])
        b = Array([4,5])
        c = a.outer(b)
        self.assertEqual(c.shape, (3,2))
        
        a = Array([
            [1,2,3],
            [4,5,6]
        ])
        self.assertEqual(
            a.T.data,
            [
                [1,4],
                [2,5],
                [3,6]
            ]
        )
        
        a = Array([
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ])

        for i in range(3):
            for j in range(3):
                self.assertEqual(a[i,j], i*3+j+1)
    
    def test_invalid_operations_raise(self):
        # ===== Indexing =====
        with self.assertRaises(IndexError):
            Array([1, 2, 3])[5]

        with self.assertRaises(IndexError):
            Array([[1]])[2, 0]

        # ===== item() =====
        with self.assertRaises(IndexError):
            Array([1]).item(5)

        # ===== Inner product =====
        with self.assertRaises(ShapeError):
            Array([1, 2]).inner(Array([1, 2, 3]))

        # ===== Matrix multiplication =====
        with self.assertRaises(ShapeError):
            Array([[1, 2]]).matmul(Array([[1, 2]]))

        # ===== Matrix-vector =====
        with self.assertRaises(ShapeError):
            Array([[1, 2], [3, 4]]).matvec(Array([1]))

        # ===== Broadcasting =====
        with self.assertRaises(ShapeError):
            Array([1, 2]).broadcast_to((3, 3))

        # ===== Transpose =====
        with self.assertRaises(ShapeError):
            Array([[1, 2]]).transpose((0, 0))

        # ===== expand_dims =====
        with self.assertRaises(ShapeError):
            Array([1, 2]).expand_dims(5)

        # ===== Stacking =====
        with self.assertRaises(ShapeError):
            Array([[1, 2]]).vstack(Array([[1, 2, 3]]))

        with self.assertRaises(ShapeError):
            Array([[1, 2]]).hstack(Array([[1], [2]]))


if __name__ == "__main__":
    unittest.main()
