import unittest

from basic.initializers import (
    KaimingNormal,
    KaimingUniform,
    Normal,
    Ones,
    Uniform,
    XavierUniform,
    Zeros,
)


class InitializerTests(unittest.TestCase):
    def test_zeros_and_ones_fill_arrays(self):
        zeros = Zeros()([2, 2])
        ones = Ones()([2, 2])
        self.assertEqual(zeros.data, [[0, 0], [0, 0]])
        self.assertEqual(ones.data, [[1, 1], [1, 1]])

    def test_random_initializers_return_expected_shape(self):
        normal_values = Normal()([2, 2], mean=1.0, std=0.5)
        uniform_values = Uniform()([2, 2], low=-1.0, high=1.0)
        xavier_values = XavierUniform()([2, 2], 4, 8)
        kaiming_uniform = KaimingUniform()([2, 2], 4)
        kaiming_normal = KaimingNormal()([2, 2], 4)

        self.assertEqual(normal_values.shape, (2, 2))
        self.assertEqual(uniform_values.shape, (2, 2))
        self.assertEqual(xavier_values.shape, (2, 2))
        self.assertEqual(kaiming_uniform.shape, (2, 2))
        self.assertEqual(kaiming_normal.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
