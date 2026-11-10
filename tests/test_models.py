import math
import unittest

from basic.activations import (
    GELU,
    LinearActivation,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
)
from basic.criterions import MSE, CrossEntropy
from basic.layers import Conv2D, Layer, Linear, MaxPool2D
from basic.nn import LayerList, Sequential
from basic.optimizers import SGD, Adam
from basic.transformers import Attention, Embedding, LayerNorm, PosEmbedding, Residual
from basic.utils import im2col
from structs import Array


def _set(linear, w, b):
    linear.params["w"] = Array(w)
    linear.params["b"] = Array(b)
    return linear


class LayerTests(unittest.TestCase):
    def test_base_layer_contract_is_abstract(self):
        layer = Layer()
        self.assertEqual(layer.params, {})
        self.assertEqual(layer.grads, {})
        self.assertEqual(layer.parameters(), {})
        with self.assertRaises(NotImplementedError):
            layer.forward([])
        with self.assertRaises(NotImplementedError):
            layer.backward([])

    def test_parameters_recurses_into_submodules(self):
        parent = Layer()
        child = _set(Linear(2, 1), [[1.0, 2.0]], [0.0])
        parent.submodules["child"] = child

        params = parent.parameters()

        self.assertIn("child.w", params)
        self.assertIn("child.b", params)
        self.assertEqual(params["child.w"].tolist(), [[1.0, 2.0]])


class LinearTests(unittest.TestCase):
    def test_forward_computes_wx_plus_b(self):
        layer = _set(Linear(3, 2), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0.5, -0.5])

        out = layer.forward(Array([1.0, 1.0, 1.0]))

        self.assertEqual(out.shape, (2,))
        self.assertAlmostEqual(out.tolist()[0], 6.5)
        self.assertAlmostEqual(out.tolist()[1], 14.5)

    def test_backward_populates_parameter_and_input_gradients(self):
        layer = _set(Linear(3, 2), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0.5, -0.5])
        layer.forward(Array([1.0, 1.0, 1.0]))

        dx = layer.backward(Array([1.0, 1.0]))

        self.assertEqual(layer.grads["w"].tolist(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        self.assertEqual(layer.grads["b"].tolist(), [1.0, 1.0])
        self.assertEqual(dx.tolist(), [5.0, 7.0, 9.0])

    def test_default_initialization_has_expected_shapes(self):
        layer = Linear(4, 3)
        self.assertEqual(layer.params["w"].shape, (3, 4))
        self.assertEqual(layer.params["b"].shape, (3,))
        self.assertEqual(layer.params["b"].tolist(), [0.0, 0.0, 0.0])


class Conv2DTests(unittest.TestCase):
    def test_im2col_extracts_every_patch_in_row_major_order(self):
        image = Array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])

        patches = im2col(image, [2, 2], [1, 1])

        self.assertEqual(len(patches), 4)
        self.assertEqual(patches[0].tolist(), [[[1.0, 2.0], [4.0, 5.0]]])
        self.assertEqual(patches[-1].tolist(), [[[5.0, 6.0], [8.0, 9.0]]])

    def test_im2col_returns_empty_when_kernel_exceeds_input(self):
        image = Array([[[1.0, 2.0], [3.0, 4.0]]])
        self.assertEqual(im2col(image, [3, 3], [1, 1]), [])

    def test_forward_valid_convolution_shape_and_values(self):
        layer = Conv2D(1, 1, [2, 2], [1, 1])
        layer.params["w"] = Array([[[[1.0, 0.0], [0.0, -1.0]]]])
        layer.params["b"] = Array([0.0])
        image = Array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])

        out = layer.forward(image)

        self.assertEqual(out.shape, (1, 1, 2, 2))
        self.assertEqual(out.tolist(), [[[[-4.0, -4.0], [-4.0, -4.0]]]])

    def test_conv_parameter_shapes(self):
        layer = Conv2D(3, 5, [3, 2], [2, 1])
        self.assertEqual(layer.params["w"].shape, (5, 3, 3, 2))
        self.assertEqual(layer.params["b"].shape, (5,))
        self.assertEqual(layer.params["strd"], [2, 1])
        self.assertEqual(layer.params["krnl"], [3, 2])

    def test_backward_returns_gradient_with_input_shape(self):
        layer = Conv2D(1, 1, [2, 2], [1, 1])
        layer.params["w"] = Array([[[[1.0, 0.0], [0.0, -1.0]]]])
        layer.params["b"] = Array([0.0])
        image = Array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
        out = layer.forward(image)

        dx = layer.backward(Array.ones(out.shape))

        self.assertEqual(dx.shape, image.shape)
        self.assertEqual(layer.grads["w"].shape, layer.params["w"].shape)
        self.assertEqual(layer.grads["b"].shape, layer.params["b"].shape)

    def test_maxpool_is_not_implemented_yet(self):
        with self.assertRaises(NotImplementedError):
            MaxPool2D().forward(Array([[[1.0]]]))


class ActivationTests(unittest.TestCase):
    def test_linear_activation_is_identity(self):
        layer = LinearActivation()
        x = Array([1.0, -2.0, 3.0])
        self.assertEqual(layer.forward(x).tolist(), x.tolist())
        self.assertEqual(layer.backward(x).tolist(), x.tolist())

    def test_relu_forward_is_elementwise(self):
        out = ReLU().forward(Array([-1.0, 0.0, 2.0]))
        self.assertEqual(out.tolist(), [0.0, 0.0, 2.0])

    def test_relu_backward_masks_non_positive_inputs(self):
        layer = ReLU()
        layer.forward(Array([-1.0, 0.0, 2.0]))
        self.assertEqual(layer.backward(Array([1.0, 1.0, 1.0])).tolist(), [0.0, 0.0, 1.0])

    def test_sigmoid_forward_matches_reference(self):
        layer = Sigmoid()
        out = layer.forward(Array([-1.0, 0.0, 1.0])).tolist()
        expected = [1 / (1 + math.e), 0.5, 1 / (1 + math.exp(-1))]
        for got, want in zip(out, expected):
            self.assertAlmostEqual(got, want)

    def test_sigmoid_backward_uses_forward_output(self):
        layer = Sigmoid()
        out = layer.forward(Array([0.0, 1.0]))
        grad = layer.backward(Array([1.0, 1.0])).tolist()
        for got, p in zip(grad, out.tolist()):
            self.assertAlmostEqual(got, p * (1 - p))

    def test_tanh_forward_and_backward(self):
        layer = Tanh()
        out = layer.forward(Array([0.0, 1.0])).tolist()
        self.assertAlmostEqual(out[0], 0.0)
        self.assertAlmostEqual(out[1], math.tanh(1.0))

        grad = layer.backward(Array([1.0, 1.0])).tolist()
        self.assertAlmostEqual(grad[0], 1.0)
        self.assertAlmostEqual(grad[1], 1 - math.tanh(1.0) ** 2)

    def test_gelu_forward_and_backward_match_tanh_approximation(self):
        layer = GELU()
        out = layer.forward(Array([0.0, 1.0])).tolist()
        self.assertAlmostEqual(out[0], 0.0)
        self.assertAlmostEqual(out[1], 0.8411919906082768)

        grad = layer.backward(Array([1.0, 1.0])).tolist()
        self.assertAlmostEqual(grad[1], 1.0829640838457826)

    def test_softmax_1d_normalizes(self):
        out = Softmax().forward(Array([1.0, 2.0, 3.0])).tolist()
        self.assertAlmostEqual(sum(out), 1.0)
        self.assertAlmostEqual(out[0], 0.09003057317038046)
        self.assertAlmostEqual(out[2], 0.6652409557748218)

    def test_softmax_2d_normalizes_each_row(self):
        out = Softmax().forward(Array([[1.0, 2.0], [3.0, 6.0]])).tolist()
        self.assertAlmostEqual(sum(out[0]), 1.0)
        self.assertAlmostEqual(sum(out[1]), 1.0)
        self.assertAlmostEqual(out[1][0], 0.04742587317756679)

    def test_softmax_backward_jacobian_vector_product(self):
        layer = Softmax()
        p = layer.forward(Array([1.0, 2.0, 3.0]))
        grad = layer.backward(Array([1.0, 0.0, 0.0])).tolist()

        expected = [
            p.tolist()[i] * ((1.0 if i == 0 else 0.0) - p.tolist()[0])
            for i in range(3)
        ]
        for got, want in zip(grad, expected):
            self.assertAlmostEqual(got, want)


class CriterionTests(unittest.TestCase):
    def test_mse_forward_is_elementwise_half_squared_error(self):
        loss = MSE().forward(Array([1.0, 2.0]), Array([1.0, 1.0]))
        self.assertEqual(loss.tolist(), [0.0, 0.5])

    def test_mse_backward_is_prediction_minus_target(self):
        grad = MSE().backward(Array([1.0, 2.0]), Array([1.0, 1.0]))
        self.assertEqual(grad.tolist(), [0.0, 1.0])

    def test_cross_entropy_forward_is_mean_negative_log_likelihood(self):
        loss = CrossEntropy().forward(Array([0.7, 0.3]), Array([1.0, 0.0]))
        self.assertAlmostEqual(loss, -math.log(0.7) / 2)

    def test_cross_entropy_backward_is_prediction_minus_target_over_n(self):
        grad = CrossEntropy().backward(Array([0.7, 0.3]), Array([1.0, 0.0]))
        self.assertAlmostEqual(grad.tolist()[0], -0.15)
        self.assertAlmostEqual(grad.tolist()[1], 0.15)


class OptimizerTests(unittest.TestCase):
    def _linear_with_grads(self):
        layer = _set(Linear(2, 2), [[1.0, 1.0], [1.0, 1.0]], [0.0, 0.0])
        layer.grads["w"] = Array([[1.0, 1.0], [1.0, 1.0]])
        layer.grads["b"] = Array([1.0, 1.0])
        return layer

    def test_sgd_step_applies_learning_rate_times_gradient(self):
        layer = self._linear_with_grads()
        SGD([layer], lr=0.1).step()
        self.assertEqual(layer.params["w"].tolist(), [[0.9, 0.9], [0.9, 0.9]])
        self.assertEqual(layer.params["b"].tolist(), [-0.1, -0.1])

    def test_adam_first_step_uses_bias_corrected_update(self):
        layer = self._linear_with_grads()
        Adam([layer], lr=0.1).step()
        for row in layer.params["w"].tolist():
            for value in row:
                self.assertAlmostEqual(value, 0.900000001)
        for value in layer.params["b"].tolist():
            self.assertAlmostEqual(value, -0.099999999)

    def test_zero_grad_clears_layer_gradients(self):
        layer = self._linear_with_grads()
        optimizer = SGD([layer], lr=0.1)
        optimizer.zero_grad()
        self.assertEqual(layer.grads, {})

    def test_step_skips_layers_without_gradients(self):
        layer = Linear(2, 2)
        before = layer.params["w"].tolist()
        SGD([layer], lr=0.1).step()
        self.assertEqual(layer.params["w"].tolist(), before)


class SequentialTests(unittest.TestCase):
    def _model(self):
        first = _set(Linear(3, 4), Array.ones((4, 3)).tolist(), [0.0] * 4)
        second = _set(Linear(4, 2), Array.ones((2, 4)).tolist(), [0.0] * 2)
        return Sequential([first, Tanh(), second]), first, second

    def test_forward_applies_layers_in_order(self):
        model, _, _ = self._model()
        out = model.forward(Array([1.0, 2.0, 3.0]))
        self.assertEqual(out.shape, (2,))

    def test_backward_applies_layers_in_reverse_and_populates_grads(self):
        model, first, second = self._model()
        model.forward(Array([1.0, 2.0, 3.0]))

        dx = model.backward(Array([1.0, 1.0]))

        self.assertEqual(dx.shape, (3,))
        self.assertEqual(second.grads["w"].shape, (2, 4))
        self.assertEqual(first.grads["w"].shape, (4, 3))

    def test_empty_sequence_is_identity(self):
        model = Sequential([])
        x = Array([1.0, 2.0])
        self.assertEqual(model.forward(x).tolist(), x.tolist())


class LayerListTests(unittest.TestCase):
    def test_parameters_are_keyed_by_index(self):
        layers = LayerList([Linear(2, 2), Linear(2, 1)])
        self.assertEqual(
            sorted(layers.parameters().keys()),
            ["0.b", "0.w", "1.b", "1.w"],
        )

    def test_attribute_access_returns_layer(self):
        layers = LayerList([Linear(2, 2)])
        self.assertIsInstance(layers.submodules["0"], Linear)
        self.assertIs(layers.__getattr__("0"), layers[0])

    def test_unknown_attribute_raises(self):
        with self.assertRaises(AttributeError):
            LayerList([Linear(2, 2)]).missing


class TransformerTests(unittest.TestCase):
    def test_embedding_forward_looks_up_rows(self):
        layer = Embedding(3, 2, table=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = layer.forward([0, 2])
        self.assertEqual(out.tolist(), [[1.0, 2.0], [5.0, 6.0]])

    def test_embedding_backward_accumulates_token_gradients(self):
        layer = Embedding(3, 2, table=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        layer.forward([1, 2])
        layer.backward(Array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(layer.grads["w"].tolist(), [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])

    def test_positional_embedding_adds_table_rows(self):
        layer = PosEmbedding(3, 2, table=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        out = layer.forward(Array([[10.0, 20.0], [30.0, 40.0]]), [0, 2])
        self.assertEqual(out.tolist(), [[11.0, 21.0], [33.0, 43.0]])

    def test_positional_embedding_backward_scatters_and_passes_gradient(self):
        layer = PosEmbedding(3, 2, table=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        layer.forward(Array([[0.0, 0.0], [0.0, 0.0]]), [0, 2])
        grad = Array([[1.0, 2.0], [3.0, 4.0]])
        out = layer.backward(grad)
        self.assertEqual(out.tolist(), grad.tolist())
        self.assertEqual(layer.grads["w"].tolist(), [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])

    def test_layer_norm_normalizes_last_axis_and_scales(self):
        layer = LayerNorm(3)
        layer.params["w"] = Array([2.0, 2.0, 2.0])
        layer.params["b"] = Array([1.0, 1.0, 1.0])
        out = layer.forward(Array([[1.0, 2.0, 3.0]]))
        self.assertAlmostEqual(out.tolist()[0][0], 2 * (-1.2247356859083902) + 1)
        self.assertAlmostEqual(out.tolist()[0][1], 1.0)
        self.assertAlmostEqual(out.tolist()[0][2], 2 * 1.2247356859083902 + 1)

    def test_layer_norm_backward_returns_input_gradient(self):
        layer = LayerNorm(3)
        layer.forward(Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        dx = layer.backward(Array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        self.assertEqual(dx.shape, (2, 3))
        self.assertEqual(layer.grads["w"].shape, (3,))
        self.assertEqual(layer.grads["b"].shape, (3,))

    def test_residual_adds_branch_and_backpropagates_both_paths(self):
        inner = _set(Linear(3, 3), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0])
        layer = Residual(inner)
        x = Array([1.0, 2.0, 3.0])

        out = layer.forward(x)
        dx = layer.backward(Array([1.0, 1.0, 1.0]))

        self.assertEqual(out.tolist(), [2.0, 4.0, 6.0])
        self.assertEqual(dx.tolist(), [2.0, 2.0, 2.0])

    def test_attention_forward_produces_attended_values(self):
        layer = Attention(2, 4)
        out = layer.forward(Array.randn((2, 4)))
        self.assertEqual(out.shape, (2, 4))

        row_sums = [sum(row) for row in layer.attention.tolist()]
        for total in row_sums:
            self.assertAlmostEqual(total, 1.0)

    def test_attention_backward_populates_projection_gradients(self):
        layer = Attention(2, 4)
        layer.forward(Array.randn((2, 4)))
        dx = layer.backward(Array.randn((2, 4)))

        self.assertEqual(dx.shape, (2, 4))
        for key in ("Q", "K", "V"):
            self.assertEqual(layer.grads["w"][key].shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
