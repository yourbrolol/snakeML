from .layers import Layer
from structs import Array
from basic.optimizers import SGD
from basic.activations import Softmax
import math

class Embedding(Layer):
    def __init__(self, vocab_size, d_model, table=None):
        super().__init__()
        if table is not None:
            self.params['w'] = Array.wraparray(table)
        else:
            self.params['w'] = Array.randn((vocab_size, d_model))
        self.grads['w'] = Array.zeros((vocab_size, d_model))
    def forward(self, x):
        self.input = x
        return self.params['w'][x]
    def backward(self, grad):
        for token, g in zip(self.input, grad):
            self.grads["w"][token] += g
        return None

class PosEmbedding(Layer):
    def __init__(self, ctx_len, d_model, table=None):
        super().__init__()
        if table is not None:
            self.params['w'] = Array.wraparray(table)
        else:
            self.params['w'] = Array.randn((ctx_len, d_model))
        self.grads['w'] = Array.zeros((ctx_len, d_model))
    def forward(self, x, pos_x):
        self.input = pos_x
        return x + self.params['w'][pos_x]
    def backward(self, grad):
        for pos, g in zip(self.input, grad):
            self.grads["w"][pos] += g
        return grad

class LayerNorm(Layer):
    def __init__(self, input_size):
        super().__init__()
        self.params['w'] = Array.ones((input_size,))
        self.params['b'] = Array.zeros((input_size,))
    def forward(self, input, eps=1e-5):
        self.input = Array.wraparray(input)
        x = self.input
        self.eps = eps
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = ((x - self.mean) ** 2).mean(axis=-1, keepdims=True)
        self.inv_std = 1 / (self.var + eps).sqrt()
        self.x_hat = (x - self.mean) * self.inv_std
        return self.params['w'] * self.x_hat + self.params['b']
    def backward(self, grad):
        # parameter gradients
        self.grads['w'] = (grad * self.x_hat).sum(axis=tuple(range(grad.ndim - 1)))
        self.grads['b'] = grad.sum(axis=tuple(range(grad.ndim - 1)))
        # input gradient
        dx_hat = grad * self.params['w']
        mean1 = dx_hat.mean(axis=-1, keepdims=True)
        mean2 = (dx_hat * self.x_hat).mean(axis=-1, keepdims=True)
        dX = self.inv_std * (dx_hat - mean1 - self.x_hat * mean2)
        return dX

class Residual(Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, fx=None):
        self.input = x
        if fx is None: fx = self.fn.forward(x)
        return x + fx
    def backward(self, grad): return grad + self.fn.backward(grad)

class Attention(Layer):
    def __init__(self, window_size, d_model):
        super().__init__()
        self.params['w'] = {
            "Q": Array.randn((d_model, d_model)),
            "K": Array.randn((d_model, d_model)),
            "V": Array.randn((d_model, d_model)),
        }
        self.grads['w'] = {
            "Q": Array.randn((d_model, d_model)),
            "K": Array.randn((d_model, d_model)),
            "V": Array.randn((d_model, d_model)),
        }
        self.d_model = d_model
        self.softmax = Softmax()
    def forward(self, X):
        Wq, Wk, Wv = self.params['w']["Q"], self.params['w']["K"], self.params['w']["V"]
        Q, K, V = X @ Wq, X @ Wk, X @ Wv
        scores = ((Q @ K.T) / math.sqrt(self.d_model))
        attention = self.softmax.forward(scores)
        self.input, self.Q, self.K, self.V, self.scores, self.attention = X, Q, K, V, scores, attention
        return attention @ V
    def backward(self, grad):
        dA = grad @ self.V.T
        dV = self.attention.T @ grad
        dS = self.softmax.backward(dA)
        dScores = dS / math.sqrt(self.d_model)
        dQ = dScores @ self.K
        dK = dScores.T @ self.Q
        dXq = dQ @ self.params['w']["Q"].T
        dWq = self.input.T @ dQ
        dXk = dK @ self.params['w']["K"].T
        dWk = self.input.T @ dK
        dXv = dV @ self.params['w']["V"].T
        dWv = self.input.T @ dV
        dX = dXq + dXk + dXv
        self.grads['w']["Q"], self.grads['w']["K"], self.grads['w']["V"] = dWq, dWk, dWv
        return dX

class MHAttention(Layer):
    def __init__(self, window_len, d_model, num_heads=8):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"Cannot split d_model={d_model} over num_heads={num_heads}!"
            )

        self.window_len = window_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.params["w"] = {
            "Q": Array.randn((num_heads, d_model, self.head_dim)),
            "K": Array.randn((num_heads, d_model, self.head_dim)),
            "V": Array.randn((num_heads, d_model, self.head_dim)),
            "O": Array.randn((d_model, d_model)),
        }
        
        self.softmax = Softmax()
    def forward(self, x):
        S, D = x.shape

        # (S, H, Dh)
        Q = x.dot(self.params["w"]["Q"], axes=([1], [1]))
        K = x.dot(self.params["w"]["K"], axes=([1], [1]))
        V = x.dot(self.params["w"]["V"], axes=([1], [1]))

        heads = []

        for h in range(self.num_heads):
            q = Q[:, h, :]          # (S, Dh)
            k = K[:, h, :]          # (S, Dh)
            v = V[:, h, :]          # (S, Dh)

            # (S, S)
            scores = q @ k.transpose()
            scores /= math.sqrt(self.head_dim)

            weights = self.softmax.forward(scores)

            # (S, Dh)
            heads.append(weights @ v)

        # (S, H, Dh)
        out = Array.stack(heads, axis=1)

        # (S, D)
        out = out.reshape(S, D)

        # (S, D)
        out = out @ self.params['w']["O"]

        return out
    def backward(self, grad):
        pass

if __name__ == "__main__":
    att = MHAttention(4, 16)
    print(att.forward(Array.randn((4, 16))))
    # print(att.backward(Array.randn((16, 4))))