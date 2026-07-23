from structs import Array
from basic.utils import im2col
from debug import get_logger, operation_context

logger = get_logger(__name__)

class Layer():
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.input = None
    def forward(self, input_data): raise NotImplementedError
    def backward(self, output_grad): raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_dim, output_dim, initializer):
        super().__init__()
        self.params['w'] = initializer([output_dim, input_dim], input_dim, output_dim)
        self.params['b'] = Array([0 for _ in range(output_dim)])
    def forward(self, input_data):
        self.input = Array(input_data) if not isinstance(input_data, Array) else input_data
        with operation_context("linear.forward", input_shape=self.input.shape, weight_shape=self.params['w'].shape):
            logger.debug("linear forward %s -> %s", self.params['w'].shape, self.input.shape)
            out = (self.params['w'].matmul(self.input)) + self.params['b']
            return out
    def backward(self, loss_grad):
        grad = Array(loss_grad) if not isinstance(loss_grad, Array) else loss_grad
        self.grads['w'] = grad.outer(self.input)
        self.grads['b'] = grad
        logger.debug("linear backward grad=%s", grad.shape, operation="linear.backward")
        return self.params['w'].T.matmul(grad)

class Conv2D(Layer):
    def __init__(self, input_dim, output_dim, initializer, kernel, stride=[1,1]):
        super().__init__()
        self.params['w'] = initializer([output_dim, input_dim, *kernel], input_dim, output_dim)
        self.params['b'] = Array([0.1] * output_dim)
        self.params['strd'] = stride
        self.params['krnl'] = kernel
        self.input = None
    def forward(self, input):
        self.input = Array.wraparray(input)
        with operation_context("conv2d.forward", input_shape=self.input.shape, kernel=self.params['krnl'], stride=self.params['strd']):
            logger.debug("conv2d forward", input_shape=self.input.shape, kernel=self.params['krnl'], stride=self.params['strd'])
            out = (self.params['w'].dot(Array.wraparray((im2col(self.input, self.params['krnl'], self.params['strd']))), axes=[[1,2,3], [1,2,3]]) + self.params['b']).reshape(3,1,3,3)
            logger.debug("conv2d output", output_shape=out.shape)
            return out
    def backward(self, grad):
        self.grads['w'], self.grads['b'], dX = grad.dot(self.input, [[1], [0]]), grad, grad.dot(self.params['w'], [[0], [0]])
        logger.debug("conv2d backward", grad_shape=getattr(grad, 'shape', None), input_shape=getattr(self.input, 'shape', None))
        return dX

class MaxPool2D(Layer):
    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    from basic.initializers import XavierNormal
    model = Conv2D(1, 3, XavierNormal(), [2, 2], [1,1])
    x = Array([[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]])
    logger.info("forward result %s", model.forward(x), operation="main")
