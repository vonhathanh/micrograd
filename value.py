import math
import random

from visualize import draw_dot


class Value:
    def __init__(self, data, chilren=(), _op='', label=''):
        self.data = data
        self._prev = set(chilren)
        self._op = _op
        self.label = label
        self.grad = 0
        # each node must calculate the grad by itself, this is imposible
        # because it doesn't know its successor in the graph
        # so the solution is parent node calculates the grad for children nodes
        # when _backward() is called, it's chilren grad is calculated
        # this is reasonable, since we already had parent.grad
        self._backward = lambda : None

    def __repr__(self):
        return f"Value(data={self.data}))"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backdward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backdward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backdward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backdward

        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return self * (other**-1)

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supporting integer/float powers for now"

        out = Value(self.data ** power, (self, ), f'**{power}')

        def _backdward():
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backdward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self + (-other)

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backdward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backdward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), chilren=(self, ), _op='exp')

        def _backdward():
            self.grad += out.data * out.grad

        out._backward = _backdward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(1, -1)) for _ in range(nin)]
        self.b = Value(random.uniform(1, -1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == '__main__':
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # desired targets
    ypred = [n(x) for x in xs]
    print(ypred)

    loss = sum([(ygt - ypred)**2 for ygt, ypred in zip(ys, ypred)])

    print(loss)

    loss.backward()

    for p in n.parameters():
        p.data += -0.01 * p.grad

    ypred = [n(x) for x in xs]
    loss = sum([(ygt - ypred) ** 2 for ygt, ypred in zip(ys, ypred)])

    print(loss)

