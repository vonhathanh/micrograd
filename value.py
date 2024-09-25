import math

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

    def __rtruediv__(self, other):
        return self * other**-1

    def __pow__(self, power):
        assert isinstance(power, (int, float), "only supporting integer/float powers for now")

        out = Value(self.data ** power, (self, ), f'**{power}')

        def _backdward():
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backdward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
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


if __name__ == '__main__':
    a = Value(1.0)
    print(a+10, a*10)
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    b = Value(6.8813735870195432, label='b')

    x1w1 = x1 * w1
    x1w1.label = 'x1*w1'

    x2w2 = x2 * w2
    x2w2.label = 'x2*w2'

    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = 'x1w1 + x2w2'

    n = x1w1x2w2 + b; n.label = 'n'

    o = n.tanh()
    o.label = 'o'

    o.grad = 1.0

    o.backward()

    draw_dot(o)
