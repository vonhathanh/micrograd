from visualize import draw_dot


class Value:
    def __init__(self, data, chilren=(), _op='', label=''):
        self.data = data
        self._prev = chilren
        self._op = _op
        self.label = label
        self.grad = 0

    def __repr__(self):
        return f"Value(data={self.data}))"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')


if __name__ == '__main__':
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')

    d = a*b + c
    print(d, d._prev, d._op)

    draw_dot(d)