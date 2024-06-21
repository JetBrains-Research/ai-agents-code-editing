def foo(a: int, b: int) -> int:
    return a + b


def bar(a: int, b: int) -> int:
    return a * b


def baz(a: int, b: int) -> int:
    return a - b


class A:
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b

    def foo(self) -> int:
        return self.a + self.b

    def bar(self) -> int:
        return self.a * self.b

    def baz(self) -> int:
        return self.a - self.b
