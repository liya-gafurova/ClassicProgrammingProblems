# The Fibonacci sequence is a sequence of numbers such that any number, except
# for the first and second, is the sum of the previous two:
# 0, 1, 1, 2, 3, 5, 8, 13, 21...
# The value of the first Fibonacci number in the sequence is 0 . The value of the
# fourth Fibonacci number is 2 . It follows that to get the value of any Fibonacci num-
# ber, n , in the sequence, one can use the formula
# fib(n) = fib(n - 1) + fib(n - 2)
from typing import Generator
from functools import lru_cache
from typing import TypeVar, Generic, List


def factorial(m):
    return m * factorial(m - 1) if m != 1 else m


def FibonacciRecursive(n):
    pass


@lru_cache(maxsize=None)
def fib4(n: int) -> int:
    # Outpusts single value of the whole sequence in the end
    if n < 2:
        return n
    return fib4(n - 2) + fib4(n - 1)  # recursive case


def fib6(n: int) -> Generator[int, None, None]:
    # iterative approach + generator
    yield 0
    if n > 0: yield 1
    last: int = 0
    next: int = 1
    for _ in range(1, n):
        last, next = next, last + next
        yield next  # main generation step


# TASK #2
#  Hanoi Tower

T = TypeVar('T')


class Stack(Generic[T]):
    def __init__(self, name) -> None:
        self.tower_name = name
        self._container: List[T] = []

    def push(self, item: T) -> None:
        self._container.append(item)

    def pop(self) -> T:
        return self._container.pop()

    def last(self):
        try:
            return self._container[self._count() - 1]
        except Exception as ex:
            None

    def _count(self):
        return len(self._container)

    def __repr__(self) -> str:
        return repr(f"{self.tower_name}: {self._container}")


def hanoi(begin: Stack[int], temp: Stack[int], end: Stack[int], n: int) -> None:
    if n == 1:
        print(f"Form tower {begin} put disk on tower {end}")
        end.push(begin.pop())
    else:

        hanoi(begin, end, temp, n - 1)
        hanoi(begin, temp, end, 1)
        hanoi(temp, begin, end, n - 1)


def towerOfHanoi(begin, temp1, temp2, end, n):
    if n == 0:
        return
    if n == 1:
        print(f"Form tower {begin} put disk on tower {end}")
        end.push(begin.pop())
        return
    else:
        towerOfHanoi(begin, temp1, end, temp1, n - 2)
        towerOfHanoi(begin, temp1, end, temp2, 1)
        towerOfHanoi(begin, temp1, temp2, end, 1)
        towerOfHanoi(temp2, begin, temp1, end, 1)
        towerOfHanoi(temp1, begin, temp2, end, n - 2)
