# The Fibonacci sequence is a sequence of numbers such that any number, except
# for the first and second, is the sum of the previous two:
# 0, 1, 1, 2, 3, 5, 8, 13, 21...
# The value of the first Fibonacci number in the sequence is 0 . The value of the
# fourth Fibonacci number is 2 . It follows that to get the value of any Fibonacci num-
# ber, n , in the sequence, one can use the formula
# fib(n) = fib(n - 1) + fib(n - 2)
from typing import Generator
from functools import lru_cache


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


for i in fib6(50):
    print(i)

print(fib4(50))

print(factorial(5))
