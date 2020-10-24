from small_problems.lib import fib6, fib4, factorial, Stack, hanoi

# Fibonaccci Secuence
#  recursive, cache, iterative
for i in fib6(50):
    print(i)

print(fib4(50))

print(factorial(5))
# ====================================================================
# Hanoi tower
num_discs: int = 3
tower_a: Stack[str] = Stack()
tower_b: Stack[str] = Stack()
tower_c: Stack[str] = Stack()

for i in range(num_discs , 0, -1):
    tower_a.push(f"item _{i}")

print(tower_a)
print(tower_b)
print(tower_c)
hanoi(begin=tower_a, end=tower_c, temp=tower_c, n = 3)
print(tower_a)
print(tower_b)
print(tower_c)

# returns
# ['item _3', 'item _2', 'item _1']
# []
# []
# -----------
# []
# []
# ['item _1', 'item _2', 'item _3']  - ERROR