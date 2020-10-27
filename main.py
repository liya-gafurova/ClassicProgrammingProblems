from typing import List

from small_problems.lib import fib6, fib4, factorial, Stack, hanoi, towerOfHanoi

# Fibonaccci Secuence
#  recursive, cache, iterative
for i in fib6(50):
    print(i)

print(fib4(50))

print(factorial(5))


# ====================================================================
# Hanoi tower


def show_towers(towers: List[Stack[str]]) -> None:
    for tower in towers:
        print(tower)
    print("------------")


def test_hanoi_tower(num_disks=3, num_towers=3):
    TOWER_NAMES = ["A", "B", "C", "D", "E", 'F']
    towers = [Stack(f"{TOWER_NAMES[i]}") for i in range(num_towers)]

    for i in range(num_disks, 0, -1):
        towers[0].push(i)

    show_towers(towers)
    if num_towers > 3:
        towerOfHanoi(*towers, n=num_disks)
    else:
        hanoi(*towers, n=num_disks)
    show_towers(towers)


test_hanoi_tower(4, 4)
