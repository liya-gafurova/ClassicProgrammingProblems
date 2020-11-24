from typing import List, Optional, Callable

from small_problems.lib import fib6, fib4, factorial, Stack, hanoi, towerOfHanoi
from search_problems.lib import Codon, Gene, Nucleotide, linear_search, string_to_gene, binary_search, linear_contains, \
    binary_contains, Maze, dfs, Node, MazeLocation, node_to_path, bfs, manhattan_distance, astar


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

# ==================================================
# search algorithms
# ==================================================

gene_str: str = "ACGTGGCTCTCTAACGTACGTACGTACGGGGTTTATATATACCCTAGGACTCCCTTT"

ttt: Codon = (Nucleotide.T, Nucleotide.T, Nucleotide.T)
agt: Codon = (Nucleotide.A, Nucleotide.G, Nucleotide.T)
gene = string_to_gene(gene_str)
## LINEAR SEARCH

# linear search for Gene type
print(f"Linear search for Gene.")
print(f"Looking for codon: {ttt} --> {linear_search(gene, ttt)}")
print(f"Looking for codon: {agt} --> {linear_search(gene, agt)}\n")

# linear search for UNIVERSAL type
print(f"Linear search for Universal type.")
print(f"Looking for codon: {ttt} --> {linear_contains(gene, ttt)}")
print(f"Looking for codon: {agt} --> {linear_contains(gene, agt)}\n")

# BINARY SEARCH. Sort the search source, ONLY then make search in sorted array


gene_sorted = sorted(gene)

print(f"Binary search for Gene.")
print(f"Looking for codon: {ttt} --> {binary_search(gene_sorted, ttt)}")
print(f"Looking for codon: {agt} --> {binary_search(gene_sorted, agt)}\n")

print(f"Binary search for Universal type.")
print(f"Looking for codon: {ttt} --> {binary_contains(gene_sorted, ttt)}")
print(f"Looking for codon: {agt} --> {binary_contains(gene_sorted, agt)}")
print(f"{binary_contains(sorted(['kate', 'mary', 'julia' , 'lia', 'kevin']), 'maryss')}")
print(linear_contains([1, 5, 15, 15, 15, 15, 20], 5)) # True
print(binary_contains(["a", "d", "e", "f", "z"], "f")) # True
print(binary_contains(sorted(["john", "mark", "ronald", "sarah"]), "sheila"))

# ==================================================
# Maze solving
# ==================================================

# Test DFS
m: Maze = Maze()
print(m)
solution1: Optional[Node[MazeLocation]] = dfs(m.start, m.goal_test,m.successors)
if solution1 is None:
    print("No solution found using depth-first search!")
else:
    path1: List[MazeLocation] = node_to_path(solution1)
    m.mark(path1)
    print(m)
    m.clear(path1)

# Test BFS
solution2: Optional[Node[MazeLocation]] = bfs(m.start, m.goal_test, m.successors)
if solution2 is None:
    print("No solution found using breadth-first search!")
else:
    path2: List[MazeLocation] = node_to_path(solution2)
    m.mark(path2)
    print(m)
    m.clear(path2)

# Test A*
distance: Callable[[MazeLocation], float] = manhattan_distance(m.goal)
solution3: Optional[Node[MazeLocation]] = astar(m.start, m.goal_test, m.successors, distance)
if solution3 is None:
    print("No solution found using A*!")
else:
    path3: List[MazeLocation] = node_to_path(solution3)
    m.mark(path3)
    print(m)