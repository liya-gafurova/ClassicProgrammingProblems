import random
from enum import IntEnum, Enum
from typing import Tuple, List, Optional, TypeVar, Iterable, Protocol, Any, NamedTuple, Generic, Callable, Set, Deque, \
    Dict
from .data_structures import Stack, C, T, Node, Queue, PriorityQueue

Nucleotide: IntEnum = IntEnum("Nucleotide", ["A", "C", "G", "T"])

Codon = Tuple[Nucleotide, Nucleotide, Nucleotide]  # type alias
Gene = List[Codon]  # type alias


class Comparable(Protocol):
    def __eq__(self, other: Any) -> bool:
        ...

    def __lt__(self: C, other: C) -> bool:
        ...

    def __gt__(self: C, other: C) -> bool:
        return (not self < other) and self != other

    def __le__(self: C, other: C) -> bool:
        return self < other or self == other

    def __ge__(self: C, other: C) -> bool:
        return not self < other


def string_to_gene(gene_str: str) -> Gene:
    gene: Gene = []
    len_of_gene_str = len(gene_str)
    for i in range(0, len_of_gene_str, 3):
        if i + 2 >= len_of_gene_str:
            return gene
        codon: Codon = (Nucleotide[gene_str[i]],
                        Nucleotide[gene_str[i + 1]],
                        Nucleotide[gene_str[i + 2]])
        gene.append(codon)
    return gene


def linear_search(gene: Gene, codon_to_be_found: Codon):
    for codon in gene:
        if codon == codon_to_be_found:
            return True
    return False


# # TODO ASK. Recursion -> going down / going up for returning value
# def binary_search(gene_str_sorted: str, codon_to_be_found: Codon):
#     global RESULT_VAL
#     mid_ind = len(gene_str_sorted) // 2
#     if mid_ind == 0:
#         RESULT_VAL = False
#         return RESULT_VAL
#     if gene_str_sorted[mid_ind] == codon_to_be_found:
#         RESULT_VAL = True
#         return RESULT_VAL
#     elif codon_to_be_found < gene_str_sorted[mid_ind]:
#         binary_search(gene_str_sorted[:mid_ind], codon_to_be_found)
#     elif codon_to_be_found > gene_str_sorted[mid_ind]:
#         binary_search(gene_str_sorted[mid_ind:], codon_to_be_found)
#     return RESULT_VAL


# DONE
def binary_search(gene_str_sorted: str, codon_to_be_found: Codon):
    mid_ind = len(gene_str_sorted) // 2
    if mid_ind == 0:
        return False
    if gene_str_sorted[mid_ind] == codon_to_be_found:
        return True
    elif codon_to_be_found < gene_str_sorted[mid_ind]:
        r = binary_search(gene_str_sorted[:mid_ind], codon_to_be_found)
    elif codon_to_be_found > gene_str_sorted[mid_ind]:
        r = binary_search(gene_str_sorted[mid_ind:], codon_to_be_found)
    return r


def linear_contains(iterable: Iterable[T], key: T):
    for item in iterable:
        if item == key:
            return True
    return False


def binary_contains(sorted_sequence: Iterable[C], key: C):
    start: int = 0
    end: int = len(sorted_sequence) - 1
    while start <= end:
        mid_id = (start + end) // 2
        if key > sorted_sequence[mid_id]:
            start = mid_id + 1
        elif key < sorted_sequence[mid_id]:
            end = mid_id - 1
        else:
            return True
    return False


class Cell(str, Enum):
    EMPTY = " "
    BLOCKED = "X"
    START = "S"
    GOAL = "G"
    PATH = "*"


class MazeLocation(NamedTuple):
    row: int
    column: int


class Maze:
    def __init__(self, rows: int = 10, columns: int = 10, sparseness: float =
    0.2, start: MazeLocation = MazeLocation(0, 0), goal: MazeLocation =
                 MazeLocation(9, 9)) -> None:
        # initialize basic instance variables
        self._rows: int = rows
        self._columns: int = columns
        self.start: MazeLocation = start
        self.goal: MazeLocation = goal
        # fill the grid with empty cells
        self._grid: List[List[Cell]] = [[Cell.EMPTY for c in range(columns)] for r in range(rows)]
        # populate the grid with blocked cells
        self._randomly_fill(rows, columns, sparseness)
        # fill the start and goal locations in
        self._grid[start.row][start.column] = Cell.START
        self._grid[goal.row][goal.column] = Cell.GOAL

    def _randomly_fill(self, rows: int, columns: int, sparseness: float):
        for row in range(rows):
            for column in range(columns):
                if random.uniform(0, 1.0) < sparseness:
                    self._grid[row][column] = Cell.BLOCKED

    # return a nicely formatted version of the maze for printing
    def __str__(self) -> str:
        output: str = ""
        for row in self._grid:
            output += "| ".join([c.value for c in row]) + "\n"
            output += '-' * 30 + '\n'
        return output

    def goal_test(self, ml: MazeLocation) -> bool:
        return ml == self.goal

    def successors(self, ml: MazeLocation) -> List[MazeLocation]:
        locations: List[MazeLocation] = []

        if ml.row + 1 < self._rows and self._grid[ml.row + 1][ml.column] != Cell.BLOCKED:
            locations.append(MazeLocation(ml.row + 1, ml.column))
        if ml.row - 1 >= 0 and self._grid[ml.row - 1][ml.column] != Cell.BLOCKED:
            locations.append(MazeLocation(ml.row - 1, ml.column))
        if ml.column + 1 < self._columns and self._grid[ml.row][ml.column + 1] != Cell.BLOCKED:
            locations.append(MazeLocation(ml.row, ml.column + 1))
        if ml.column - 1 >= 0 and self._grid[ml.row][ml.column - 1] != Cell.BLOCKED:
            locations.append(MazeLocation(ml.row, ml.column - 1))
        return locations

    def mark(self, path: List[MazeLocation]):
        for maze_location in path:
            self._grid[maze_location.row][maze_location.column] = Cell.PATH
            self._grid[self.start.row][self.start.column] = Cell.START
            self._grid[self.goal.row][self.goal.column] = Cell.GOAL

    def clear(self, path: List[MazeLocation]):
        for maze_location in path:
            self._grid[maze_location.row][maze_location.column] = Cell.EMPTY
        self._grid[self.start.row][self.start.column] = Cell.START
        self._grid[self.goal.row][self.goal.column] = Cell.GOAL


def dfs(initial: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]]) -> Tuple[Optional[Node[T]], int]:
    # frontier is where we've yet to go
    frontier: Stack[Node[T]] = Stack()
    frontier.push(Node(initial, None))
    # explored is where we've been
    explored: Set[T] = {initial}
    # keep going while there is more to explore
    state_counter = 0
    while not frontier.empty:
        current_node: Node[T] = frontier.pop()
        current_state: T = current_node.state
        # if we found the goal, we're done
        state_counter += 1
        if goal_test(current_state):
            return current_node, state_counter
        # check where we can go next and haven't explored
        for child in successors(current_state):
            if child in explored:  # skip children we already explored
                continue
            explored.add(child)
            frontier.push(Node(child, current_node))
    return None, state_counter  # went through everything and never found goal


def node_to_path(node: Node[T]) -> List[T]:
    path: List[T] = [node.state]
    # work backwards from end to front
    while node.parent is not None:
        node = node.parent
        path.append(node.state)
    path.reverse()
    return path


def bfs(initial: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]]) -> Tuple[Optional[Node[T]], int]:
    # frontier is where we've yet to go
    frontier: Queue[Node[T]] = Queue()
    frontier.push(Node(initial, None))
    # explored is where we've been
    explored: Set[T] = {initial}
    # keep going while there is more to explore
    state_counter =0
    while not frontier.empty:
        current_node: Node[T] = frontier.pop()
        current_state: T = current_node.state
        # if we found the goal, we're done
        state_counter +=1
        if goal_test(current_state):
            return current_node,state_counter
        # check where we can go next and haven't explored
        for child in successors(current_state):
            if child in explored:  # skip children we already explored
                continue
            explored.add(child)
            frontier.push(Node(child, current_node))
    return None, state_counter  # went through everything and never found goal


def manhattan_distance(goal: MazeLocation) -> Callable[[MazeLocation], float]:
    def distance(ml: MazeLocation) -> float:
        xdist: int = abs(ml.column - goal.column)
        ydist: int = abs(ml.row - goal.row)
        return (xdist + ydist)

    return distance


def astar(initial: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]],
          heuristic: Callable[[T], float]) -> Tuple[Optional[Node[T]], int]:
    # frontier is where we've yet to go
    frontier: PriorityQueue[Node[T]] = PriorityQueue()
    frontier.push(Node(initial, None, 0.0, heuristic(initial)))
    # explored is where we've been
    explored: Dict[T, float] = {initial: 0.0}
    # keep going while there is more to explore
    state_counter:int = 0
    while not frontier.empty:
        current_node: Node[T] = frontier.pop()
        current_state: T = current_node.state
        # if we found the goal, we're done
        state_counter += 1
        if goal_test(current_state):
            return current_node, state_counter
        # check where we can go next and haven't explored
        for child in successors(current_state):
            new_cost: float = current_node.cost + 1  # 1 assumes a grid, need a cost function for more sophisticated apps
            if child not in explored or explored[child] > new_cost:
                explored[child] = new_cost
                frontier.push(Node(child, current_node, new_cost, heuristic(child)))
                print('')

    return None, state_counter  # went through everything and never found goal
