# Search problems

1. DNA search 
2. Maze solving 
3. Missionaries and cannibals
4. Real-world applications

## ?? CHOOSE TASK TO SOLVE ??

## Data structures 

Enum - Enumerations are a set of restricted immutable values that can be assigned to a variable.

* class **enum.Enum**
Base class for creating enumerated constants. See section Functional API for an alternate construction syntax.

* class **enum.IntEnum**
Base class for creating enumerated constants that are also subclasses of int.

* **enum.unique()**
Enum class decorator that ensures only one name is bound to any one value.

Basic idea: create custom class with access methods
* Stack
* Queue
* PriorityQueue (heappop, heappush)

## Algorithms
###Search Algorithms:  
 - linear search. Worse case - go through all sequence, O(n)
 - binary search. If sequence is sorted, worse case - O(n log(0n))
 
 If we are only going to run our search once, and our original data structure is unsorted, it probably makes
sense to just do a linear search.   
But if the search is going to be performed many times, the time cost of doing the sort is worth it, to reap the benefit of the greatly reduced
time cost of each individual search.


### Search algorithms in graph
- depth-first search. Store Nodes to be explored in Stack. 
  
- breadth-first search. Store Nodes to be explored in Queue.

- A* algorithm. 

## Python instruments

Generics

1. Create type using *TypeVar* from *typing* . You may need to override operators (>, <, >=, <=, ==) with *bound* parameter of *TypeVar*: for that create class inherited from *typing.Protocol*

2. if function is called many  times with the repeated number of parametres, use function witch returns other function (see *lib.manhattan_distance()*)