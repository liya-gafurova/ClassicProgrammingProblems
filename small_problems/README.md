# Small problems

1. The Fibonacci sequence
2. Trivial compression
3. Unbreakable encryption 
4. Calculating PI
5. The Towers of Hanoi
6. Real-world applications


## ?? CHOOSE ONE OF 6 TO IMPLEMENT ??
- Hanoi towers

## Some theory
**Memoization** is a technique in which you store the results of computational tasks when
they are completed so that when you need them again, you can look them up instead
of needing to compute them a second (or millionth) time. Can be used when we have recursive function, for example

Memoization can be implemented:

1. Make dictionary and store results. If required result already calculated, then use it and do not perform unnecessary function calling
2. Using function with decorator *@lru_cache(maxsize=None)*
## 
Remember, any problem that can be solved recursively can also be solved iteratively.  
Pay attention to how many function calls occur with the recursive and iterative approach. Sometimes it is more efficient to use an iterative approach
##
Generators
##
Compression - *If it is more storage-efficient to compress data, then why is all data not com-
pressed? There is a tradeoff between time and space.*  
If the number of possible different values that a type is meant to represent is less than
the number of values that the bits being used to store it can represent, it can likely be
more efficiently stored. For example, A-G-C-T in DNA can be encoded with 2 bytes.   
Can be implemented can be implemented manually using bit operations.
##
