### Python thinking

- better use ''.join() instead of a_str + b_str  
- better not to get names to *lambda* functions -> just ordinary function  
- Prefer Using ““.startswith() and ””.endswith() -> for small strings  
- Use the isinstance() Method Instead of type() for Comparison  
- Finised logic within fuction -> do onle what name of the function says  
- Using Docstrings -> code documentation. Add dot (.) in docstring  
   Sphinx:  http://www.sphinx-doc.org/en/stable/, Pycco:  https://pycco-docs.github.io/pycco/ , Read the docs:  https://readthedocs.org/, Epydocs:  http://epydoc.sourceforge.net/,  
- list comprechension is more readable than *filter / map*  
- *loop ... else* close/ Better not to use  
- Exceptions: "Python’s credo is “It’s easier to ask forgiveness than permission.” This means that you don’t check beforehand to make sure you won’t get an exception; instead, if you get exception, you handle it"  
- Exceptions: *Finally* close better to use   
- "Prefer to Have Minimum Code Under try" **TO FIX**  
--------------------------------------------

### Data structures

- Use Sets for Speed
- Use namedtuple for Returning and Accessing Data **TO FIX**  
- Return the Data -- named tuple instead of tuple  
- "Finally, if you want to convert namedtuple to a dict or convert a list to namedtuple, namedtuple gives you methods to do it easily. So, they are flexible as well. "  
- "Use Lists Carefully and Prefer Generators" **TRY TO FIX**  
- Counter -> **TO FIX !!!!!** pg 50  
- collections - deque, defaultdict, oredereddict, namedtuple

--------------------------------------------

### Writing better Functions and Classes  

- "Raise Exceptions Instead of Returning None"  
- "Do Not Return None Explicitly" -> "You might want to raise an exception in these cases." ???  
- There are two things that you as a programmer can do before shipping code off to production to make sure that you are shipping quality code.
    • Logging  
    • Unit test (**unittest**)
- Class Variables. Usually you want to see a class variable at the top because these variables either are constants or are default instance variables.  
- Right Ways to Use @property  -- getting / setting  
- @classmethod, @staticmethod, @property,  instance methods  

--------------------------------------------

### Working with modules and metaclasses

- Use __all__ to Prevent Imports  
- Metaclasses 

--------------------------------------------

### Decorators and context managers

- decorators: Here are some examples:  
• Rate limiting  
• Caching values  
• Timing the runtime of a function  
• Logging purposes  **FOR LOGGING - FIX**  
• Caching exceptions or raising them  
• Authentication
- *functools* -> can debug wrappers
- Classes as decorators
- decorators for classes  
  https://pythonworld.ru/osnovy/dekoratory.html
- Context manager (with open(file.txt) ... )  
https://devpractice.ru/python-lesson-21-context-manager/  
  
--------------------------------------------

### Generators and iterators

https://habr.com/ru/post/488112/  
- Generators -> *yield* operator  (заменить писки на генераторы для оптимизации памяти **TRY TO FIX**)  
- Difference between Generators and Iterators https://cutt.ly/fjCgoCZ  
- Generators, *yield*, *yield from* 
- itertools  

-------------------------------------------- 

### Python features 

- asyncio, async , await, async generators, async comprehensions, async iterators,   
- types. 
- Better Path Handling Using   
- create_report(user, *, file_type, location): -> Now when you call create_report, you have to provide a keyword argument after *. 

-------------------------------------------- 

### Debugging and Testing Python Code

- logging configuration <-- **important**  
- UnitText
