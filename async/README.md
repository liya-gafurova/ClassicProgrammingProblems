# TESTING OF SYNC AND ASYNC METHODS

File nlp.py is a file presenting nlp service, which is checking text for grammar mistakes.  
It is a simple web service with URL /check_grammar  
Checking grammar takes 5 seconds (sleep for 5 seconds)  

#### ----------

File backend.py is file which sends requests to nlp service.  
If 10 requests are sent using *requests*  **(sync)** library, it will take 5 seconds * 10 requests = 25 seconds.  
If **async** tasks are created, it takes approximately 5 seconds for 10 requests (takes time for the longest resuest) 

