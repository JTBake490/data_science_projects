# Applause
Assignment Submission for Applause

Please fork this repository and clone it to your local machine. Once cloned, run the Applause.py file in the Applause_assignments folder via a command line face. 

The script to run the application is:
    
    python Applause.py

The thought exercise is located in the same folder.

---------------------------------------------------------------------------------------------------------------------------------------------
Edit: 

Added SQL code to create a database and four tables in that database.
Also added a file that gives four quiers that is the equivalent of the python script added. 

-----------------------------------------------------------------------------------------------------------------------------------

Added a notebook that makes use of psycopg2. This library allowed me to retrieve the data two different ways. The first method was to create a cursor (after starting PostgreSQL on the backend) and retrieve each row with a for loop. The second method was to create a querty, execute that query, and place the data into a Pandas dataframe.
