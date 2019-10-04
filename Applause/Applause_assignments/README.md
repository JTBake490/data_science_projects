This folder contains:
 * the coding exercise named Applause.py
 * The thought excercise call Applause_thought_ex

To run the application simply have python installed and run 
    
    python Applause.py 

in a command line.

While the app is running.

1. Type in the number of Countries you would like to search (as an integer) or type all.
2. If all was not entered for the first step, enter a two letter country code
3. Type in the number of Devices you would like to search (as an integer) or type all.
4. If all was not entered for the thrid step, enter a device name.

Example 1:
* all
* all

Example 2:
* all
* 2
* iphone4
* iphone5

Example 3:
* 2
* us
* jp
* all

Example 4:
* 2
* us
* jp
* 2
* iphone4
* iphone5

---------------------------------------------------------------------------------------------------------------------------------------------

Edit:
To create the Postgresql database use: 

    \i 'relative path'

Replace 'relative path' with your actual relative path.

To change the quieries simply change the number of countries/devices or change the countries/devices.

--------------------------------------------------------------------------------------------------------------------------------

Edit: To run the quieries in the pyscopg2 notebook replace the user variable to the appropriate user and execute the cells. To change the quieries simple change the countries/devices within the quieries.
