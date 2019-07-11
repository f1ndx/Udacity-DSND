
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

For this project, I was interestested in using Airbnb data, specificially for NYC from 2018 and 2019 datasets to better understand:

1. Which NYC areas have the most listings?
2. What does availability for listings look like throughout the year?
3. How do listing prices vary by location?
4. How do listing prices change throughout the year?


## File Descriptions <a name="files"></a>

The following are the files found in this repo with their corresponding descriptions:
* Airbnb_NYC_EDA.ipynb - a notebook containing the analysis done to answer the above questions; it follows the CRISP-DM process.
* calendar_june2018.csv, calendar_june2019.csv - csv's containing listing_id, date, availability, and price for each listing.
* calendar_oct2018.csv - same as above, but only used to validate one hypothesis.
* listings_june2018.csv, listings_june2018.csv - csv's containing listings data, used to pull the neighborhood features per listing_id.

The full set of Airbnb data files are publicly available and can be found [here](http://insideairbnb.com/get-the-data.html).


## Results<a name="results"></a>

The main findings of the code can be found in this Medium [post](https://medium.com/@atanasoffa/exploring-airbnb-in-nyc-68a9ce0e0101).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thank you to *Inside Airbnb* for the data used in this analysis.  Again you can find the Licensing for the data and other descriptive information at the link available [here](http://insideairbnb.com/get-the-data.html).  Otherwise, feel free to use the code here as you would like. 

