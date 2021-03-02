
# Recommendations with IBM

## Table of Contents

1. [Description](#description)
2. [Tasks](#tasks)
3. [Getting Started](#getting_started)
	1. [Installation](#installation)
	2. [Instructions](#instructions)
4. [Licensing, Authors, and Acknowledgements](#licensing)


## Description<a name="descripton"></a>
For this project, I analyzed the interactions that users have with articles on the IBM Watson Studio platform, and made recommendations about new articles that they may potentially be interested in.


## Tasks<a name="tasks"></a>
The project is divided into the following tasks:

**I. Exploratory Data Analysis**
Before making recommendations of any kind, I performed EDA on the data.

**II. Rank Based Recommendations**
To get started in building recommendations, I first found the most popular articles simply based on the most interactions. Since there are no ratings for any of the articles, it is easy to assume the articles with the most interactions are the most popular. These are then the articles we might recommend to new users (or anyone depending on what we know about them).

**III. User-User Based Collaborative Filtering**
In order to build better recommendations for the users of IBM's platform, I looked at users that are similar in terms of the items they have interacted with. These items could then be recommended to the similar users (which moves towards more personal recommendations for the users). 

**IV. Matrix Factorization**
Finally, I completed a machine learning approach to building recommendations. Using the user-item interactions, I built out a matrix decomposition to get an idea of how well I could predict new articles an individual might interact with and discussed which methods I might use moving forward, and how I could test how well my recommendations are working for engaging users.


## Getting Started<a name="getting_started"></a>

### Installation<a name="installation"></a>

* Python 3.*
* Libraries: NumPy, Pandas
* Data Visualization: Matplotlib, Plotly


### Instructions:<a name="instructions"></a>

Run one of the following commands:

`ipython notebook Recommendations_with_IBM.ipynb` or `jupyter notebook Recommendations_with_IBM.ipynb`

This will open the iPython Notebook software and project file in your browser.


## Licensing, Authors, and Acknowledgements<a name="licensing"></a>

* Author: [Anastasia Atanasoff](https://github.com/atanasoffa)
* Acknowledgements: [IBM](https://www.ibm.com/cloud/watson-studio) for providing the dataset used for this project.
