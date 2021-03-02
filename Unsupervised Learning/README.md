# Identify Customer Segments with Arvato

*Summary:* In this project, I worked with real-life data provided by Udacity's Bertelsmann partners AZ Direct and Arvato Finance Solution (a company that performs mail-order sales in Germany).  Their main question of interest was to identify facets of the population that are most likely to be purchasers of their products for a mailout campaign. 

Thus, the objective in this project was to use unsupervised learning techniques on the given demographic and spending data (for a sample of German households) to organize the general population into clusters, then use those clusters to see which of them comprise the main user base for the company. The business benefit here would be that these segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns.

Prior to applying the machine learning methods, I had to assess and clean the data in order to convert the data into a usable form.

---
The following libraries were used in this project:
* NumPy
* pandas
* Sklearn / scikit-learn
* Matplotlib (for data visualization)
* Seaborn (for data visualization)

### The Data
* Udacity_AZDIAS_Subset.csv (not included here): Demographic data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
* Udacity_CUSTOMERS_Subset.csv (not included here): Demographic data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
* AZDIAS_Feature_Summary.csv (not included here): Summary of feature attributes for demographic data.
* Identify_Customer_Segments.ipynb: Jupyter Notebook for completing the project.

### Preprocessing
Created a cleaning procedure that I appled first to the general demographic data, then later to the customers data.

### Feature Transformatiom
Once data was cleaned, I used dimensionality reduction techniques (feature scaling and PCA using sklearn) to identify relationships between variables in the dataset, resulting in the creation of a new set of variables that account for those correlations.

### Clustering
Once data was transformed, I applied clustering techniques (k-means, sklearn) to identify groups in the general demographic data. I then appled the same clustering model to the customers dataset to see how market segments differ between the general population and the mail-order sales company. 