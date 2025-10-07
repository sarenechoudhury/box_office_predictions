# Box Office Predictions using an AugBoost Hybrid LightGBM–Neural Regression Model


### 1. This model trains an LGBM regressor to predict log-based revenue

### 2. Computes residuals by using a neural network to model the residuals and augments the dataset, adding the ANN's predictions as new features

### 3. Iteratively repeats this process to enrich the data and then trains the final LGBM model on that feature set 


*Data from Kaggle*

*https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download*

*https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews?resource=download*


