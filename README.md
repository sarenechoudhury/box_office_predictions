# Box Office Predictions using an AugBoost Hybrid LightGBM–Neural Regression Model

<br />
<br />

### 1. This model trains an LGBM regressor to predict log-based revenue

### 2. Computes residuals by using a neural network to model the residuals and augments the dataset, adding the ANN's predictions as new features

### 3. Iteratively repeats this process to enrich the data and then trains the final LGBM model on that feature set 

<br />

#### Notes: The box_office_preds file contains an optional feature pruning function that may affect performance depending on specific datasets used. It also contains graphing functions throughout to visualize data at various points, and a few more comments. The box_office_preds_clean file does not contain those.

<br />
<br />


*Data from Kaggle:*

*https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download*

*https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews?resource=download*

<br />
<br />

*Inspired by:*

*https://knowledge.wharton.upenn.edu/wp-content/uploads/2022/03/1329.pdf*

*www.sciencedirect.com/science/article/pii/S0957417405001399*

*www.ijcai.org/proceedings/2019/0493.pdf*

*www.sciencedirect.com/science/article/abs/pii/S156625351930497X*


