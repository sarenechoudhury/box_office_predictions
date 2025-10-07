import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap

def safe_qcut(series, q=3, labels=None):
    """Safely perform qcut with duplicate handling."""
    try:
        result = pd.qcut(series, q=q, labels=labels, duplicates='drop')
    except ValueError:
        result = pd.cut(series, bins=q, labels=labels)
    return result

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("cleaned_movies_metadata.csv")

rt = pd.read_csv("rotten_tomatoes_movies.csv")
df = df.merge(rt[['title', 'rating']], how='left', on='title')
df['rating'] = df['rating'].astype('category').cat.codes

credits = pd.read_csv("credits.csv")
df = df.merge(credits[['id', 'cast', 'crew']], how='left', on='id')

# Missing Values Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.isnull(), cbar = False)
plt.title("Missing Values")
plt.tight_layout()
plt.savefig("figures/missing_values.png")
plt.close()

# Parse cast and crew features
def parse_cast(cast_str):
    try:
        cast_list = ast.literal_eval(cast_str)
        return len([c for c in cast_list if c.get('order', 999) < 5])
    except:
        return 0

def parse_crew(crew_str, department):
    try:
        crew_list = ast.literal_eval(crew_str)
        return sum(1 for c in crew_list if c.get('department') == department)
    except:
        return 0

df['num_top_cast'] = df['cast'].apply(parse_cast)
df['num_directors'] = df['crew'].apply(lambda x: parse_crew(x, 'Directing'))
df['num_writers'] = df['crew'].apply(lambda x: parse_crew(x, 'Writing'))
df['num_producers'] = df['crew'].apply(lambda x: parse_crew(x, 'Production'))

plt.figure(figsize=(10,8))
sns.heatmap(df[['budget','popularity','runtime','rating',
                'num_top_cast','num_directors','num_writers',
                'num_producers','revenue']].corr(),
            annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("figures/correlation_matrix.png")
plt.close()

# Compare feature distributions by revenue group
df['revenue_group'] = safe_qcut(df['revenue'], q=3, labels=['Low','Medium','High'])
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='revenue_group', y='budget')
plt.title("Budget Distribution Across Revenue Groups")
plt.tight_layout()
plt.savefig("figures/budget_by_revenue_group.png")
plt.close()

# Visualization 1: Revenue Distribution
plt.figure(figsize=(8, 5))
sns.histplot(np.log1p(df['revenue'].dropna()), bins='auto', kde=True)
plt.title("Distribution of Log-Transformed Revenue (0-15 Range)")
plt.xlabel("Log(1 + Revenue)")
plt.ylabel("Frequency")
plt.yscale('log')
plt.xlim(0, 15)
plt.tight_layout()
plt.savefig("figures/revenue_distribution.png")
plt.close()

# Visualization 2: Budget vs Revenue
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='budget', y='revenue', alpha=0.5)
plt.title("Budget vs Revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("figures/budget_vs_revenue.png")
plt.close()

# Feature selection
features = ['budget', 'popularity', 'runtime', 'rating', 
            'num_top_cast', 'num_directors', 'num_writers', 'num_producers']
df['main_genre_encoded'] = df['main_genre'].astype('category').cat.codes
features.append('main_genre_encoded')
target = 'log_revenue'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize augmented features
X_train_aug = X_train.copy()
X_test_aug = X_test.copy()
y_resid = y_train.copy()

# AugBoost iterations
N_ITER = 3
for iteration in range(N_ITER):
    print(f"\n=== AugBoost Iteration {iteration + 1} ===")

    lgb_train = lgb.Dataset(X_train_aug, label=y_resid)
    lgb_model = lgb.train({'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}, 
                          lgb_train, num_boost_round=100)

    pred_train = lgb_model.predict(X_train_aug)
    y_resid = y_train - pred_train

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test_aug)

    ann_model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)
    ])
    ann_model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
    ann_model.fit(X_train_scaled, y_resid, 
                  validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)
    
    try:
    

        # Use small subset to keep computation light
        X_background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
        X_explain = X_test_scaled[:100]

        deep_explainer = shap.DeepExplainer(ann_model, X_background)
        shap_values_deep = deep_explainer.shap_values(X_explain)

        # Plot SHAP summary for this iteration
        shap.summary_plot(shap_values_deep, X_explain, show=False)
        plt.title(f"Deep SHAP Summary (ANN Iteration {iteration + 1})")
        plt.savefig(f"figures/deep_shap_iter{iteration + 1}.png", bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"⚠️ Skipped Deep SHAP at iteration {iteration + 1}: {e}")


    ann_train_features = ann_model.predict(X_train_scaled)
    ann_test_features = ann_model.predict(X_test_scaled)

    X_train_aug[f'ann_out_{iteration}'] = ann_train_features.flatten()
    X_test_aug[f'ann_out_{iteration}'] = ann_test_features.flatten()

# Final LGBM training
final_lgb_train = lgb.Dataset(X_train_aug, label=y_train)
final_model = lgb.train({'objective': 'regression', 'metric': 'rmse', 'verbosity': -1},
                        final_lgb_train, num_boost_round=100)

# Final predictions
final_preds = final_model.predict(X_test_aug)
rmse = np.sqrt(mean_squared_error(y_test, final_preds))
print(f"\n✅ Final AugBoost RMSE: {rmse:.4f}")

importances = final_model.feature_importance()
feature_names = final_model.feature_name()

imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
imp_df = imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=imp_df.head(15), x='Importance', y='Feature')
plt.title("Top 15 Feature Importances (LightGBM)")
plt.tight_layout()
plt.savefig("figures/feature_importance.png")
plt.close()

explainer = shap.TreeExplainer(final_model)

# Use a manageable sample for efficiency
X_sample = X_test_aug.sample(n=min(500, len(X_test_aug)), random_state=42)

shap_values = explainer.shap_values(X_sample)

# Summary plot of feature importance
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("Tree SHAP Summary: Final LightGBM Model")
plt.savefig("figures/shap_summary_lightgbm.png", bbox_inches='tight')
plt.close()

# Optional: Feature importance bar plot (SHAP-based)
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title("Tree SHAP Feature Importance (Bar)")
plt.savefig("figures/shap_bar_lightgbm.png", bbox_inches='tight')
plt.close()

# Save final performance comparison plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, final_preds, alpha=0.5)
plt.xlabel("Actual Log Revenue")
plt.ylabel("Predicted Log Revenue")
plt.title("Final Model: Actual vs Predicted Log Revenue")
plt.tight_layout()
plt.savefig("figures/final_model_performance.png")
plt.close()

# === Model Evaluation and Diagnostics ===
residuals = y_test - final_preds

# Residual Plot
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Actual Log Revenue")
plt.xlabel("Actual Log Revenue")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("figures/residual_plot.png")
plt.close()

# Residual Distribution
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Distribution of Prediction Errors (Residuals)")
plt.xlabel("Residual (y_true - y_pred)")
plt.tight_layout()
plt.savefig("figures/residual_distribution.png")
plt.close()

# Actual vs Predicted with Trend Line
plt.figure(figsize=(8,5))
sns.regplot(x=y_test, y=final_preds, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title("Actual vs Predicted Log Revenue with Trend Line")
plt.xlabel("Actual Log Revenue")
plt.ylabel("Predicted Log Revenue")
plt.tight_layout()
plt.savefig("figures/actual_vs_predicted_trend.png")
plt.close()

# === Actual vs Predicted Revenue in Dollars ===
plt.figure(figsize=(8,5))
sns.scatterplot(x=np.expm1(y_test), y=np.expm1(final_preds), alpha=0.5)
plt.title("Actual vs Predicted Revenue ($)")
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig("figures/actual_vs_predicted_revenue_dollars.png")
plt.close()





