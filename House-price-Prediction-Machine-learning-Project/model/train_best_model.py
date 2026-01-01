import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor

# Load and preprocess data (same as notebook)
df = pd.read_csv("../House_Price_Data.csv")
df1 = df.drop(["area_type", "society", "availability", "balcony"], axis=1)
df3 = df1.dropna()
df3["BHK"] = df3["size"].apply(lambda x: int(x.split(' ')[0]))

def convert_sqf_to_number(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

df4 = df3.copy()
df4["total_sqft"] = df4["total_sqft"].apply(convert_sqf_to_number)
df5 = df4.copy()
df5["price_per_sqft"] = df5["price"] * 100000 / df5["total_sqft"]
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_less_than_10 = location_stats[location_stats <= 10]
df5.location = df5.location.apply(lambda x: "other" if x in location_less_than_10 else x)
df6 = df5[~(df5.total_sqft / df5.BHK < 300)]

def remove_outlier(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        sd = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - sd)) & (subdf.price_per_sqft <= (m + sd))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df7 = remove_outlier(df6)

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby("location"):
        bhk_stats = {}
        for BHK, bhk_df in location_df.groupby("BHK"):
            bhk_stats[BHK] = {
                "mean": np.mean(bhk_df.price_per_sqft),
                "std": np.std(bhk_df.price_per_sqft),
                "count": bhk_df.shape[0]
            }
        for BHK, bhk_df in location_df.groupby("BHK"):
            stats = bhk_stats.get(BHK - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < stats["mean"]].index.values)
    return df.drop(exclude_indices, axis="index")

df8 = remove_bhk_outliers(df7)
df9 = df8[df8.bath < df8.BHK + 2]
df10 = df9.drop(["size", "price_per_sqft"], axis="columns")
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10, dummies.drop("other", axis=1)], axis=1)
df12 = df11.drop('location', axis=1)
X = df12.drop("price", axis=1)
Y = df12.price

# Find best model
def find_best_model_using_GridSearchCV(X, Y):
    algo = {
        "Linear_Regression": {
            'model': LinearRegression(),
            'params': {}
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        "decisiontree": {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ["squared_error", "friedman_mse"],
                'splitter': ['random', 'best']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)
    for algo_name, config in algo.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=True)
        gs.fit(X, Y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=["model", 'best_score', 'best_params'])

best_model_results = find_best_model_using_GridSearchCV(X, Y)
best_model_name = best_model_results.loc[best_model_results['best_score'].idxmax()]['model']
best_params = best_model_results.loc[best_model_results['best_score'].idxmax()]['best_params']

if best_model_name == 'Linear_Regression':
    model = LinearRegression()
elif best_model_name == 'lasso':
    model = Lasso(**best_params)
elif best_model_name == 'decisiontree':
    model = DecisionTreeRegressor(**best_params)

model.fit(X, Y)

# Save model
with open("house_price_prediction.pickle", "wb") as f:
    pickle.dump(model, f)

# Save columns
columns = {
    "data_columns": [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))

print("Best model trained and saved:", best_model_name, "with score:", best_model_results['best_score'].max())
