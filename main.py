# ────────────────────────────────────────────────
# California Housing Prices - Clean Initial EDA
# ────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
# Optional: nicer plots
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Libraries loaded!")
print(f"NumPy      {np.__version__}")
print(f"Pandas     {pd.__version__}")
print(f"Matplotlib {plt.matplotlib.__version__}")
print(f"scikit-learn {pd.__version__}\n")  # small typo fix if you want sklearn version

# ─── 1. Load the data ───────────────────────────────────────
# Make sure housing.csv is in the same folder
df = pd.read_csv("housing.csv")

# Most people rename it to 'housing' for tradition
housing = df.copy()           # ← important: now we use 'housing'


print("Dataset shape:", housing.shape)
print("\nFirst 5 rows:")
print(housing.head())

# ─── 2. Quick look at the data ──────────────────────────────
print("\nInfo:")
housing.info()

print("\nMissing values:")
print(housing.isna().sum())

print("\nBasic statistics:")
print(housing.describe())

# ─── 3. Plot ALL histograms ─────────────────────────────────
housing.hist(bins=50, figsize=(15, 13), color='skyblue', edgecolor='black')
plt.tight_layout()
plt.show()

housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)

plt.title("Lat and Longit")
plt.show()

## Price range heatmap

housing.plot(kind = "scatter", x = "longitude", y= "latitude", alpha = 0.4, s= housing["population"]/100, label = "population", figsize = (10,7), c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True,)
plt.legend()
plt.show()
# ─── 4. Create income category for stratified splitting ─────
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
    include_lowest=True          # important in newer pandas
)

print("\nIncome category distribution:")
print(housing["income_cat"].value_counts(normalize=True).sort_index() * 100)

housing["income_cat"].hist(color='coral', edgecolor='black')
plt.title("Income Category Distribution")
plt.show()

# ─── 5. Stratified train/test split (recommended) ───────────
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].copy()
    strat_test_set  = housing.loc[test_index].copy()



# Quick check — proportions should be very similar
print("\nTrain / Test income category proportions:")
print("Train:\n", strat_train_set["income_cat"].value_counts(normalize=True).sort_index())
print("Test:\n",  strat_test_set["income_cat"].value_counts(normalize=True).sort_index())

# Optional: drop helper column if you don't need it anymore
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

print("\n→ Stratified train set shape:", strat_train_set.shape)
print("→ Stratified test set shape :", strat_test_set.shape)

# Looking for correlations
# Right after loading / before or after the split
numeric_housing = housing.drop("ocean_proximity", axis=1)

corr_matrix = numeric_housing.corr()


from pandas.plotting import scatter_matrix

attributes = ["median_house_value","median_income","total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12,8))
plt.show()

housing.plot(kind = "scatter", x= "median_income", y = "median_house_value", alpha = 0.1)
plt.show()



#-------------------Looking for correlations--------------------


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

#-----------------correlation matrix-------------------
corr_matrix = housing.corr(numeric_only= True)
corr_matrix["median_house_value"].sort_values(ascending = False)
print(corr_matrix)

housing.dropna(subset = ["total_bedrooms"])

imputer = SimpleImputer(strategy = "median")
#----copy of the data without the text attribute-----
housing_num = housing.drop("ocean_proximity", axis = 1)


housing_cat = housing["ocean_proximity"]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing[["ocean_proximity"]])   # ← double brackets!
print(housing_cat_encoded.shape[:10])

# You can now work with strat_train_set for EDA and model building
# Save them if you want:
# strat_train_set.to_csv("housing_train_stratified.csv", index=False)
# strat_test_set.to_csv("housing_test_stratified.csv", index=False)