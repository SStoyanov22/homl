

# Python ≥3.5 is required

import sys

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tarfile
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"


'''1.Settings'''
# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


'''2.Get Data'''
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

#Create directory, downloads .tgz file, extracts .csv file in the directory
def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#Load .csv data into dataframe
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data(HOUSING_URL,HOUSING_PATH)
housing_df = load_housing_data(HOUSING_PATH)
print(housing_df.head())

housing_df.info()
#total bedrooms has null values and is an object

#after looking at the dataset we notice that there are a lot of repeating values for this column
#so we check the value counts  as it is most probably categorical
housing_df["ocean_proximity"].value_counts()
housing_df.hist(bins=50, figsize=(20,15))
plt.show()

#Create test set
train_set, test_set = train_test_split(housing_df, test_size=0.2, random_state=42)

#Create income category attribute and display in histogram
housing_df["income_cat"] = pd.cut(housing_df["median_income"],
                               bins = [0, 1.5, 3, 4.5, 6, np.inf],
                               labels = [1, 2, 3, 4, 5])
housing_df["income_cat"].hist()
plt.show()

#Apply stratified sampling to avoid sampling bias
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_df, housing_df["income_cat"]):
    strat_train_set = housing_df.loc[train_index]
    strat_test_set = housing_df.loc[test_index]


strat_test_set["income_cat"].value_counts() / len(strat_test_set)

housing_df["income_cat"].value_counts() / len(housing_df)

#compare Stratified vs Random
def income_cat_proportions(data):
    return data["income_cat"].value_counts()/len(data)

train_set, test_set = train_test_split(housing_df, test_size=0.2, random_state=42)
compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing_df),
    "Stratified":income_cat_proportions(strat_test_set),
    "Random":income_cat_proportions(test_set),
}).sort_index()

compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
print(compare_props)
#drop the categorical column as it is no longer needed
for strat_set in (strat_train_set,strat_test_set):
    strat_set.drop("income_cat", axis=1 , inplace=True)


'''3.Discover and Visualize the Data to Gain Insights'''
#Visualizing Geographical Data
housing = strat_train_set.copy()

housing.plot(kind="scatter",x="longitude",y="latitude")
save_fig("bad_visualization_plot")#the plot looks like California


housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
save_fig("better_visualization_plot")#we can see that there is density around LA and San Diego


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")

#Looking for Correlations
corr_matrix=housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
print("end")