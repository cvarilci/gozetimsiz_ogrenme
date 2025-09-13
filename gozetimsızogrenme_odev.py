import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("29-country_data.csv")

print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

import math

def plot_all_histograms(df, title_prefix=""):
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"{title_prefix}{col}")
        plt.xlabel("")
        plt.ylabel("")

    plt.tight_layout()
    plt.show()

plot_all_histograms(df)

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

df2 = df.drop("country", axis=1)

print(df2.head())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df2 = scaler.fit_transform(df2)

df.columns

df2 = pd.DataFrame(df2, columns=['child_mort', 'exports', 'health', 'imports', 'income',
       'inflation', 'life_expec', 'total_fer', 'gdpp'])

plot_all_histograms(df2)

from sklearn.decomposition import PCA

pca = PCA()

pca_df2 = pd.DataFrame(pca.fit_transform(df2))

print(pca_df2.head())

pca.explained_variance_ratio_

plt.step(list(range(1,10)), np.cumsum(pca.explained_variance_ratio_))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.ylabel("Variance Covered")
plt.title("Variance Covered")
plt.show()

pca_df2 = pca_df2.drop(columns = [3,4,5,6,7,8])

print(pca_df2)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pca_df2)
    wcss.append(kmeans.inertia_)

print(wcss)

plt.plot(range(1,11),wcss)
plt.xticks(range(1,11))
plt.xlabel("number of clusters")
plt.ylabel("WCSS")
plt.show()

model = KMeans(n_clusters=3)
model.fit(pca_df2)

labels = model.labels_

print(silhouette_score(pca_df2,labels))

df['Class'] = labels

print(df.head())

fig, ax = plt.subplots(nrows = 1, ncols= 2 , figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data=df, x="Class", y="child_mort")
plt.title("child_mort vs class")

plt.subplot(1,2,2)
sns.boxplot(data=df, x="Class", y="income")
plt.title("income vs class")

plt.show()

# 0 -> no budget needed
# 1 -> budget needed
# 2 -> in between

import plotly.express as px

pca_df2.insert(0, column= "Country", value = df['country'])

pca_df2['Class'] = labels

pca_df2.loc[pca_df2['Class'] == 1, 'Class'] = "Budget Needed"
pca_df2.loc[pca_df2['Class'] == 2, 'Class'] = "In Between"
pca_df2.loc[pca_df2['Class'] == 0, 'Class'] = "No Budget Needed"

print(pca_df2)

fig = px.choropleth(
    pca_df2[['Country', 'Class']],
    locationmode = "country names",
    locations = "Country",
    title = "Needed Budget by Country",
    color = pca_df2['Class'],
    color_discrete_map= {
                        "Budget Needed" : "Red",
                        "In Between" : "Yellow",
                        "No Budget Needed": "Green"
    })
fig.update_geos(fitbounds = "locations", visible = True)
fig.show()


