# practical5_sales_clustering.py
# Requirements: pandas, numpy, sklearn, matplotlib, seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("sales_data_sample.csv", encoding='unicode_escape')
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')

# Build RFM
snapshot = df['ORDERDATE'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CUSTOMERNAME').agg(
    Recency = ('ORDERDATE', lambda x: (snapshot - x.max()).days),
    Frequency = ('ORDERNUMBER', 'nunique'),
    Monetary = ('SALES', 'sum')
).dropna()

# Log + scale
rfm_log = np.log1p(rfm)
scaler = StandardScaler()
X = scaler.fit_transform(rfm_log)

# Elbow method
sse = []
K_range = range(1,9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    sse.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(list(K_range), sse, marker='o')
plt.xlabel('k'); plt.ylabel('SSE'); plt.title('Elbow Method')
plt.grid(True); plt.show()
print("Choose K at the elbow (common choices: 3,4,5)")

# Choose K (set after inspecting elbow)
k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = km.fit_predict(X)
rfm['Cluster'] = labels

# Show cluster counts and cluster means
print("\nCluster counts:\n", rfm['Cluster'].value_counts())
print("\nCluster means:\n", rfm.groupby('Cluster').mean().round(2))

# Visualize clusters in 2D using PCA
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
plt.figure(figsize=(7,5))
sns.scatterplot(x=coords[:,0], y=coords[:,1], hue=labels, palette='tab10', s=50)
plt.title(f"KMeans clusters (k={k}) in PCA space")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(title='Cluster')
plt.show()
