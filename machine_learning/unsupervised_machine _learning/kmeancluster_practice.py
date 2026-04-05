'''

K-Means :  is an unsupervised machine learning algorithm used to group similar data points into clusters.
The algorithm divides the dataset into K number of clusters based on similarity.
Example:
Grouping customers based on age and spending habits.

K → Number of clusters
Means → The cluster center is calculated using the mean (average) of the points in that cluster.

Goal of K-Means : 
The goal is to minimize the distance between data points and their cluster centroid.
Each data point should belong to the nearest cluster center.

K-Means is commonly used for:
Customer segmentation
Image compression
Market segmentation
Document clustering
Anomaly detection


Limitations of K-Means :
Need to choose K beforehand
Sensitive to outliers
Works best for spherical clusters
Different initial centroids can produce different results

A shopping mall owner wants to divide customers into groups based on their annual income and spending score so that marketing strategies can be targeted.
Dataset:

Customer	Income (k$)	Spending Score
    C1  	    15	    39
    C2	        16	    81
    C3	        17	    6
    C4	        18	    77
    C5	        19	    40
    C6	        20	    76

We want to divide customers into K = 2 clusters.


Step 2: Randomly Choose Centroids
Suppose the algorithm randomly selects:
Centroid 1 → (15,39)
Centroid 2 → (18,77)
These represent temporary cluster centers.

Step 3: Assign Points to Nearest Centroid
Now calculate distance from each customer to both centroids.
Example: Customer C5
C5 = (19,40)
Distance to Centroid1:
√((19-15)² + (40-39)²)
= √(16 + 1)
= √17 ≈ 4.12

Distance to Centroid2:
√((19-18)² + (40-77)²)
= √(1 + 1369)
= √1370 ≈ 37

Since 4.12 < 37

C5 joins Cluster 1.
The algorithm repeats this for all customers.

Result after assignment:
Cluster 1
C1, C3, C5

Cluster 2
C2, C4, C6

Step 4: Recalculate Centroids
Now compute mean of each cluster.
Cluster 1 points:
(15,39)
(17,6)
(19,40)

New centroid:
Income = (15+17+19)/3 = 17
Spending = (39+6+40)/3 ≈ 28.3

New centroid1:
(17 , 28.3)

Cluster 2 points:
(16,81)
(18,77)
(20,76)

New centroid:
Income = (16+18+20)/3 = 18
Spending = (81+77+76)/3 ≈ 78

New centroid2:
(18 , 78)
Step 5: Repeat

Again calculate distances using new centroids.
If points stay in the same clusters, the algorithm stops.
This means clusters are stable. 
Final Clusters

Cluster 1
Low spenders

Cluster 2
High spenders

Mall owner can now:

give discount offers to Cluster 1
give premium promotions to Cluster 2
'''


'''
In the below example we are going to do colour compression in images .


'''


from sklearn.cluster import k_means
from sklearn.datasets import load_sample_image
from matplotlib import pyplot as plt
import numpy as np

flower = load_sample_image("flower.jpg")

print(flower.shape)

data = flower / 255.0  #use 0....1 scale
data = data.reshape(427*640,3)
print(data.shape)


def plot_pixels(data, title , colors=None, N=10000):

    if colors is None:
        colors = data

    rng  = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G,color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0,1), ylim=(0,1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red',ylabel='Blue',xlim=(0,1),ylim=(0,1))
    fig.suptitle(title, size=20)


plot_pixels(data, title='Input color space: 16 million possible colors')
plt.show()

import warnings; warnings.simplefilter('ignore')
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
plot_pixels(data,colors=new_colors,title = "Reduced color space: 16 colors")

plt.show()
china_recolored = new_colors.reshape(flower.shape)

