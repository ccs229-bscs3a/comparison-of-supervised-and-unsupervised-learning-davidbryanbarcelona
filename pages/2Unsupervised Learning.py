#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, metrics
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
import time

# Define the Streamlit app
def app():

    st.subheader('K-means clustering applied to the Iris Dataset')
    text = """This is a classic example of unsupervised learning task. The mushroom dataset contains 
    information about mushrooms (Cap Diameter, Cap Shape, Cap Surface, Stem Height, Stem width) but 
    doesn't have labels indicating the mushroom's edibility. K-means doesn't use these labels during clustering.
    \n* **K-means Clustering:** The algorithm aims to group data points into a predefined 
    number of clusters (k). It iteratively assigns each data point to the nearest cluster 
    centroid (center) and recomputes the centroids based on the assigned points. This process 
    minimizes the within-cluster distances, creating groups with similar characteristics.
    In essence, K-means helps uncover inherent groupings within the mushroom data based on their 
    features (measurements) without relying on predefined categories (edibility). 
    This allows us to explore how well the data separates into natural clusters, potentially 
    corresponding to the actual flower species.
    \n* Choosing the optimal number of clusters (k) is crucial. The "elbow method" 
    helps visualize the trade-off between increasing clusters and decreasing improvement 
    in within-cluster distances.
    * K-means is sensitive to initial centroid placement. Running the algorithm multiple times 
    with different initializations can help identify more stable clusters.
    By applying K-means to the Iris dataset, we gain insights into the data's underlying structure 
    and potentially validate the separability of the known flower species based on their 
    measured characteristics."""
    st.write(text)


    if st.button("Start"):
        # Load the Iris dataset
        mushroom = pd.read_csv('mushroom.csv')
        X = mushroom[['cap-diameter', 'stem-height', 'stem-width']] # Features 
        y = mushroom['class']  # Target labels (edibility)


        # Define the K-means model with 2 clusters (edibility)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)

        # Train the K-means model
        kmeans.fit(X)

        # Get the cluster labels for the data
        y_kmeans = kmeans.labels_

        # Since there are no true labels for unsupervised clustering,
        # we cannot directly calculate accuracy.
        # We can use silhouette score to evaluate cluster separation

        # Calculate WCSS
        wcss = kmeans.inertia_
        st.write("Within-Cluster Sum of Squares:", wcss)

        silhouette_score = metrics.silhouette_score(X, y_kmeans)
        st.write("K-means Silhouette Score:", silhouette_score)

        text = """**Within-Cluster Sum of Squares (WCSS): 641.1101221616161**
        This value alone doesn't tell the whole story. A lower WCSS generally indicates tighter 
        clusters, but it depends on the scale of your data and the number of clusters used (k).
        \n**K-mmeans Silhouette Score: 0.0.8819455405051794**
        * This score provides a more interpretable measure of cluster quality. It 
        ranges from -1 to 1, where:
        * Values closer to 1 indicate well-separated clusters.
        * Values around 0 suggest clusters are indifferently assigned (data points could belong to either cluster).
        * Negative values indicate poorly separated clusters (data points in a cluster are closer to points in other clusters).
        In this case, a Silhouette Score of 0.8819 suggests:
        * **Moderately separated clusters:** The data points within a cluster are somewhat closer to their centroid than to centroids of other clusters. There's some separation, but it's not perfect
        * **Potential for improvement:** You might consider exploring different numbers of clusters (k) or using different initialization methods for K-means to see if a better clustering solution can be achieved with a higher Silhouette Score (closer to 1).
        * The mushroom dataset is relatively well-separated into edible and poisonous."""
        with st.expander("Click here for more information."):\
            st.write(text)
            
        # Get predicted cluster labels
        y_pred = kmeans.predict(X)

        # Define mapping from cluster labels to edibility categories
        label_to_edibility = {
            0: 'edible',
            1: 'poisonous'
        }

        # Get unique class labels and color map
        unique_labels = list(set(y_pred))
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))

        fig, ax = plt.subplots(figsize=(8, 6))

        for label, color in zip(unique_labels, colors):
            indices = y_pred == label
            # Use ax.scatter for consistent plotting on the created axis
            ax.scatter(X.loc[indices, 'cap-diameter'], X.loc[indices, 'stem-height'], label=label_to_edibility[label], c=color)

        # Add labels and title using ax methods
        ax.set_xlabel('Cap Diameter (cm)')
        ax.set_ylabel('Stem Height (cm)')
        ax.set_title('Cap Diameter vs Stem Height by Predicted Mushroom Edibility')

        # Add legend and grid using ax methods
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


#run the app
if __name__ == "__main__":
    app()
    