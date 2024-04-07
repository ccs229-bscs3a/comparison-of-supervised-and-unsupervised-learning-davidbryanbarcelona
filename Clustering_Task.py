#Input the relevant libraries
import streamlit as st
# Define the Streamlit app
def app():

    text = """Comparing Supervised and Unsupervised Learning: KNN vs KMeans"""
    st.subheader(text)

    text = """David Bryan S. Barcelona\n\n
    CCS 229 - Intelligent Systems
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('mushrooms.jpg', caption="The Mushroom Dataset""")

    text = """Data App: Supervised vs Unsupervised Learning Performance
    \nThis data app allows users to compare the performance of supervised learning (KNN) and unsupervised 
    learning (K-Means) gorithms for clustering tasks. 
    \nOnce configured, users can initiate the analysis. The app will run the KNN and K-Means algorithms on 
    the mushroom dataset."""
    st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
    