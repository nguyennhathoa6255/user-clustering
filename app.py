import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import umap.umap_ as umap
from sentence_transformers import SentenceTransformer

from scipy.spatial.distance import euclidean
import clean as cl
import chart as ch
import os
import google.generativeai as genai


st.set_page_config(
    layout="wide",  
    page_title="Topic Modeling",      
    page_icon="🧊"  
)


# Caching data for faster reloads
@st.cache_data
def load_product_data():
    return pd.read_csv('./products/data_product.csv', encoding='ISO-8859-1')

@st.cache_data
def load_posttext_data():
    return pd.read_csv("./products/df_PostText_groupby.csv", encoding='ISO-8859-1')

@st.cache_data
def load_embedding_data():
    return pd.read_csv("./products/e5-base_embbeding.csv", encoding='ISO-8859-1')

df_product = load_product_data()
df_PostText = load_posttext_data()
embedding_df = load_embedding_data()

list_post_per_user = list(df_PostText["post_text"])


# --- Clustering
with st.sidebar:
    num_cluters = st.selectbox("Select number of clusters", (5, 6, 7, 8))



@st.cache_resource
def kmeans_clustering(embedding_df, num_clusters, return_value="both"):
    clustering_df = embedding_df.copy()
    kmeans = KMeans(n_clusters=num_clusters, init="k-means++", max_iter=300, n_init=10, random_state=42)
    clustering_df["cluster"] = kmeans.fit_predict(clustering_df.values)
    
    # Calculate centroids
    centroids = kmeans.cluster_centers_

    # Return based on user's request
    if return_value == "embedding":
        return clustering_df
    elif return_value == "centroids":
        return centroids
    else:
        return clustering_df, centroids

@st.cache_data
def compute_umap(clustering_df):
    Umap = umap.UMAP(n_components=3, random_state=42)
    umap_result = Umap.fit_transform(clustering_df.iloc[:, :-1])
    df_umap = pd.DataFrame(data=umap_result, columns=['first_dim', 'second_dim', 'third_dim'])
    return df_umap


clustering_df = kmeans_clustering(embedding_df, num_cluters, return_value="embedding")
df_umap = compute_umap(clustering_df)
df_umap['cluster'] = clustering_df['cluster']
#---


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def get_response(name_cluster, GOOGLE_API_KEY):
    # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    input_prompt="""You are an experience-based user product recommendation system with a deep understanding of the field of product advertising on social media platforms. Your task is to recommend products and product links based on the names of a clustered group of people.
    You have to watch the product name very carefully and you should provide the best support to give recommendations and products to the users. 
    I want to recommend specific products (e.g. shirts, pants, movie tickets, balls, etc.) as well as product links for users in the following topic groups: {output}
    """
    response = model.generate_content(input_prompt.format(output=name_cluster))
    response_dict = response.to_dict()
    # Now you can access it like a dictionary
    text = response_dict["candidates"][0]["content"]["parts"][0]["text"]
    return text
                
# --- Input type selection
with st.sidebar:
    input_type = st.radio("Input type", ["Text", "Image"])

# Handle text input
if input_type == "Text":
    with st.form('form_text_input'):
        text_input = st.text_area("Type a post text", value="a close up of a collage of bollywood movies with a large number of them")
        text_input = cl.post_cleaning(text_input)
        text_input = cl.post_cleaning_Nomeaningword(text_input)
        submitted = st.form_submit_button("Submit")
        if submitted:
            text_input = text_input

# Handle image upload
else:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        text_input = cl.process_image(uploaded_file, cl.query)
    else:
        test = cl.query("./Image/bolly.jpg")
        text_input = test[0]['generated_text']
    with st.expander("Example image"):
        if uploaded_file is not None:
            st.image(uploaded_file, use_column_width=True)
        else:
            st.image("./Image/bolly.jpg", use_column_width=True)
        st.write(f'Image to text: {text_input}')
        text_input = cl.post_cleaning(text_input)
        text_input = cl.post_cleaning_Nomeaningword(text_input)

# Embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('intfloat/e5-base-v2')

model = load_embedding_model()

embeddings = model.encode(text_input)
embedding_test = pd.DataFrame(embeddings).T

centroids = kmeans_clustering(embedding_df, num_cluters, return_value="centroids")
min_euclidean_dist = float('inf')  # Initialize with a very high value
min_cluster = -1

for i in range(len(centroids)):  # Ensure you're iterating through the DataFrame rows
    euclidean_dist = euclidean(embedding_test.iloc[0].values, centroids[i])
    
    if euclidean_dist < min_euclidean_dist:  # Looking for the smallest distance
        min_euclidean_dist = euclidean_dist
        min_cluster = i

print(f"Min Euclidean distance is for cluster {min_cluster +1}: {min_euclidean_dist:.4f}")


# Get closest cluster
name_cluster = ch.custom_labels(num_cluters)[min_cluster]
cluster_id = min_cluster + 1


# Display cluster info
with st.expander("See cluster number"):
    fig = ch.create_3d_scatter_plot(df_umap=df_umap, custom_labels=ch.custom_labels(num_cluters))
    st.plotly_chart(fig)
    st.success(f"User's posts belong to group **{name_cluster}**", icon="✅")


st.write(f"User's posts belong to group **{name_cluster}**")
 
# Display recommended products
@st.cache_data
def get_products_by_cluster(cluster_number, cluster_id, df):
    cluster_data = df[(df['cluster_number'] == cluster_number) & (df['cluster_id'] == cluster_id)]
    return cluster_data[['product_group', 'products', 'link']]

products = get_products_by_cluster(num_cluters, cluster_id, df_product)
# st.write('Recommended products by system:')
# st.write(products)


with st.expander("Recommended products by system:"):
    st.write(products)
with st.expander("Recommended products by AI:"):
    st.write(get_response(name_cluster, GOOGLE_API_KEY))