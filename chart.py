import streamlit as st
import plotly.graph_objects as go


def custom_labels(num_clusters):
    if num_clusters == 5:
        return [
            '1: Pop Culture Icons: Kardashian, Music & Fashion',
            '2: NBA & WWE',
            '3: Music: Pop-Rock & EDM',
            '4: Bollywood & Sports ',
            '5: Hollywood & Entertainment Icons'
        ]
    elif num_clusters == 6:
        return [
            '1: Bollywood & Fashion Trends',
            '2: Football, WWE, NBA & Tennis',
            '3: Global Pop & EDM Icons',
            '4: Pop & Hip-Hop Stars in Entertainment',
            '5: Hollywood & Political Icons in Entertainment',
            '6: Pop Culture Icons: Kardashian, Music & Fashion'
        ]
    elif num_clusters == 7:
        return [
            '1: World Sports: Football, Tennis & Racing', 
            '2: Pop, Rock & EDM Music',
            '3: Pop, Hip-Hop & R&B',
            '4: Hollywood & TV Entertainment',
            '5: The Kardashian-Jenner Lifestyle & Beyond',
            '6: NBA & WWE & Entertainment Icons',
            '7: Bollywood & Cricket'
        ]
    else:
        return [
            '1: Pop & Hip-Hop',
            '2: Celebrities & Entertainment Icons',
            '3: Football, Tennis & MMA',
            '4: Pop Music & Entertainment ',
            '5: Pop Culture Icons: Kardashian, Music & Fashion',
            '6: NBA, WWE & Celebrity TV Hosts',
            '7: Talk Show Hosts & Hollywood Actors',
            '8: Bollywood & Cricket'
        ]


def create_3d_scatter_plot(df_umap, custom_labels):
    # Centers of each cluster
    centers = df_umap.groupby('cluster')[['first_dim', 'second_dim', 'third_dim']].mean().values

    # Custom colors for each cluster
    custom_colors = ['#4535C1', '#FF8225', '#ED3EF7', '#00712D', '#F5004F', '#FFEB55', '#FF8C9E', '#836FFF']

    # Create 3D scatter plot figure
    fig = go.Figure()

    # Loop through each cluster and add scatter points
    for cluster_id, center in enumerate(centers):
        cluster_points = df_umap[df_umap['cluster'] == cluster_id]

        fig.add_trace(go.Scatter3d(
            x=cluster_points['first_dim'],
            y=cluster_points['second_dim'],
            z=cluster_points['third_dim'],
            mode='markers',
            marker=dict(
                size=8,
                color=custom_colors[cluster_id],
                opacity=0.7
            ),
            name=custom_labels[cluster_id]
        ))

        # Add cluster center text
        fig.add_trace(go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode='text',
            textfont=dict(size=15, color='black', family='Arial'),
            showlegend=False
        ))

    # Customize axis labels and title
    fig.update_layout(
        autosize=True,
        scene=dict(
            xaxis_title='First Dimension',
            yaxis_title='Second Dimension',
            zaxis_title='Third Dimension'
        ),
        title='Users Clustering in 3D',
        legend_title='Choose Topics',
        # Adjust legend to appear outside the plot area
        legend=dict(
            x=1.05, y=1, orientation='v',
            font=dict(size=10),  # Reduce font size for better visibility on mobile
            itemclick="toggleothers",
            itemdoubleclick="toggle"
        ),
        margin=dict(l=0, r=0, b=0, t=30)  # Reduce margins for mobile responsiveness
    )

    return fig


def display_products_by_group(products):
    st.markdown("""
    <style>
    .product-card {
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
        border-radius: 15px;
        padding: 20px;
        background-color: white;
        transition: transform 0.3s;
        text-align: center;
    }
    .product-card:hover {
        transform: scale(1.05);
        box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.2);
    }
    .product-card img {
        max-width: 100%;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .product-card a {
        text-decoration: none;
        font-weight: bold;
        color: #007BFF;
    }
    .product-card a:hover {
        color: #0056b3;
    }
    </style>
    """, unsafe_allow_html=True)
    
    for group in products['product_group'].unique():
        st.subheader(f"{group}")  # Show product group
        
        group_products = products[products['product_group'] == group]
        rows = [group_products.iloc[i:i + 3] for i in range(0, len(group_products), 3)]
        
        for row in rows:
            cols = st.columns(len(row))
            for idx, (_, product) in enumerate(row.iterrows()):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="product-card">
                        <img src="{product['image']}" alt="{product['products']}">
                        <a href="{product['link']}" target="_blank">{product['products']}</a>
                    </div>
                    """, unsafe_allow_html=True)