import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
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
    # Centers của mỗi cluster
    centers = df_umap.groupby('cluster')[['first_dim', 'second_dim', 'third_dim']].mean().values

    # Màu sắc tùy chỉnh cho từng cụm
    custom_colors = ['#4535C1', '#FF8225', '#ED3EF7', '#00712D', '#F5004F', '#FFEB55', '#FF8C9E', '#836FFF']

    # Tạo figure 3D scatter plot
    fig = go.Figure()

    # Lặp qua từng cluster và thêm scatter cho mỗi cluster
    for cluster_id, center in enumerate(centers):
        cluster_points = df_umap[df_umap['cluster'] == cluster_id]

        # marker_size = 15 if cluster_id == 0 else 8

        fig.add_trace(go.Scatter3d(
            x=cluster_points['first_dim'],
            y=cluster_points['second_dim'],
            z=cluster_points['third_dim'],
            mode='markers',
            marker=dict(
                size=8,
                color=custom_colors[cluster_id],  # Áp dụng màu sắc tùy chỉnh
                opacity=0.7
            ),
            # name=f'Cluster {cluster_id + 1}'
            name=custom_labels[cluster_id]
        ))

        # Thêm tên cụm vào vị trí trung tâm của cluster
        fig.add_trace(go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode='text',
            # text=f'{cluster_id + 1}',
            textfont=dict(size=15, color='black', family='Arial'),
            showlegend=False
        ))

    # Tùy chỉnh các nhãn trục và tiêu đề
    fig.update_layout(
        autosize=True, 
        scene=dict(
            xaxis_title='First Dimension',
            yaxis_title='Second Dimension',
            zaxis_title='Third Dimension'
        ),
        title='Users Clustering in 3D',
        legend_title='Choose Topics',
        legend=dict(x=1.05, y=0.5, orientation='v',  # Điều chỉnh vị trí của legend trong biểu đồ
                    itemclick="toggleothers", itemdoubleclick="toggle")
        # width=1200,  
        # height=600,  
    )

    return fig


def plot_clusters_with_hulls(df_umap, custom_labels):
    # Tính toán tâm cụm
    centers = df_umap.groupby('cluster')[['first_dim', 'second_dim']].mean().values

    # Tạo danh sách các màu tùy chỉnh cho các cụm
    custom_colors = ['#8DEEEE', '#F4A460', '#FFDEAD','#7FFFD4','#FFB6C1','#836FFF','#FF6A6A', '#507687', '#624E88', '#654520']

    # Vẽ biểu đồ scatterplot
    
    fig, ax = plt.subplots(figsize=(12, 8))
    scatterplot = sns.scatterplot(data=df_umap, x='first_dim', y='second_dim', hue='cluster', palette=custom_colors, alpha=0.7, edgecolor='black')

    # Lặp qua từng cụm và vẽ đường bao convex hull
    for cluster_id, center in enumerate(centers):
        cluster_points = df_umap[df_umap['cluster'] == cluster_id][['first_dim', 'second_dim']].values
        hull = ConvexHull(cluster_points)
        hull_points = cluster_points[hull.vertices]
        hull_points = np.vstack((hull_points, hull_points[0]))  # Thêm điểm cuối trùng với điểm đầu tiên
        ax.fill(hull_points[:, 0], hull_points[:, 1], color=custom_colors[cluster_id], alpha=0.3)  # Vẽ màu nền cho đường bao
        ax.plot(hull_points[:, 0], hull_points[:, 1], color=custom_colors[cluster_id], linewidth=2)  # Vẽ đường bao

        # Hiển thị tên cụm với màu tương ứng
        cluster_name = f'{cluster_id + 1}'  # Thay đổi chỉ số tên cụm
        ax.text(center[0], center[1], cluster_name, fontsize=15, fontweight='bold', color="Black", ha='center', va='center')

    # Cập nhật nhãn chú thích với tên cụm tùy chỉnh
    handles, labels = scatterplot.get_legend_handles_labels()

    ax.legend(handles, custom_labels, loc='best', title='Topics')

    # Hiển thị biểu đồ
    ax.set_xlabel('First Dimension')
    ax.set_ylabel('Second Dimension')
    ax.set_title('Users Clustering')
    
    return fig