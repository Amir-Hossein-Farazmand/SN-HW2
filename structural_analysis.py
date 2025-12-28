import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse import linalg as spla
from scipy import sparse

def load_political_network(nodes_path, edges_path):

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    # Create Graph
    G = nx.from_pandas_edgelist(edges_df, 'id_1', 'id_2')
    
    # Add node attributes (name)
    node_attr = nodes_df.set_index('id')['page_name'].to_dict()
    nx.set_node_attributes(G, node_attr, 'name')
    
    return G, nodes_df

def compute_centrality_metrics(G):

    # Calculate Centralities
    # nx.degree_centrality returns Normalized Degree by default
    deg_cent = nx.degree_centrality(G)
    
    eig_cent = nx.eigenvector_centrality(G, max_iter=1000) 
    
    clo_cent = nx.closeness_centrality(G)
    
    # Create DataFrame
    data = []
    for node_id in G.nodes():
        data.append({
            'id': node_id,
            'name': G.nodes[node_id].get('name', 'Unknown'),
            'Degree': deg_cent[node_id],
            'Eigenvector': eig_cent[node_id],
            'Closeness': clo_cent[node_id]
        })
    
    df = pd.DataFrame(data)
    
    # Calculate Ranks 
    df['Degree_Rank'] = df['Degree'].rank(ascending=False, method='min').astype(int)
    df['Eigenvector_Rank'] = df['Eigenvector'].rank(ascending=False, method='min').astype(int)
    df['Closeness_Rank'] = df['Closeness'].rank(ascending=False, method='min').astype(int)
    
    return df

def plot_degree_vs_eigenvector(df):

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Degree', y='Eigenvector', alpha=0.5, edgecolor=None)
    
    plt.title('Gap Analysis: Normalized Degree vs. Eigenvector Centrality')
    plt.xlabel('Normalized Degree Centrality (Quantity)')
    plt.ylabel('Eigenvector Centrality (Quality)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def get_hidden_power_candidates(df):

    candidates = df[
        (df['Degree_Rank'] > 100) & 
        (df['Eigenvector_Rank'] <= 50)
    ].copy()
    
    return candidates.sort_values('Eigenvector_Rank')

def identify_gap_nodes(df, deg_rank_thresh=100, eig_rank_thresh=50):

    gap_nodes = df[
        (df['Degree_Rank'] > deg_rank_thresh) & 
        (df['Eigenvector_Rank'] <= eig_rank_thresh)
    ].copy()
    
    gap_nodes['Rank_Gap'] = gap_nodes['Degree_Rank'] - gap_nodes['Eigenvector_Rank']
    
    return gap_nodes.sort_values('Rank_Gap', ascending=False)

def compute_betweenness_metrics(G, df):

    print("Starting Betweenness calculation (this may take a while)...")
    # Calculate Betweenness
    bet_cent = nx.betweenness_centrality(G, normalized=True)
    
    df['Betweenness'] = df['id'].map(bet_cent)
    
    # Calculate Rank
    df['Betweenness_Rank'] = df['Betweenness'].rank(ascending=False, method='min').astype(int)
    
    return df

def get_top_bridges(df, top_n=10):

    # Sort by Betweenness
    top_bridges = df.sort_values('Betweenness', ascending=False).head(top_n).copy()
    
    top_bridges['Rank_Gap'] = top_bridges['Degree_Rank'] - top_bridges['Betweenness_Rank']
    
    return top_bridges

def identify_efficient_monitors(df):

    monitors = df[
        (df['Closeness_Rank'] <= 20) & 
        (df['Degree_Rank'] > 100)
    ].sort_values('Closeness', ascending=False)
    
    return monitors

def plot_efficiency_analysis(df, monitors_df):

    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(data=df, x='Degree', y='Closeness', alpha=0.3, color='grey', label='All Politicians')
    
    # Highlighting Efficient Monitors
    sns.scatterplot(data=monitors_df, x='Degree', y='Closeness', s=100, color='red', label='Efficient Monitors')
    
    for i in range(min(3, len(monitors_df))):
        row = monitors_df.iloc[i]
        plt.text(
            row['Degree'] + 0.0005, 
            row['Closeness'], 
            row['name'], 
            fontsize=11, 
            weight='bold', 
            color='darkred'
        )
        
    plt.title('Power in Local Structures: Efficiency Analysis')
    plt.xlabel('Normalized Degree (Communication Cost)')
    plt.ylabel('Closeness Centrality (Access Speed)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def visualize_ego_network(G, center_id, center_name, k):

    # Create Ego Graph 
    ego_G = nx.ego_graph(G, center_id, radius=1)
    
    plt.figure(figsize=(10, 10))
    
    # Layout (Force-directed)
    pos = nx.spring_layout(ego_G, k=k, iterations=50, seed=42)
    
    # Node Sizes based on Degree
    d = dict(ego_G.degree)
    node_sizes = [v * 100 for v in d.values()]
    
    # Draw edges
    nx.draw_networkx_edges(ego_G, pos, alpha=0.3, edge_color='gray')
    
    # Draw nodes
    node_colors = ['red' if node == center_id else 'blue' for node in ego_G.nodes()]
    nx.draw_networkx_nodes(ego_G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
    
    # Label only the central node
    labels = {center_id: center_name}
    nx.draw_networkx_labels(ego_G, pos, labels=labels, font_size=12, font_weight='bold', font_color='black')
    
    plt.title(f"Ego Network of {center_name}\n(Efficient Monitor)", fontsize=15)
    plt.axis('off')
    plt.show()



def compute_largest_eigenvalue(G):

    A = nx.adjacency_matrix(G).astype(float)
    
    eigenvalue, eigenvector = spla.eigs(A, k=1, which='LM')
    
    return float(eigenvalue.real)

def compute_bonacich_power(G, beta):

    A = nx.adjacency_matrix(G).astype(float)
    n = A.shape[0]
    I = sparse.eye(n)
    
    # Degree Vector
    degree_vector = A.sum(axis=1)
    
    M = I - beta * A
    
    centrality = spla.spsolve(M, degree_vector)
    
    # Normalize
    norm = np.linalg.norm(centrality)
    if norm > 0:
        centrality = centrality / norm
        
    return centrality.flatten()

def plot_bump_chart(df, top_n=10):

    subset_ids = set(df.nsmallest(top_n, 'Rank_Neutral')['id']) | \
                 set(df.nsmallest(top_n, 'Rank_Supportive')['id']) | \
                 set(df.nsmallest(top_n, 'Rank_Suppressive')['id'])
    
    plot_data = df[df['id'].isin(subset_ids)].copy()
    
    plt.figure(figsize=(14, 10))
    
    regimes = ['Suppressive', 'Neutral', 'Supportive']
    
    for _, row in plot_data.iterrows():
        y_values = [row['Rank_Suppressive'], row['Rank_Neutral'], row['Rank_Supportive']]
        
        if row['Rank_Supportive'] < row['Rank_Suppressive'] - 5: 
            color = 'green'
            alpha = 0.6
        elif row['Rank_Supportive'] > row['Rank_Suppressive'] + 5:
            color = 'red'
            alpha = 0.6
        else:
            color = 'grey' 
            alpha = 0.3
            
        plt.plot(regimes, y_values, marker='o', color=color, alpha=alpha, linewidth=2)
        
        plt.text(0, row['Rank_Suppressive'], f"{row['name']} ({int(row['Rank_Suppressive'])})", ha='right', fontsize=9, color='black')
        plt.text(2, row['Rank_Supportive'], f"{int(row['Rank_Supportive'])} {row['name']}", ha='left', fontsize=9, color='black')

    plt.gca().invert_yaxis() 
    plt.title(f'Bonacich Power Dynamics: Rank Trajectories (Top {top_n} Nodes)', fontsize=16)
    plt.xlabel('Power Regime (Beta)', fontsize=12)
    plt.ylabel('Rank (Lower is Better)', fontsize=12)
    plt.grid(True, axis='x', linestyle='--')
    plt.tight_layout()
    plt.show()

def categorize_bonacich_roles(df):

    df['Rank_Shift'] = df['Rank_Supportive'] - df['Rank_Suppressive']
    
    amplifiers = df.sort_values('Rank_Shift').head(5) 
    inhibitors = df.sort_values('Rank_Shift', ascending=False).head(5) 
    
    stable = df[df['Rank_Neutral'] <= 20].copy()
    stable['Abs_Shift'] = stable['Rank_Shift'].abs()
    stable = stable.sort_values('Abs_Shift').head(5)
    
    return amplifiers, inhibitors, stable