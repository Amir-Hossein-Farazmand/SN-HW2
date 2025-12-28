import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_wiki_graph(filepath):

    G = nx.read_edgelist(filepath, create_using=nx.DiGraph(), nodetype=int, comments='#')
    return G

def compute_rankings_comparison(G):

    print("Computing HITS Authority Scores...")

    try:
        _, auth_scores = nx.hits(G, max_iter=1000, normalized=True)
    except nx.PowerIterationFailedConvergence:
        print("Warning: HITS did not converge. Using approximations.")
        _, auth_scores = nx.hits(G, max_iter=1000, tol=1e-04, normalized=True)

    print("Computing PageRank Scores (alpha=0.85)...")
    pr_scores = nx.pagerank(G, alpha=0.85)
    
    # Create DataFrame
    data = []
    for node in G.nodes():
        data.append({
            'id': node,
            'HITS_Score': auth_scores.get(node, 0),
            'PageRank_Score': pr_scores.get(node, 0)
        })
    
    df = pd.DataFrame(data)
    
    # Calculate Ranks 
    df['HITS_Rank'] = df['HITS_Score'].rank(ascending=False, method='min').astype(int)
    df['PageRank_Rank'] = df['PageRank_Score'].rank(ascending=False, method='min').astype(int)
    
    return df

def plot_rank_divergence(df):

    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    sns.scatterplot(data=df, x='HITS_Rank', y='PageRank_Rank', alpha=0.3, edgecolor=None, s=15)
    
    # Diagonal Line (y=x) for reference
    max_rank = max(df['HITS_Rank'].max(), df['PageRank_Rank'].max())
    plt.plot([1, max_rank], [1, max_rank], color='red', linestyle='--', label='Perfect Correlation (y=x)')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.title('Ranking Comparison: HITS Authority vs. PageRank (Log-Log)')
    plt.xlabel('HITS Authority Rank (Endorsement by Hubs)')
    plt.ylabel('PageRank Rank (Weighted Voting)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.show()

def identify_divergent_nodes(df, top_n=5):

    df['Div_HITS_Favored'] = df['PageRank_Rank'] - df['HITS_Rank']

    df['Div_PR_Favored'] = df['HITS_Rank'] - df['PageRank_Rank']
    
    hits_favored = df.sort_values('Div_HITS_Favored', ascending=False).head(top_n)
    pr_favored = df.sort_values('Div_PR_Favored', ascending=False).head(top_n)
    
    return hits_favored, pr_favored

def analyze_alpha_sensitivity(G, alpha_range=None):

    if alpha_range is None:
        alpha_range = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    
    print(f"Running PageRank sensitivity analysis for alphas: {alpha_range}")
    
    all_ranks = {}
    
    for alpha in alpha_range:
        scores = nx.pagerank(G, alpha=alpha)
        s_scores = pd.Series(scores)
        s_ranks = s_scores.rank(ascending=False, method='min').astype(int)
        all_ranks[f'Alpha_{alpha:.2f}'] = s_ranks
        
    df_sensitivity = pd.DataFrame(all_ranks)
    df_sensitivity.reset_index(inplace=True)
    df_sensitivity.rename(columns={'index': 'id'}, inplace=True)
    
    return df_sensitivity

def plot_rank_trajectories(df, nodes_of_interest):

    plot_data = df[df['id'].isin(nodes_of_interest)].copy()
    melted = plot_data.melt(id_vars=['id'], var_name='Alpha_Col', value_name='Rank')
    melted['Alpha'] = melted['Alpha_Col'].apply(lambda x: float(x.split('_')[1]))
    
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=melted, x='Alpha', y='Rank', hue='id', marker='o', palette='tab20', linewidth=2.5)
    
    plt.gca().invert_yaxis()
    plt.title('PageRank Stability Analysis: Rank Trajectories vs. Damping Factor (alpha)', fontsize=14)
    plt.xlabel('Damping Factor (Alpha)', fontsize=12)
    plt.ylabel('Rank (Lower is Better)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Node ID')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()