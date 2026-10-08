import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

from sklearn.model_selection import train_test_split

#**************************************
#  Classification
#**************************************

# --------------------------------
# 1. Load dataset
# --------------------------------

# =================================
# Load Facebook Artist dataset
# =================================

import pandas as pd
import networkx as nx
import torch

from torch_geometric.utils import from_networkx


df = pd.read_csv(
    "artist_edges.csv"
)

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# =================================
# Create graph
# =================================

G = nx.from_pandas_edgelist(
    df,
    source="node_1",
    target="node_2"
)

print("\nNumber of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())


# =================================
# Convert to PyTorch Geometric
# =================================

data = from_networkx(G)


# =================================
# Create node features
# =================================

degrees = torch.tensor(
    [G.degree(node) for node in G.nodes()],
    dtype=torch.float
)

data.x = degrees.view(-1, 1)

print("\nPyTorch Geometric data:")
print(data)

print("Node features:", data.x.shape)


#**************************************
#  Community Detection
#**************************************


import networkx as nx
import community.community_louvain as community_louvain

from torch_geometric.utils import to_networkx


# Convert PyTorch Geometric graph
G = to_networkx(
    data,
    to_undirected=True
)


# Detect communities
communities = community_louvain.best_partition(G)


# Count communities
number_of_communities = len(
    set(communities.values())
)

print(
    "Number of communities:",
    number_of_communities
)


# Display first 20 nodes
for node in list(communities.keys())[:20]:

    print(
        "Node:",
        node,
        "Community:",
        communities[node]
    )

#*****************************
#  Link Analysis
#*****************************

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling


class LinkPredictionGNN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

    def encode(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x

    def decode(self, z, edge_index):

        source = z[edge_index[0]]

        target = z[edge_index[1]]

        return (
            source * target
        ).sum(dim=1)


model = LinkPredictionGNN(
    data.num_node_features,
    64
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


for epoch in range(1, 101):

    model.train()

    optimizer.zero_grad()

    z = model.encode(
        data.x,
        data.edge_index
    )

    # Existing edges
    positive_edges = data.edge_index

    # Generate non-existing edges
    negative_edges = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=positive_edges.size(1)
    )

    positive_score = model.decode(
        z,
        positive_edges
    )

    negative_score = model.decode(
        z,
        negative_edges
    )

    positive_labels = torch.ones(
        positive_score.size(0)
    )

    negative_labels = torch.zeros(
        negative_score.size(0)
    )

    scores = torch.cat([
        positive_score,
        negative_score
    ])

    labels = torch.cat([
        positive_labels,
        negative_labels
    ])

    loss = F.binary_cross_entropy_with_logits(
        scores,
        labels
    )

    loss.backward()

    optimizer.step()

    if epoch % 20 == 0:

        print(
            f"Epoch {epoch}, Loss: {loss.item():.4f}"
        )