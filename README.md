# Social Network Analysis using GNN

This project performs **Social Network Analysis** on an artist network using **Graph Neural Networks (GNNs)** and graph-based techniques.

The project loads an artist edge dataset, creates a network graph, converts it into a PyTorch Geometric graph, detects communities using the Louvain algorithm, and performs link prediction using a Graph Convolutional Network (GCN).

## Features

* Load artist network edge data using Pandas
* Create a graph using NetworkX
* Analyze nodes and edges
* Convert the graph to PyTorch Geometric format
* Create node features using node degree
* Detect communities using the Louvain algorithm
* Build a GCN-based link prediction model
* Generate negative edges for link prediction
* Train the GNN for 100 epochs
* Display training loss

## Technologies Used

* Python
* Pandas
* NetworkX
* PyTorch
* PyTorch Geometric
* Scikit-learn
* Community Louvain
* GCN (Graph Convolutional Network)

## Dataset

The project uses:

```text
artist_edges.csv
```

The dataset contains connections between artists using:

* `node_1`
* `node_2`

These columns represent connections between two nodes in the artist network.

## How It Works

### 1. Load the Dataset

The artist edge dataset is loaded using Pandas.

```python
df = pd.read_csv("artist_edges.csv")
```

### 2. Create the Graph

NetworkX is used to create a graph from the artist connections.

```python
G = nx.from_pandas_edgelist(
    df,
    source="node_1",
    target="node_2"
)
```

The project then displays the number of nodes and edges.

### 3. Convert to PyTorch Geometric

The NetworkX graph is converted into a PyTorch Geometric `Data` object.

```python
data = from_networkx(G)
```

### 4. Create Node Features

The degree of each node is used as its node feature.

```python
degrees = torch.tensor(
    [G.degree(node) for node in G.nodes()],
    dtype=torch.float
)

data.x = degrees.view(-1, 1)
```

### 5. Community Detection

The Louvain algorithm is used to identify groups of closely connected artists.

```python
communities = community_louvain.best_partition(G)
```

The project then calculates and displays the number of detected communities.

### 6. Link Prediction

A two-layer Graph Convolutional Network is created using `GCNConv`.

The model learns node embeddings and uses the similarity between node embeddings to predict whether connections exist between nodes.

Existing edges are treated as positive examples, while randomly generated non-existing edges are used as negative examples.

### 7. Model Training

The model is trained for 100 epochs using the Adam optimizer and Binary Cross Entropy loss.

Example output:

```text
Epoch 20, Loss: ...
Epoch 40, Loss: ...
Epoch 60, Loss: ...
Epoch 80, Loss: ...
Epoch 100, Loss: ...
```

## Project Structure

```text
Social-Network-Analysis-GNN/
│
├── artist_edges.csv
├── Social_Network_Analysis_GNN.py
└── README.md
```

## Purpose

The purpose of this project is to understand how **Graph Neural Networks can be applied to social and artist networks**.

It demonstrates:

* Graph representation
* Node features
* Community detection
* Graph convolution
* Node embeddings
* Link prediction

## Conclusion

This project combines traditional graph analysis with deep learning to analyze relationships within an artist network. The GNN learns representations of connected nodes and uses these representations for link prediction.
