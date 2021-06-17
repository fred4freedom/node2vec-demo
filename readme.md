# Introduction

This is a simple demo on how to use the library for embedding the nodes in a graph into feature vectors, after which standard machine learning techniques can be applied. As this is an example only, the graph constructed, as well as the parameters selected are intentionally kept simple / small to make it easier to run. In real applications, more iterations and a graph where the edges are more explicitly modelled are needed for the modelling to be more accurate.

This is an implementation on spark for learning an embeddings for nodes in a graph as described in the paper:

node2vec: Scalable Feature Learning for Networks. Aditya Grover and Jure Leskovec. Knowledge Discovery and Data Mining, 2016.

# Library

The notebook `Example.ipynb` contains the simple demo on how to use the library, while the library consists of two files `graph_sampler.py` and `graph_embedder.py`.

This library is for use in Spark.

Generally, we can instantiate a model by:

```
model = Node2Vec(
    num_dim=256,
    num_samples_per_node=10,
    path_len=15,
    return_param=1,
    inout_param=1
)
```

where `num_dim` is the number of dimensions of the learnt feature vector that we want, `num_samples_per_node` are approximately the number of times we want to sample a path from each node. To ensure a good representation of the graph, the number of times we want to sample a path should be in the same order of magnitude as the average degree of the graph. `path_len` is the length of the path that we are sampling, the longer the path length, the more information we capture about the graph, but at the cost of a higher runtime. 

`return_param` and `inout_param` control the characteristics of the embeddings learnt from the graph. A smaller `return_param` relative to `inout_param` will make the embeddings learn more about the structural properties in the graph. Conversely, when `inout_param` is smaller relative to `return_param`, it learns more about the communities in the graph.

Assuming we have a `node` spark dataframe with a column `node` indicating an unique id of each node in the graph, and an `edge` spark dataframe where a `start` and `end` column points to the unique id of the nodes corresponding to the edges, and a `weight` column that weighs the relative strength of the edge, then we can learn an embedding by:

```
model.fit(nodes, edges, undirected=True, verbose=True)
```

By setting `undirected` to True (default: True), the model will make sure that the edges are symmetrical before attempting to learn an embedding. If `undirected` is False, we should make sure that the graph will have at least the given `path_len` length of path from each node in order not to have an error.

`verbose` when set to True (default: True) will print out helpful messages to track the progress of the fitting the graph. 

Subsequently, once the model is fitted, we can obtain the embeddings of each node by:

```
features = model.transform(nodes)
```

Note that the embeddings for nodes that were not found in the original graph will be `Null`.

# Demo

Start the docker container by:

```
docker-compose up
```

which will set up a local PySpark environment with Jupyter. There will be a link in the verbose messages which you should copy and paste into your browser. There, you will see a standard jupyter interface. Open up the `Example.ipynb` notebook and explore.

*Note*: The demo requires your system to have docker and docker-compose installed. You can refer to https://docs.docker.com/get-docker/ on more information in setting up Docker.



