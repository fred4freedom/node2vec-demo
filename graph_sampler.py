"""
Node2Vec implementation using spark

This is an implementation on spark for learning an embeddings for nodes in a graph as 
described in the paper:

node2vec: Scalable Feature Learning for Networks. Aditya Grover and Jure Leskovec. 
Knowledge Discovery and Data Mining, 2016.
"""
from pyspark.sql import Window
import pyspark.sql.types as ptypes
import pyspark.sql.functions as F
from pyspark.ml.feature import Word2Vec



class GraphSampler(object):
    """
    GraphSampler


    Samples the graph to produce a dataframe of paths (for learning an embedding)

    """
    def __init__(self, return_param=1, inout_param=1):
        """
        Initialize the Graph sampler

        Params:
            return_param:     Return hyperparameter, also known as p. Default is 1.0. The smaller p 
                              is compared to q (in-out hyperparameter), the more the random walks 
                              will keep close to the starting node, which primes the model to learn 
                              structural equivalence in the graph

            inout_param:      In-out hyperparameter, also known as q. Default is 1.0. The smaller q 
                              is compared to p (return hyperparameter), the more the random walks 
                              will move away from the starting node, which primes the model to learn 
                              about communities in the graph
        """
        self.p = return_param
        self.q = inout_param


        
    @property
    def return_param(self):
        """
        Return hyperparameter, also known as p. Default is 1.0. The smaller p 
        is compared to q (in-out hyperparameter), the more the random walks 
        will keep close to the starting node, which primes the model to learn 
        structural equivalence in the graph 
        """
        return self.p
    
    
    
    @return_param.setter
    def return_param(self, value):
        """
        Set the return hyperparameter. The smaller the return hyperparameter compared to 
        the in-out hyperparameter, the more the samples will keep close to the starting 
        node, and the more the model will learn about structural equivalence in the graph
        
        Params:
            value:    value to set the return hyperparameter to
        """
        self.p = value
        
        
    
    def set_return_param(self, value):
        """
        Set the return hyperparameter. The smaller the return hyperparameter compared to 
        the in-out hyperparameter, the more the samples will keep close to the starting 
        node, and the more the model will learn about structural equivalence in the graph
        
        Params:
            value:    value to set the return hyperparameter to
          
        Returns:
            self
        """
        self.p = value
        return self
    
    
    
    @property
    def inout_param(self):
        """
        In-out hyperparameter, also known as q. Default is 1.0. The smaller q 
        is compared to p (return hyperparameter), the more the random walks 
        will move away from the starting node, which primes the model to learn 
        about communities in the graph
        """
        return self.q
      
    
    
    @inout_param.setter
    def inout_param(self, value):
        """
        Set the in-out hyperparameter. The smaller the in-out hyperparameter compared to 
        the return hyperparameter, the more the samples will explore away from the starting 
        node, and the more the model will learn about communities in the graph
        
        Params:
            value:    value to set the in-out hyperparameter to
        """
        self.q = value
        
        
        
    def set_inout_param(self, value):
        """
        Set the in-out hyperparameter. The smaller the in-out hyperparameter compared to 
        the return hyperparameter, the more the samples will explore away from the starting 
        node, and the more the model will learn about communities in the graph
        
        Params:
            value:    value to set the in-out hyperparameter to
          
        Returns:
            self
        """
        self.q = value
        return self
    


    @staticmethod
    @F.udf(ptypes.ArrayType(ptypes.StringType()))
    def add_path(*args):
        """
        Spark user defined function (udf) to concat the nodes sampled into a path 
        which is an array of nodes
        
        Params:
            args:    a variable list of columns
          
        Returns:
            a flattened array of the values in the columns
        """
        y = []
        for k in args:
            if isinstance(k, list):
                y += k
            elif k is not None:
                y.append(k)
        return y


        
    def sample(self, nodes, edges, num_samples_per_node, path_len, undirected=True, verbose=True):
        """
        Sample the given graph

        Params:
            nodes:                   Dataframe consisting of a column `node` indicating the id of the nodes 
                                     in the graph. This id must be unique for each node
            edges:                   Dataframe consisting of columns 'start` and `end` indicating an edge pointing 
                                     from a node with the id in `start` to a node with the id specified in `end`, 
                                     and an additional column `weight` that indicates the weight of the edge
            num_samples_per_node:    The number of samples to generate per node in the graph
            path_len:                The length of the path of each sample
            
        Returns:
            a dataframe with the column `path` which contains an array of the nodes of the paths 
            sampled
        """

        # If undirected, make sure the edges are symmetric
        if undirected:
            edges = (
                edges
                .select('start', 'end', 'weight')
                .union(
                    edges
                    .withColumnRenamed('start', '_end')
                    .withColumnRenamed('end', 'start')
                    .withColumnRenamed('_end', 'end')
                    .select('start', 'end', 'weight')
                )
            ).persist()
                
        # Augment the edges
        # If there are duplicates in the edges, sum the weights
        edges = (
            edges
            .localCheckpoint()
            .groupBy(['start', 'end'])
            .agg(
                F.sum('weight').alias('weight')
            )
            .withColumnRenamed('end', '_next')
            .repartition('start')
        ).persist()


        # Iterate through the number of samples per nodes
        all_samples = None
        for iter in range(num_samples_per_node):
          
            if verbose:
                print(f"Iteration #{iter + 1} / {num_samples_per_node}")

            if verbose:
                print(f"Sampling step 1 / {path_len}")
                
            # Start with the first node in the path
            samples = (
                nodes
                .select('node')
                .dropDuplicates()
                .withColumn('id', F.monotonically_increasing_id())
                .withColumnRenamed('node', '_cur')
                .withColumn('_prev', F.lit(None))
                .repartition('_cur')
            )

            # For each step in the path
            for k in range(1, path_len):

                if verbose:
                  print(f"Sampling step {k + 1} / {path_len}")

                # `_prev` is the previous node in the path, `_cur` 
                # is the current node in the path, and `end` is one 
                # of the possible next nodes in the path

                nodes_next = (
                    samples
                    .join(
                        edges,
                        on=(samples['_cur'] == edges['start']),
                        how='inner'
                    )
                    .drop(edges['start'])
                )

                # Find out which next nodes are adjacent to the previous 
                # node
                nodes_adjacent = (
                    nodes_next
                    .join(
                        edges,
                        on=(nodes_next['_prev'] == edges['start']) & (nodes_next['_next'] == edges['_next']),
                        how='left_semi'
                    )
                    .drop(edges['start'])
                )

                # Find out which next nodes are actually the previous node
                nodes_previous = (
                    nodes_next
                    .filter(F.col('_prev') == F.col('_next'))
                )

                # Find out which next nodes are neither adjacent or previous
                nodes_others = (
                    nodes_next
                    .join(nodes_adjacent, on=['id', '_next'], how='left_anti')
                    .join(nodes_previous, on=['id', '_next'], how='left_anti')
                )

                # Weigh the nodes
                nodes_adjacent = (
                    nodes_adjacent
                    .withColumnRenamed('weight', '_weight')
                    .withColumn('weight', F.col('_weight') * self.p * self.q)
                    .drop('_weight')
                )

                nodes_previous = (
                    nodes_previous
                    .withColumnRenamed('weight', '_weight')
                    .withColumn('weight', F.col('_weight') * self.q)
                    .drop('_weight')
                )

                nodes_others = (
                    nodes_others
                    .withColumnRenamed('weight', '_weight')
                    .withColumn('weight', F.col('_weight') * self.p)
                    .drop('_weight')
                )

                # Combine the next nodes
                cols = nodes_others.columns
                nodes_next = (
                    nodes_others
                    .select(cols)
                    .union(
                        nodes_adjacent
                        .select(cols)
                    )
                    .union(
                        nodes_previous
                        .select(cols)
                    )
                )

                # Compute the total weight
                nodes_weight_total = (
                    nodes_next
                    .groupBy('id')
                    .agg(
                        F.sum('weight').alias('total')
                    )
                    .select('id', 'total')
                )

                # Normalize the weights to [0, 1]
                nodes_next_norm = (
                    nodes_next
                    .join(
                        nodes_weight_total,
                        on='id',
                        how='inner'
                    )
                    .withColumn('normalized_weight', F.col('weight') / F.col('total'))
                    .drop('total')
                    .drop('weight')
                )

                # Compute the cumulative probability distribution
                win = (
                    Window
                    .partitionBy('id')
                    .orderBy('normalized_weight')
                    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
                )
                nodes_next_norm = (
                    nodes_next_norm
                    .withColumn('cumulative_probability', F.sum('normalized_weight').over(win))
                    .drop('normalized_weight')
                )

                # Sample
                win = Window.partitionBy('id').orderBy('cumulative_probability')
                cols = nodes_next_norm.columns
                samples = (
                    nodes_next_norm
                    .withColumn('_random', F.rand())    
                    .filter(F.col('_random') <= F.col('cumulative_probability'))
                    .select(*cols, 'cumulative_probability', F.first('cumulative_probability', True).over(win).alias('_selected_prob'))
                    .filter((F.col('cumulative_probability') == F.col('_selected_prob')))
                    .drop('cumulative_probability')
                    .drop('_selected_prob')
                    .drop('_random')
                    .dropDuplicates()
                )

                # Rename the columns
                samples = (
                    samples
                    .drop('_prev')
                    .withColumnRenamed('_cur', '_prev')
                    .withColumnRenamed('_next', '_cur')
                )

                # Concatenate the paths
                if k == 1:

                    samples = (
                        samples
                        .withColumn('path', self.add_path('_prev', '_cur'))
                    )

                else:

                    samples = (
                        samples
                        .withColumnRenamed('path', '_path')
                        .withColumn('path', self.add_path('_path', '_cur'))
                        .drop('_path')
                    )

                samples = samples.repartition('_cur').localCheckpoint()
                
            samples = (
              samples
              .repartition('path')
              .drop('_prev')
              .drop('_cur')
              .drop('id')
            )
            
            # Remove history of previous computations, this is important in an 
            # iterative algorithm (such as this) to prevent slow downs as the 
            # number of iterations increases
            if iter == 0:
              all_samples = samples.localCheckpoint()
            else:
              all_samples = all_samples.union(samples).localCheckpoint()
              
        return all_samples

