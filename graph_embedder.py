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

from graph_sampler import GraphSampler


class ModelNotFitted(Exception):
  """
  Model has not been fitted
  """
  pass

        

class Node2Vec(object):
    """
    Node2Vec


    Learns an embedding from a graph

    """
    def __init__(self, num_dim=256, num_samples_per_node=2, path_len=15, return_param=1, inout_param=1):
        """
        Initialize the Node2Vec model

        Params:
            num_dim:                 The number of dimensions for the embeddings
            num_samples_per_node:    The number of samples to generate per node in the graph
            
            path_len:                The length of the path of each sample
            
            return_param:            Return hyperparameter, also known as p. Default is 1.0. The smaller p 
                                     is compared to q (in-out hyperparameter), the more the random walks 
                                     will keep close to the starting node, which primes the model to learn 
                                     structural equivalence in the graph

            inout_param:             In-out hyperparameter, also known as q. Default is 1.0. The smaller q 
                                     is compared to p (return hyperparameter), the more the random walks 
                                     will move away from the starting node, which primes the model to learn 
                                     about communities in the graph
        """
        self._num_dim = num_dim
        self._num_samples_per_node = num_samples_per_node
        self._path_len = path_len
        self.p = return_param
        self.q = inout_param
        self.model = None
        self.fitted_model = None
        
        

    @property
    def num_dim(self):
      """
      Number of dimensions of the embedding to learn
      """
      return self._num_dim
    
    
    
    @num_dim.setter
    def num_dim(self, value):
      """
      Set the number of dimensions of the embedding to learn
      
      Params:
          value:    The number of dimensions of the embedding
      """
      self._num_dim = value
      
      
      
    def set_num_dim(self, value):
      """
      Set the number of dimensions of the embedding to learn
      
      Params:
          value:    The number of dimensions of the embedding
          
      Returns:
          self
      """
      self._num_dim = value
      return self

    
    
    @property
    def num_samples_per_node(self):
      """
      Number of paths to sample per node to be used for learning the embedding
      """
      return self._num_samples_per_node
    
    
    
    @num_samples_per_node.setter
    def num_samples_per_node(self, value):
      """
      Set the number of paths to sample per node to be used for learning the embedding
      
      Params:
          value:    The number of paths to sample per node
      """
      self._num_samples_per_node = value
      
      
      
    def set_num_samples_per_node(self, value):
      """
      Set the number of paths to sample per node to be used for learning the embedding
      
      Params:
          value:    The number of paths to sample per node
          
      Returns:
          self
      """
      self._num_samples_per_node = value
      return self

    
    
    @property
    def path_len(self):
      """
      Length of path to sample to be used for learning the embedding
      """
      return self._path_len
    
    
    
    @path_len.setter
    def path_len(self, value):
      """
      Set the Length of path to sample to be used for learning the embedding
      
      Params:
          value:    The length of the path to sample
      """
      self._path_len = value
      
      
      
    def set_path_len(self, value):
      """
      Set the Length of path to sample to be used for learning the embedding
      
      Params:
          value:    The length of the path to sample
          
      Returns:
          self
      """
      self._path_len = value
      return self
    

    
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

    
    def _fit_samples(self, samples):
      """
      Fit the model to the generated samples
      
      Params:
          samples:    Dataframe with a column `path` containing an array of node ids
          
      Returns:
          self, trained with the samples
      """
      # Get the maximum length of path
      samples_path_len = (
        samples
        .withColumn('path_len', F.size('path'))
      )
      max_path_len = (
        samples_path_len
        .groupBy(F.lit(1))
        .agg(
          F.max('path_len').alias('max_path_len')
        )
        .collect()
      )[0]['max_path_len']
      
      # Compute the window size
      window_size = max(1, max_path_len // 2)
      
      self.model = Word2Vec(
        inputCol='path',
        minCount=1,
        windowSize=window_size,
        maxSentenceLength=max_path_len
      ).setVectorSize(self.num_dim)
      
      # Fit the model
      self.fitted_model = self.model.fit(samples)
      
      return self
      
      

    def fit_samples(self, samples):
      """
      Fit the model to the generated samples
      
      Params:
          samples:    Dataframe with a column `path` containing an array of node ids
          
      Returns:
          self, trained with the samples
      """
      return self._fit_samples(samples)
    
    
      
    def fit(self, nodes, edges, undirected=True, verbose=True):
      """
      Fit the model to the given nodes and edges
      
      Params:
            nodes:                   Dataframe consisting of a column `node` indicating the id of the nodes 
                                     in the graph. This id must be unique for each node
            edges:                   Dataframe consisting of columns 'start` and `end` indicating an edge pointing 
                                     from a node with the id in `start` to a node with the id specified in `end`, 
                                     and an additional column `weight` that indicates the weight of the edge
          
      Returns:
            self, trained with the samples
      """
      sampler = GraphSampler(
        return_param=self.return_param,
        inout_param=self.inout_param
      )
      
      # Sample from the graph
      samples = sampler.sample(
          nodes, 
          edges, 
          num_samples_per_node=self.num_samples_per_node, 
          path_len=self.path_len,
          undirected=undirected,
          verbose=verbose
      )
      
      # Fit from the samples
      return self._fit_samples(samples)
      
      
    @staticmethod
    @F.udf(ptypes.ArrayType(ptypes.FloatType()))
    def to_array(x):
      """
      Convert column to an array
      
      Params:
          x:    column to convert to an array
          
      Returns:
          Column converted to an array
      """
      return x.toArray().tolist()
      
      
      
    def transform(self, nodes, flag_default_zero=False):
      """
      Transform the given nodes from a fitted model
      
      Params:
            nodes:                   Dataframe consisting of a column `node` indicating the id of the nodes 
                                     in the graph. This id must be unique for each node
            flag_default_zero:       Boolean indicating whether to return a zero vector if node does not exist 
                                     in the fitted model (if set to true), else returns null
          
      Returns:
            A dataframe with an added column `feature` which is an array of floats corresponding to the embedding 
            of the nodes
      """
      
      if not self.model or not self.fitted_model:
        raise ModelNotFitted
        
      vectors = (
        self.fitted_model
        .getVectors()
        .withColumnRenamed('word', 'node')
        .withColumn('feature', self.to_array('vector'))
        .drop('vector')
      ).cache()
      
      # Compute the features
      result = (
        nodes
        .join(
          vectors,
          on='node',
          how='left'
        )
      ).cache()
      
      # Fill the nulls if needed
      if flag_default_zero:
        
          result_null = (
            result
            .filter(F.col('feature').isNull())
          )
          
          result_nonnull = (
            result
            .filter(F.col('feature').isNotNull())
          )
          
          result_null = (
            result
            .drop('feature')
            .withColumn('feature', F.array([F.lit(0.0) for k in range(self.num_dim)]))
          )
          
          cols = result_nonnull.columns
          result = (
            result_nonnull
            .select(cols)
            .union(
              result_null
              .select(cols)
            )
          )
          
      # Returns the result
      return result
