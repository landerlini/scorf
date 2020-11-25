"""
RandomForestSampler
-------------------

The construction of MC generators able to reproduce distributions obtained 
from a dataset of real data is of crucial importance to parametrize the 
response of complex systems in simulation and what-if studies. 
`RandomForestSampler` is a thin overlay on top of scikit-learn 
`RandomForestClassifier` designed to learn the distribution of a 
training dataset from the comparison with a uniformly distributed sample.
The trained forest can then be used to sample a single tree, and from 
that tree a leaf, possibly depending on some condition of part of the 
variables. A new data is obtained from uniform generation within the 
leaf volume. 

"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .BaseSampler import BaseSampler

from ._traversals import scorf_sample 

class RandomForestSampler (BaseSampler):
  """
  RandomForestSampler
  -------------------

  A thin layer built on top of scikit-learn `RandomForestClassifier`
  to learn the distribution of an input dataset and efficiently 
  sampling it. 

  Most of the input parameters are passed to `RandomForestClassifier`
  but some default value was tweaked to make it more sensible for 
  density estimation. 

  Additional parameters are :
   - domain_extent (float, default: 0.1)
     The domain of the input dataset is estimated from the training 
     data set, a frame of `domain_extent`*var_range is built all around
     the domain to ensure the training dataset is fully contained

   - normalization_ratio (float, default: 10)
     A uniform dataset is generated to build a density estimation tree
     out of a decision tree. The number of generated events is 
     `normalization_ratio` times the number of events passed as 
     training dataset
  
   - normalization_weight (float, default: 100)
     a weight applied during the training to the normalization dataset.
     It should be as large as not compromising numerical accuracy 
  
  """
  def __init__ (self, 
        n_estimators=100, *, 
        criterion='entropy', 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=5, 
        min_weight_fraction_leaf=0.0, 
        max_features='auto', 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.0, 
        min_impurity_split=None, 
        bootstrap=True, 
        oob_score=False, 
        n_jobs=None, 
        random_state=None, 
        verbose=0, 
        warm_start=False, 
        class_weight=None, 
        ccp_alpha=0.0, 
        max_samples=None,
        ##
        domain_extent = 0.1, 
        normalization_ratio = 10., 
        normalization_weight = 100., 
      ):

    ## Create the forest 
    classifier = RandomForestClassifier (
        n_estimators = n_estimators, 
        criterion = criterion, 
        max_depth = max_depth, 
        min_samples_split = min_samples_split, 
        min_samples_leaf = min_samples_leaf, 
        min_weight_fraction_leaf = min_weight_fraction_leaf, 
        max_features = max_features, 
        max_leaf_nodes = max_leaf_nodes, 
        min_impurity_decrease = min_impurity_decrease, 
        min_impurity_split = min_impurity_split, 
        bootstrap = bootstrap, 
        oob_score = oob_score, 
        n_jobs = n_jobs, 
        random_state = random_state, 
        verbose = verbose, 
        warm_start = warm_start, 
        class_weight = class_weight, 
        ccp_alpha = ccp_alpha, 
        max_samples = max_samples
      )
    
    BaseSampler.__init__ (self, classifier, 
          domain_extent = domain_extent, 
          normalization_ratio = normalization_ratio, 
          normalization_weight = normalization_weight, 
        )


  def predict (self, X): 
    """
    Randomly generates a sample distributed according to the underlying pdf of 
    the training sample. 

    If the parameter X is an array, it must define the conditions to the 
    generated sample, and only the generated Y variables are returned. 
    If it is instead an integer, then both X and Y variables are generated 
    and returned in a unique, stacked array. 

    Parameters:
     - X: np.ndarray or int
       Either the conditions for the sampled dataset, 
       expressed as a (n_entries, n_conditional_features) array; 
       or an integer defining the number of samples to be generated.

    Returns: np.ndarray
      An array with shape (n_entries, n_features) where n_features is 
      n_conditional_features + n_conditioned_features if X is an integer 
      defining n_entries. Or, otherwise, of an array of shape 
      (n_entries, n_conditioned_features). 

    """
    if isinstance (X, np.ndarray):
      if len(X.shape) < 2: 
        raise ValueError ("Ambiguous array for X, did you mean np.c_[X]?") 

      if self.cached_tree_arrays_ is None:
        self._cache_trees() 

      ret = np.empty ( (len(X), self.classifier_.n_features_ - self.n_conditions_ ) )

      return (scorf_sample ( X, 
                self.domain_,
                self.cached_tree_arrays_ ['feature'], 
                self.cached_tree_arrays_ ['threshold'], 
                self.cached_tree_arrays_ ['value'], 
                self.cached_tree_arrays_ ['children_left'], 
                self.cached_tree_arrays_ ['children_right'], 
                ret
              ))
    elif isinstance (X, int):
      if X <= 0:
        raise ValueError ("Can not produce a negative number of events") 

      if self.cached_tree_arrays_ is None:
        self._cache_trees() 

      ret = np.empty ( (X, self.classifier_.n_features_ ) )

      return (scorf_sample ( np.empty ( (X,0)), 
                self.domain_,
                self.cached_tree_arrays_ ['feature'], 
                self.cached_tree_arrays_ ['threshold'], 
                self.cached_tree_arrays_ ['value'], 
                self.cached_tree_arrays_ ['children_left'], 
                self.cached_tree_arrays_ ['children_right'], 
                ret
              ))

  def _cache_trees ( self ): 
    "Internal. Stores all trees in a set of large arrays readable with Cython"
    self.cached_tree_arrays_ = dict() 
    trees = [t.tree_ for t in self.classifier_.estimators_] 
    n_nodes = np.max ( [len(t.feature) for t in trees] ) 
    for var in ['feature', 'threshold', 'children_left', 'children_right']:
      self.cached_tree_arrays_ [var] = np.stack ( [
          np.resize(getattr(t,var), n_nodes) for t in trees 
        ] )
    self.cached_tree_arrays_ ['value'] = np.stack ([
        np.resize(t.value[:,:,1], n_nodes) for t in trees 
      ]) 


