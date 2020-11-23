"""
GBDTSampler
-----------

Gradient Boosting usually outperforms Random Forest classifiers, 
but it does not offer a sampling technique as efficient. 
In this implementation, a GradientBoosting classifier is used 
to learn the joint pdf of the training sample, while a rejection 
method is used to formulate the predictions. 

Preprocessing using quantile transform is highly recommended to enhance 
the prediction efficiency. 
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from ._traversals import scorf_sample, scorf_bdt_sample
from .BaseSampler import BaseSampler

class GBDTSampler (BaseSampler):
  """
  GBDTSampler
  -----------

  A thin layer built on top of scikit-learn `GradientBoostingClassifier`
  to learn the distribution of an input dataset and efficiently 
  sampling it. 

  Most of the input parameters are passed to `GradientBoostingClassifier`
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
        loss='deviance', 
        learning_rate=0.1, 
        n_estimators=100, 
        subsample=1.0, 
        criterion='friedman_mse', 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, 
        max_depth=3, 
        min_impurity_decrease=0.0, 
        min_impurity_split=None, 
        init=None, 
        random_state=None, 
        max_features=None, 
        verbose=0, 
        max_leaf_nodes=None, 
        warm_start=False, 
        presort='deprecated', 
        validation_fraction=0.1, 
        n_iter_no_change=None, 
        tol=0.0001, 
        ccp_alpha=0.0, 
        ##
        domain_extent = 0.1, 
        normalization_ratio = 10., 
        normalization_weight = 100., 
      ):

    ## Create the forest 
    classifier = GradientBoostingClassifier (
        loss = loss, 
        learning_rate = learning_rate, 
        n_estimators = n_estimators, 
        subsample = subsample, 
        criterion = criterion, 
        min_samples_split = min_samples_split, 
        min_samples_leaf = min_samples_leaf, 
        min_weight_fraction_leaf = min_weight_fraction_leaf, 
        max_depth = max_depth, 
        min_impurity_decrease = min_impurity_decrease, 
        min_impurity_split = min_impurity_split, 
        init = init, 
        random_state = random_state, 
        max_features = max_features, 
        verbose = verbose, 
        max_leaf_nodes = max_leaf_nodes, 
        warm_start = warm_start, 
        presort = presort, 
        validation_fraction = validation_fraction, 
        n_iter_no_change = n_iter_no_change, 
        tol = tol, 
        ccp_alpha = ccp_alpha, 
      )

    BaseSampler.__init__ (self, classifier, 
          domain_extent = 0.1, 
          normalization_ratio = 10., 
          normalization_weight = 100., 
        )
    self._learning_rate = learning_rate
    
  @property 
  def learning_rate (self):
    "Learning rate used to train the BDT"
    return self._learning_rate

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
 
      return (scorf_bdt_sample( X, 
                self.domain_,
                self.cached_tree_arrays_ ['feature'], 
                self.cached_tree_arrays_ ['threshold'], 
                self.cached_tree_arrays_ ['value'], 
                self.cached_tree_arrays_ ['children_left'], 
                self.cached_tree_arrays_ ['children_right'], 
                self.learning_rate, 
                ret
              ))
    elif isinstance (X, int):
      if X <= 0:
        raise ValueError ("Can not produce a negative number of events") 
 
      if self.cached_tree_arrays_ is None:
        self._cache_trees() 
 
      ret = np.empty ( (X, self.classifier_.n_features_ ) )
 
      return (scorf_bdt_sample ( np.empty ( (X,0)), 
                self.domain_,
                self.cached_tree_arrays_ ['feature'], 
                self.cached_tree_arrays_ ['threshold'], 
                self.cached_tree_arrays_ ['value'], 
                self.cached_tree_arrays_ ['children_left'], 
                self.cached_tree_arrays_ ['children_right'], 
                self.learning_rate, 
                ret
              ))



  def _cache_trees ( self ): 
    "Internal. Stores all trees in a set of large arrays readable with Cython"
    self.cached_tree_arrays_ = dict() 
    trees = [t[0].tree_ for t in self.classifier_.estimators_] 
    n_nodes = np.max ( [len(t.feature) for t in trees] ) 
    for var in ['feature', 'threshold', 'children_left', 'children_right']:
      self.cached_tree_arrays_ [var] = np.stack ( [
          np.resize(getattr(t,var), n_nodes) for t in trees 
        ] )
    self.cached_tree_arrays_ ['value'] = np.stack ([
        np.resize(t.value[:,0,0], n_nodes) for t in trees 
      ]) 



