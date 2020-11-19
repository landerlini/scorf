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

from ._traversals import scorf_sample 

class RandomForestSampler:
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
    self.forest_ = RandomForestClassifier (
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
    
    self.domain_        = None
    self.n_conditions_  = None
    self.domain_extent_ = domain_extent
    self.normalization_ratio_ = normalization_ratio
    self.cached_tree_arrays_ = None
    self.normalization_weight_ = normalization_weight

  @property 
  def n_estimators_ (self):
    "Number of estimators, property binded to `sklearn.RandomForestClassifier`"
    return self.tree.n_estimators_

  @n_estimators_.setter
  def n_estimators_ (self, newval):
    self.tree.n_estimators_ = newval


  def _update_domain (self, XY): 
    "Computes var_range to guess the domain, widening previous guess if needed"
    m = np.min ( XY, axis = 0 ) 
    M = np.max ( XY, axis = 0 ) 
    R = M-m 

    ## Creates or widen the domain 
    if self.domain_ is None:
      self.domain_ = np.stack ( [m - R*self.domain_extent_, M + R*self.domain_extent_] ).T 
    else: 
      new_domain = np.stack ( [m - R*self.domain_extent_, M + R*self.domain_extent_] ).T 
      self.domain_ = np.stack ( [
            np.minimum ( self.domain_[0], new_domain[0] ), 
            np.maximum ( self.domain_[1], new_domain[1] ), 
          ] ).T


  
  def _validate_input ( self, X, Y = None, sample_weight = None ):
    "Validates the X, Y and weight inputs, returning the merged XY dataset"
    if sample_weight is not None and len(sample_weight) != len(X): 
      raise ValueError ("Inconsistent shape of the weight input array")

    if len(X.shape) == 1: X = np.expand_dims ( X, -1 ) 
    if len(X.shape) != 2: 
      raise ValueError ( "Unexpected X rank: %d" % len(X.shape)) 


    if Y is not None:
      if len(X) != len(Y): 
        raise ValueError ("Inconsistent shape of X and Y input arrays")

      if len(Y.shape) == 1: Y = np.expand_dims ( Y, -1 ) 

      if len(Y.shape) != 2: 
        raise ValueError ( "Unexpected Y rank: %d" % len(X.shape)) 

      if self.n_conditions_ is None: self.n_conditions_ = X.shape[-1]
      if self.n_conditions_ != X.shape[-1]: 
        raise ValueError ("Number of conditions changed during training" )

      return np.concatenate ( [X, Y], axis = 1 ) 

    else: # Y is None 
      self.n_conditions_ = self.n_conditions_ or 0
      if self.n_conditions_ is None: self.n_conditions_ = 0
      if self.n_conditions_ != 0:
        raise ValueError ("Number of conditions changed during training" )

      return X



  def fit (self, X, Y = None, sample_weight = None):
    """Fit the density estimation tree.
    
    Arguments:
     - X: np.ndarray 
       the conditions of the pdf (Y|X) to be trained if Y is not None,
       otherwise the complete dataset for training non-conditional pdf(X)
       Must have shape (n_entries, n_conditional_features)

     - Y: np.ndarray or None
       the conditioned variables to be sampled, if None the unconditioned 
       pdf (X) is trained. 
       If not None, must have shape (n_entries, n_conditioned_features)

     - sample_weights: np.ndarray or None
       the weights of the training sample, if None the sample is assumed 
       unweighted. Must have shape (n_entries,) 

    """
    XY = self._validate_input ( X, Y, sample_weight ) 
    self._update_domain ( XY )

    nN = int(len(XY) * self.normalization_ratio_)
    w  = sample_weight or np.ones (len(XY), dtype = np.float)
    ws = np.full (nN, self.normalization_weight_, dtype = np.float)

    N = np.stack ([
        np.random.uniform ( row[0], row[1], nN ) for row in self.domain_ 
      ]).T 

    self.cached_tree_array = None 
    self.forest_.fit ( 
        np.concatenate ( [XY, N], axis = 0 ), 
        np.concatenate ( [np.ones(len(XY)), np.zeros(len(N))], axis = 0 ), 
        sample_weight = sample_weight
      )


  def validation_weights ( self, X, Y = None, sample_weight = None ):
    """
    Return validation weights to be used to assess the quality of the 
    training comparing the distribution of the trained sample to a 
    uniform dataset weighted with the returned trained weights. 

    Absolute scale of the weights has no meaning, only their relative 
    values has.
    """
    XY = self._validate_input ( X, Y, sample_weight ) 
    return self.forest_.predict_proba ( XY ) [:, 1] 


  def _predict_slow (self, X): 
    "Pure-python implementation of the prediction method, for debugging only"
    if len(X.shape) < 2: 
      raise ValueError ("Ambiguous array for X, did you mean np.c_[X]?") 

    if self.cached_tree_arrays_ is None:
      self._cache_trees() 

    trees = [t.tree_ for t in self.forest_.estimators_] 
    ret = np.empty ( (len(X), self.forest_.n_features_ - self.n_conditions_), dtype = X.dtype )
    for iRow, xRow in enumerate(X): 
      iTree = np.random.choice ( len(trees) )
      tree = trees[iTree] 
      v = tree.value[:,:,1]
      iNode = 0
      domain = self.domain_.copy() 
      while tree.feature[iNode] >= 0:
        r = np.random.uniform(0,1)
        wr = v[tree.children_left[iNode]]/v[iNode]
        th = tree.threshold[iNode]
        f = tree.feature [iNode]
        goRight = False 

        if f < self.n_conditions_: goRight = (xRow[f]>th)
        else: goRight = (r > wr) 

        if not goRight:
          domain [tree.feature[iNode],1] = min(domain [tree.feature[iNode],1], tree.threshold[iNode])
          iNode = tree.children_left[iNode]
        else:
          domain [tree.feature[iNode],0] = max(domain [tree.feature[iNode],0], tree.threshold[iNode])
          iNode = tree.children_right[iNode]

      ret [iRow] = np.array([np.random.uniform (row[0], row[1]) for row in domain[self.n_conditions_:]])

    return ret if len(ret.shape) == 2 else np.expand_dims (ret,0)

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

      ret = np.empty ( (len(X), self.forest_.n_features_ - self.n_conditions_ ) )

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

      ret = np.empty ( (X, self.forest_.n_features_ ) )

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
    trees = [t.tree_ for t in self.forest_.estimators_] 
    n_nodes = np.max ( [len(t.feature) for t in trees] ) 
    for var in ['feature', 'threshold', 'children_left', 'children_right']:
      self.cached_tree_arrays_ [var] = np.stack ( [
          np.resize(getattr(t,var), n_nodes) for t in trees 
        ] )
    self.cached_tree_arrays_ ['value'] = np.stack ([
        np.resize(t.value[:,:,1], n_nodes) for t in trees 
      ]) 


