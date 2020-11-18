import numpy as np
from sklearn.ensemble import RandomForestClassifier

class RandomForestSampler:
  def __init__ (self, 
        n_estimators=100, *, 
        criterion='gini', 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
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

  @property 
  def n_estimators_ (self):
    return self.tree.n_estimators_

  @n_estimators_.setter
  def n_estimators_ (self, newval):
    self.tree.n_estimators_ = newval


  def _update_domain (self, XY): 
    m = np.min ( XY, axis = 0 ) 
    M = np.max ( XY, axis = 0 ) 
    R = M-m 

    ## Creates or widen the domain 
    if self.domain_ is None:
      self.domain_ = np.stack ( [m - R*self.domain_extent_, M + R*self.domain_extent_] ).T 
    else: 
      new_domain = np.stack ( [m - R*self.domain_extent_, M + R*self.domain_extent_] ).T 
      self.domain_ = np.stack ( [
            np.min ( self.domain_[:,0], new_domain[:,0] ), 
            np.max ( self.domain_[:,1], new_domain[:,1] ), 
          ] )


  
  def _validate_input ( self, X, Y, sample_weight = None ):
    if len(X) != len(Y): 
      raise ValueError ("Inconsistent shape of X and Y input arrays")
    if sample_weight is not None and len(sample_weight) != len(X): 
      raise ValueError ("Inconsistent shape of the weight input array")

    if len(X.shape) == 1: X = np.expand_dims ( X, -1 ) 
    if len(Y.shape) == 1: Y = np.expand_dims ( Y, -1 ) 

    if len(X.shape) != 2: 
      raise ValueError ( "Unexpected X rank: %d" % len(X.shape)) 

    if len(Y.shape) != 2: 
      raise ValueError ( "Unexpected Y rank: %d" % len(X.shape)) 

    self.n_conditions_ = self.n_conditions_ or  X.shape[-1] 
    if self.n_conditions_ != X.shape[-1]: 
      raise ValueError ("Number of conditions changed during training" )

    return np.concatenate ( [X, Y], axis = 1 ) 


  def fit (self, X, Y, sample_weight = None):
    XY = self._validate_input ( X, Y, sample_weight ) 
    self._update_domain ( XY )

    nN = int(len(XY) * self.normalization_ratio_)

    N = np.stack ([
        np.random.uniform ( row[0], row[1], nN ) for row in self.domain_ 
      ]).T 
    print (N.shape) 

    self.cached_tree_array = None 
    self.forest_.fit ( 
        np.concatenate ( [XY, N], axis = 0 ), 
        np.concatenate ( [np.ones(len(XY)), np.zeros(len(N))], axis = 0 ), 
        sample_weight = sample_weight
      )


  def validation_weights ( self, X, Y, sample_weight = None ):
    XY = self._validate_input ( X, Y, sample_weight ) 
    return self.forest_.predict_proba ( XY ) [:, 1] 


  def predict (self, X): 
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

    print (ret.shape) 
    return ret if len(ret.shape) == 2 else np.expand_dims (ret,0)


  def _cache_trees ( self ): 
    self.cached_tree_arrays_ = dict() 
    trees = [t.tree_ for t in self.forest_.estimators_] 
    n_nodes = np.max ( [len(t.feature) for t in trees] ) 
    for var in ['feature', 'threshold', 'value', 'children_left', 'children_right']:
      self.cached_tree_arrays_ [var] = np.stack ( [
          np.resize(getattr(t,var), n_nodes) for t in trees 
        ] )


