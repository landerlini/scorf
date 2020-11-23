"""
BaseSampler
-----------

BaseClass for a classifier-based sampler. 
"""
import numpy as np 

class BaseSampler:
  """
  BaseSampler is an abstract class defining the interface and the 
  common functions necessary to use classifiers as MC samplers.

  Arguments:

   * domain_extent - float, default: 0.1
   The domain is guessed from the training sample, with a border defined 
   relative to the span of the variables defining each axis. The width of the 
   frame is (max - min) * domain_extent 

   * normalization_ratio - float, default: 10. 
   The number of entries used for normalization relative to the input 
   sample 

   * normalization_weight - float, default: 100.
    The weight of the normalization entries relative to the signal one. 
    It should be as high as non comprimising the numerical stability of
    the training. 

  """
  def __init__ (self,
      classifier, *, 
      domain_extent = 0.1, 
      normalization_ratio = 10., 
      normalization_weight = 100., 
    ):
    self.domain_extent_ = domain_extent, 
    self.normalization_weight_ = normalization_weight
    self.normalization_ratio_ = normalization_ratio

    self.domain_        = None
    self.n_conditions_  = None
    self.cached_tree_arrays_ = None

    self.classifier_ = classifier

  @property 
  def n_estimators_ (self):
    "Number of estimators, property binded to `sklearn.RandomForestClassifier`"
    return self.classifier_.n_estimators_

  @n_estimators_.setter
  def n_estimators_ (self, newval):
    "Number of estimators"
    self.classifier_.n_estimators_ = newval

  @property 
  def domain (self):
    "Domain of estimators"
    return np.array (self.domain_) if self.domain_ is not None else None



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
            np.minimum ( self.domain_[:,0], new_domain[:,0] ), 
            np.maximum ( self.domain_[:,1], new_domain[:,1] ), 
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
    self.classifier_.fit ( 
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
    return self.classifier_.predict_proba ( XY ) [:, 1] 


