# cython: language_level=3, boundscheck=False
import cython 
import numpy as np
cimport numpy as np 

ctypedef np.float_t FLOAT_t
ctypedef np.int_t INT_t

@cython.boundscheck(False)
@cython.wraparound(False)
def scorf_sample (
      np.ndarray[FLOAT_t, ndim=2] X, 
      np.ndarray[FLOAT_t, ndim=2] domain,
      np.ndarray[INT_t,   ndim=2] feature, 
      np.ndarray[FLOAT_t, ndim=2] threshold, 
      np.ndarray[FLOAT_t, ndim=2] value, 
      np.ndarray[INT_t,   ndim=2] left, 
      np.ndarray[INT_t,   ndim=2] right, 
      np.ndarray[FLOAT_t, ndim=2] out, 
    ):
  "Fast implementation of sampling from the trained DET"

  cdef int nE = X.shape[0]
  cdef int nX = X.shape[1]
  cdef int nY = out.shape[1]
  cdef int nTrees = feature.shape[0]
  cdef int nNodes = feature.shape[1]
  cdef int goRight = False;
  cdef int estimated_depth = int ( np.log ( nNodes ) / np.log(2) ) 

  cdef np.ndarray [FLOAT_t, ndim=1] r = np.random.uniform ( 0, 1, nE*estimated_depth )
  cdef int i = 0;
  out = np.random.uniform ( 0, 1, (nE, nY) )
  cdef np.ndarray [INT_t, ndim=1] iTrees = np.random.randint ( 0, nTrees, nE )

  cdef np.ndarray [FLOAT_t, ndim=1] dmin = np.empty (nX+nY) 
  cdef np.ndarray [FLOAT_t, ndim=1] dmax = np.empty (nX+nY) 

  cdef int f = 0
  cdef int iTree = 0
  cdef float wRatio = 0.
  cdef float th = 0.

  cdef int iNode = 0
  for iRow in range ( nE ):
    for iVar in range ( nX+nY ): 
      dmin[iVar] = domain[iVar,0]
      dmax[iVar] = domain[iVar,1]

    iNode = 0
    iTree = iTrees[iRow]
    while feature [iTree, iNode] >= 0:
      f = feature [iTree, iNode] 
      th = threshold[iTree,iNode]

      wRatio = value[iTree,left[iTree,iNode]]/value[iTree,iNode]
      if f < nX: goRight = ( X[iRow, f] > th )
      else: 
        goRight = ( r[i%(nE*estimated_depth)] > wRatio )
        i += 1


      if goRight:
        if dmin [ f ] < th: dmin [ f ] = th
        iNode = right[iTree, iNode] 
      else: # go left
        if dmax [ f ] > th: dmax [ f ] = th
        iNode = left[iTree, iNode] 


    for f in range (nY):
      out [iRow, f] = out[iRow, f] * (dmax[nX+f]-dmin[nX+f]) + dmin[nX+f] 

  return out 



################################################################################
## BDT part 
################################################################################
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport exp2

cdef FLOAT_t _rnd ( FLOAT_t m, FLOAT_t M ):
  return m + (FLOAT_t (rand())/FLOAT_t(RAND_MAX)) * (M-m) 




################################################################################
## Rejection method with GBDT 
################################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
def scorf_bdt_sample (
      np.ndarray[FLOAT_t, ndim=2] X, 
      np.ndarray[FLOAT_t, ndim=2] domain,
      np.ndarray[INT_t,   ndim=2] feature, 
      np.ndarray[FLOAT_t, ndim=2] threshold, 
      np.ndarray[FLOAT_t, ndim=2] value, 
      np.ndarray[INT_t,   ndim=2] left, 
      np.ndarray[INT_t,   ndim=2] right, 
      float learning_rate,
      np.ndarray[FLOAT_t, ndim=2] out, 
    ):
  "Fast implementation of sampling from the trained Gradient Boosting Classifier"

  cdef int nE = X.shape[0]
  cdef int nX = X.shape[1]
  cdef int nY = out.shape[1]
  cdef int nTrees = feature.shape[0]
  cdef int nNodes = feature.shape[1]
  cdef int goRight = False;

  cdef FLOAT_t v = 0. 
  cdef FLOAT_t likelihood = 0. 

  cdef int iNone = 0;
  cdef int iRow = 0
  cdef int iTree = 0
  cdef int iAttempt  = 0

  for iRow in range(nE):
    for iAttempt in range (100): 
      for iOut in range (nY): 
        out[iRow, iOut] = _rnd ( domain[iOut+nX,0], domain[iOut+nX,1] )

      loglikelihood = 0. 
      for iTree in range(nTrees): 
        iNode = 0
        while feature [iTree, iNode] >= 0:
          if feature[iTree, iNode] < nX: ## X variable 
            v = X[iRow, feature[iTree, iNode]] 
          else: ## Y variable 
            v = out[iRow, feature[iTree, iNode] - nX]

          if v > threshold[iTree, iNode]:
            iNode = right[iTree, iNode]
          else:
            iNode = left[iTree, iNode] 

        loglikelihood += learning_rate * value[iTree, iNode] 
     
      if _rnd(0,1) < 1. / (1. + exp2(-loglikelihood)):
        #print ("@", iAttempt)
        break ## Accept 

    if iAttempt == 100: 
      raise RuntimeError ( 
          "Warning, sampling failed to converge after 100 iteration"
        )

  return out 


        
