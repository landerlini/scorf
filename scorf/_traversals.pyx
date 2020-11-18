import cython 
import numpy as np
cimport numpy as np 

ctypedef np.float_t FLOAT_t
ctypedef np.int_t INT_t

#@cython.boundscheck(False)
#@cython.wraparound(False)
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

  cdef int nE = X.shape[0]
  cdef int nX = X.shape[1]
  cdef int nY = out.shape[1]
  cdef int nTrees = feature.shape[0]
  cdef int nNodes = feature.shape[1]
  cdef int goRight = False;

  cdef np.ndarray [FLOAT_t, ndim=1] r = np.random.uniform ( 0, 1, nE )
  cdef np.ndarray [FLOAT_t, ndim=2] R = np.random.uniform ( 0, 1, (nE, nY) )
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
      else: goRight = ( r[iRow] > wRatio )

      if goRight:
        if dmin [ f ] < th: dmin [ f ] = th
        iNode = right[iTree, iNode] 
      else: # go left
        if dmax [ f ] > th: dmax [ f ] = th
        iNode = left[iTree, iNode] 

    for f in range (nY):
      out [iRow, f] = R[iRow, f] * (dmax[nX+f]-dmin[nX+f]) + dmin[nX+f] 

      
  return out 


  
#    i
#
#    for iRow, xRow in enumerate(X): 
#      iTree = np.random.choice ( len(trees) )
#      tree = trees[iTree] 
#      v = tree.value[:,:,1]
#      iNode = 0
#      domain = self.domain_.copy() 
#      while tree.feature[iNode] >= 0:
#        r = np.random.uniform(0,1)
#        wr = v[tree.children_left[iNode]]/v[iNode]
#        th = tree.threshold[iNode]
#        f = tree.feature [iNode]
#        goRight = False 
#
#        if f < self.n_conditions_: goRight = (xRow[f]>th)
#        else: goRight = (r > wr) 
#
#        if not goRight:
#          domain [tree.feature[iNode],1] = min(domain [tree.feature[iNode],1], tree.threshold[iNode])
#          iNode = tree.children_left[iNode]
#        else:
#          domain [tree.feature[iNode],0] = max(domain [tree.feature[iNode],0], tree.threshold[iNode])
#          iNode = tree.children_right[iNode]
#
#      ret [iRow] = np.array([np.random.uniform (row[0], row[1]) for row in domain[self.n_conditions_:]])

