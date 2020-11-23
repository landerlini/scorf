import numpy as np 
from scorf import GBDTSampler

N = 1000
X = np.random.normal ( 0, 1, N ) 
Y = X + np.random.normal ( 100, .1, N ) 

sampler = GBDTSampler()
sampler.fit (X, Y) 

goodw = sampler.validation_weights (X,Y)
badw  = sampler.validation_weights (X,-X)
#print (np.c_[X[:10], Y[:10], sampler.predict(np.c_[X])[:10,0], goodw[:10], badw[:10]])

print (sampler.predict(np.c_[X[:10]]))
print (sampler.predict(10).shape)


