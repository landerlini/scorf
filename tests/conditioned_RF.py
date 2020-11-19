################################################################################
## Generate the sample  
import numpy as np 
N = 1000
X = np.random.normal ( 0, 1, N ) 
Y = X + np.random.normal ( 100, .1, N ) 

################################################################################
## Trains a random forest to approximate the probabiliy distribution density 
from scorf import RandomForestSampler
sampler = RandomForestSampler()
sampler.fit (X, Y) 

################################################################################
## Gets weights of uniformly distributed samples to reporoduce the original pdf
uniformX, uniformY = np.random.uniform(-10,10,(2,1000))
goodw = sampler.validation_weights (uniformX, uniformY)

################################################################################
## Generate a sample according to the conditioned pdf ( Y | X )
assert sampler.predict(np.c_[X[:10]]).shape == (10,1)

################################################################################
## Generate a sample for both X and Y without conditions 
assert (sampler.predict(10).shape) == (10,2)

