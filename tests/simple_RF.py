################################################################################
## Generate the sample  
import numpy as np 
N = 1000
X = np.random.normal ( 0, 1, (N,2) ) 

################################################################################
## Trains a random forest to approximate the probabiliy distribution density 
from scorf import RandomForestSampler
sampler = RandomForestSampler()
sampler.fit (X) 

################################################################################
## Gets weights of uniformly distributed samples to reporoduce the original pdf
uniformX = np.random.uniform(-10,10,(1000,2))
goodw = sampler.validation_weights (uniformX)

################################################################################
## Generate a sample for both X and Y without conditions 
assert (sampler.predict(10).shape) == (10,2)


