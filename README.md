# scorf
Sampling with Conditions Over Random Forests

*Forest Density Estimation* is a popular technique to quickly model the 
underlying probability of large datasets. 
Since the trees are all independent, they can be trained separately, on
different chunks of the dataset, and even on different machines. 
On the other hand, the independence between the estimator enhance the 
robustness of the ensemble estimation. 

In this package we exploit forests of density estimation trees to generate 
Monte Carlo samples following the joint probability distribution function 
as a training dataset. Possibly, part of the features are constrained to 
sample a conditioned probability density function. 

### Method 
`scorf` relies on scikit-learn for the implementation of the decision trees.
Since Density Estimation Trees are not available in scikit-learn, `scorf` falls
back onto Decision Trees, trained to distinguish the training sample from a 
uniformly distributed normalization dataset. 
While suboptimal in terms of training speed, this allows to rely on the robust 
Cython-based implementations of scikit-learn and lighten significantly the maintenance 
burden of `scorf`. 

Instead, a fast *ad-hoc* implementation of the sampling method allows to transform 
the trained Decision Forest into a Cython Monte Carlo generator. 

### Example 
```
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
weights = sampler.validation_weights (uniformX, uniformY)

################################################################################
## Generate a sample according to the conditioned pdf ( Y | X )
sampler.predict(np.c_[X[:10]])

################################################################################
## Generate a sample for both X and Y without conditions 
sampler.predict(10)
```

### Related works 
While fast and robust, Random Decision Forests are not competitive with GANs in 
terms of quality of the generated dataset. 

Other packages exploring Density Estimation Trees and Forests include:
 - https://github.com/tpospisi/RFCDE
 - https://github.com/ksanjeevan/randomforest-density-python
 - https://gitlab.cern.ch/landerli/density-estimation-trees

Some of these may result into some effective replacement for scikit-learn 
DecisionTree in the future. 

### Author 
`scorf` has been developed by Lucio Anderlini (INFN Firenze) in the context 
of the exploration of techniques to speed-up the simulation of the large 
particle physics experiments at the LHC, through parametrization. 

