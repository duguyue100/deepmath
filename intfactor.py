"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: test of integer fractorization problem
"""

from deepmath.intfactor_dataset import IFDataset;                  


data=IFDataset(path="${PYLEARN2_DATA_PATH}/deepmath/factorize_data.pkl");

print data.X.shape;
print data.y.shape;

print data.X[0,:];

