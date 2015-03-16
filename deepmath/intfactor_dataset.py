'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: Provide a integer factorization dataset wrapper
'''

import numpy as np;

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix;
from pylearn2.utils.string_utils import preprocess;

class IFDataset(DenseDesignMatrix):
  """
  A special dataset sructure that reads data from csv file for
  integer factorization problem.
  
  The file should contains many rows, each row has two columns.
  each column is a string type data, the first column is input data to network
  and the second is the expected data.
  
  
  """
  
  def __init__(self,
               path,
               expect_headers=False,
               delimiter=",",
               which_set="train"):
    """
    @param path: path of a data, should be a csv file (str)
    @param expect_headers: if there is a header on the first row (bool)
    @param delimiter: delimiter of the data (str)
    @param which_set: specify which set is using (total, train, valid, test)
    """
    
    self.path=path;
    self.delimiter=delimiter;
    self.expect_headers=expect_headers;
    self.which_set=which_set;
    
    self.path=preprocess(self.path);
    
    X, y=self._load_data();
    
    start=0;
    end=X.shape[0];
    if self.which_set=="train":
      start=0;
      end*=0.6;
    elif self.which_set=="valid":
      start=end*0.6;
      end*=0.8;
    elif self.which_set=="test":
      start=end*0.8;
      
    X=X[:, start:end];
    y=y[:, start:end];
      
    
    super(IFDataset, self).__init__(X=X,
                                    y=y);
    
  def _load_data(self):
    """
    Load dataset
    """
    
    assert self.path.endswith(".csv");
    
    if self.expect_headers:
      data = np.loadtxt(self.path,
                        delimiter=self.delimiter,
                        skiprows=1)
    else:
      data = np.loadtxt(self.path, delimiter=self.delimiter);
    
    X=data[:,0];
    y=data[:,1];
    
    X, y=self.transform_data(X, y);
      
    return X, y;
  
  def transform_data(self, X, y):
    """
    transform data from string format to vector format
    """
    
    X_trans=np.asarray([]);
    y_trans=np.asarray([]);
    
    tot_len=X.size;
    
    for i in xrange(tot_len):
      X_temp=np.asarray(list(X[i]), dtype="float");
      y_temp=np.asarray(list(y[i]), dtype="float");
      
      if not X_trans.size:
        X_trans=X_temp;
      else:
        X_trans=np.hstack((X_trans, X_temp));
        
      if not y_temp.size:
        y_trans=y_temp;
      else:
        y_trans=np.hstack((y_trans, y_temp));
    
    return X_trans, y_trans;