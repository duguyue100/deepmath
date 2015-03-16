'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: Provide a integer factorization dataset wrapper
'''

import cPickle as pickle;

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix;
from pylearn2.utils.string_utils import preprocess;

class IFDataset(DenseDesignMatrix):
  """
  A special dataset sructure that reads data from pkl file for
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
    @param path: path of a data, should be a pkl file (str)
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
      
    X=X[start:end, :];
    y=y[start:end, :];
        
    super(IFDataset, self).__init__(X=X, y=y);
    
  def _load_data(self):
    """
    Load dataset
    """
    
    assert self.path.endswith(".pkl");
    
    X, y=pickle.load(open(self.path,"r"));
      
    return X, y;