'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: Some utilites functions.
'''

import numpy as np;
import cPickle as pickle;

def gen_factorize_data(in_file_name,
                       out_file_name):
  """
  Generate data for integer factorize problem
  
  @param in_file_name: input file name (str)
  @param out_file_name: output file name (str) 
  """
  
  d=np.loadtxt(in_file_name, dtype="int");
  d=np.ndarray.flatten(d)

  tot_len=(1+d.size)*d.size/2;
  idx=0;
  p=np.zeros((tot_len, 2), dtype="int")
  for i in xrange(d.size):
    for j in xrange(i, d.size):
      p[idx, 0]=d[i]*d[j];
      p[idx, 1]=d[i];
      idx+=1;

  in_wid=len(np.binary_repr(p[tot_len-1,0]));
  out_wid=len(np.binary_repr(p[tot_len-1,1]));
  np.random.shuffle(p);

  X=np.zeros((tot_len, in_wid));
  y=np.zeros((tot_len, out_wid));
  for i in xrange(tot_len):
    X[i, :]=np.asarray(list(np.binary_repr(p[i,0], in_wid)), dtype="uint8");
    y[i, :]=np.asarray(list(np.binary_repr(p[i,1], out_wid)), dtype="uint8");
    print "Processed %d" % i;

  f=open(out_file_name, "w");
  pickle.dump((X, y), f);
  
