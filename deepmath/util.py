'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: Some utilites functions.
'''

import numpy as np;

def gen_factorize_data(in_file_name,
                       out_file_name):
  """
  Generate data for integer factorize problem
  
  @param in_file_name: input file name (str)
  @param out_file_name: output file name (str) 
  """
  
  d=np.loadtxt(in_file_name, dtype="int");
  d=np.ndarray.flatten(d)

  tot_len=d.size;
  idx=0;
  p=np.zeros((tot_len, 2), dtype="int")
  for i in xrange(tot_len):
    for j in xrange(i, tot_len):
      p[idx, 0]=d[i]*d[j];
      p[idx, 1]=d[i];
      idx+=1;

  in_wid=len(p[tot_len-1,0]);
  out_wid=len(p[tot_len-1,0]);
  np.random.shuffle(p);

  product=[];
  for i in xrange(tot_len):
    in_temp=np.binary_repr(p[i,0], in_wid);
    out_temp=np.binary_repr(p[i,1], out_wid);
    temp=[in_temp, out_temp];
    product.append(temp);

  product=np.asarray(product, dtype="str");
  np.savetxt(out_file_name, product, delimiter=",", fmt="%s");
  
