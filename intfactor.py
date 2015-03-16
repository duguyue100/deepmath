"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: test of integer fractorization problem
"""

import os;

from pylearn2.termination_criteria import Or;
from pylearn2.termination_criteria import MonitorBased;
from pylearn2.termination_criteria import EpochCounter;
from pylearn2.costs.mlp.dropout import Dropout;
from pylearn2.train import Train;
import pylearn2.models.mlp as mlp;
from pylearn2.models.mlp import MLP;
from pylearn2.training_algorithms.sgd import SGD;
from pylearn2.training_algorithms.sgd import ExponentialDecay;

from deepmath.intfactor_dataset import IFDataset;                  

train_data=IFDataset(path="${PYLEARN2_DATA_PATH}/deepmath/factorize_data.pkl");
valid_data=IFDataset(path="${PYLEARN2_DATA_PATH}/deepmath/factorize_data.pkl",
                     which_set="valid");
test_data=IFDataset(path="${PYLEARN2_DATA_PATH}/deepmath/factorize_data.pkl",
                     which_set="test");

print "[MESSAGE] The datasets are loaded";                     

### build model

model=MLP(layers=[mlp.Sigmoid(layer_name="hidden_0",
                              dim=300,
                              istdev=0.1),
                  mlp.Sigmoid(layer_name="hidden_1",
                              dim=300,
                              istdev=0.1),
                  mlp.Sigmoid(layer_name="y",
                              dim=13,
                              istdev=0.01,
                              monitor_style="bit_vector_class")],
          nvis=26);
          
print "[MESSAGE] The model is built";

### build algorithm

algorithm=SGD(batch_size=100,
              learning_rate=0.05,
              monitoring_dataset={'train':valid_data,
                                  'valid':valid_data,
                                  'test':test_data},
              termination_criterion=Or(criteria=[MonitorBased(channel_name="valid_objective",
                                                              prop_decrease=0.00001,
                                                              N=40),
                                                 EpochCounter(max_epochs=200)]),
              cost = Dropout(input_include_probs={'hidden_0':1., 'hidden_1':1., 'y':0.5},
                             input_scales={ 'hidden_0': 1., 'hidden_1':1., 'y':2.}),
              update_callbacks=ExponentialDecay(decay_factor=1.0000003, 
                                                min_lr=.000001));
                                                
print "[MESSAGE] Training algorithm is built";
                              
### build training

idpath = os.path.splitext(os.path.abspath(__file__))[0]; # ID for output files.
save_path = idpath + '.pkl';

train=Train(dataset=train_data,
            model=model,
            algorithm=algorithm,
            save_path=save_path,
            save_freq=100);
            
print "[MESSAGE] Trainer is built";
            
train.main_loop();