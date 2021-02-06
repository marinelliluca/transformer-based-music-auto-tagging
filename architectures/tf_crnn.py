""" Example Usage:

import tensorflow as tf
import numpy as np
import os
from crnn import get_model

tfrecords_folder = <path_to_folder>

initial_biases = np.load(os.path.join(tfrecords_folder,"initial_biases.npy"),allow_pickle='TRUE').item()

# the initial biases dictionary was saved by compute_tf_dataset.ipynb in the given tfrecords_folder
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
                          
architecture_parameters = {"model_name": "splitCRNN",
                           "input_shape": spec_shape,
                           "output_type": "split",
                           "n_rec_units": [1536,1536], # pass None if the recurrent block is not needed
                           "last_dropout_prob": 0.5,
                           "initial_biases":initial_biases,
                           "conv_blocks":
                                   {"n_filters":[64,128,128,256,256,512,1024], 
                                    "kernel_sizes":[(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3)],
                                    "pool_sizes":  [(2,2),(2,2),(2,2),(2,2),(2,2),(2,2),(1,2)], # to leave out pooling, make pool size (1,1) for a layer
                                    "avgpool_flags":[False,False,False,False,False,False,True], # if False, max pooling instead of avg pooling. global avg pooling for regularization
                                    "normalization_axes": [3,3,3,3,3,3,3] # channel axis
                                    }
                          }

model = get_model(architecture_parameters)
model.summary()
tf.keras.utils.plot_model(model)#,rankdir='LR')
"""

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Input, layers
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Dropout, Lambda
from tensorflow.keras.layers import GRU, Bidirectional, Dense, BatchNormalization, ELU, Flatten

# reference papers:
# MULTI-LABEL MUSIC GENRE CLASSIFICATION  https://arxiv.org/abs/1707.04916
# CRNN for Music Classification https://arxiv.org/abs/1609.04243

class ConvolutionalBlock(layers.Layer):
    """ 
    Performs feature extraction on 2D inputs
    Submodelling: ConvolutionalBlock is a child of tf.keras.layers.Layer
    """
    
    def __init__(self,
                 n_filters,
                 kernel_size,
                 norm_axis,
                 pool_size,
                 avgpool_flag,
                 is_last_conv_block,
                 name_suffix,
                 **kwargs):
        super(ConvolutionalBlock, self).__init__(name= "conv_block_" + 
                                                 name_suffix, **kwargs)
        self.config = {"n_filters":n_filters,
                       "kernel_size":kernel_size,
                       "norm_axis":norm_axis,
                       "pool_size":pool_size,
                       "avgpool_flag":avgpool_flag,
                       "is_last_conv_block":is_last_conv_block,
                       "name_suffix":name_suffix}
        
        self.conv = Convolution2D(n_filters,
                                  kernel_size,
                                  padding='same')
        
        self.batchnorm = BatchNormalization(axis=norm_axis)
        self.activation = ELU() # ELU less heavy than lReLu with almost as good performance (https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/ ?)
        if avgpool_flag:
            self.pool = AveragePooling2D(pool_size)
        else:
            self.pool = MaxPooling2D(pool_size)
        self.dropout = Dropout(0.1) # as in https://arxiv.org/abs/1609.04243 
        
        if is_last_conv_block:
            # squeeze freq_axis, tf.squeeze() and raise an exception if freq_axis size is not 1
            self.squeeze_freq = Lambda(lambda x: tf.squeeze(x, [2])) 
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x) # pool before normalize as suggested in https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/
        x = self.batchnorm(x)
        x = self.activation(x)       
        x = self.dropout(x)
        if self.config["is_last_conv_block"]:
            try:
                x = self.squeeze_freq(x)
            except ValueError:
                raise Exception("Insufficient pooling along the freq_axis, "+
                                "the required resulting size is 1.")
        return x
    
    def get_config(self):
        return self.config
    
    
class RecurrentBlock(layers.Layer):
    """ 
    Aggregates a temporal sequence of features vectors into a single vector
    Submodelling: RecurrentBlock is a child of tf.keras.layers.Layer
    """
    def __init__(self, 
                 n_rec_units, # list of 2 integers: [n1, n2]
                 name_suffix, 
                 **kwargs):
        super(RecurrentBlock, self).__init__(name="recurrent_block_"+
                                             name_suffix, **kwargs)   
        
        self.config = {"n_rec_units":n_rec_units,
                       "name_suffix":name_suffix}

        self.GRU1 = GRU(n_rec_units[0],
                    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
                    # requirements to use the FAST cuDNN implementation
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        recurrent_dropout=0,
                        unroll=False,
                        use_bias=True,
                        reset_after=True, 
                        return_sequences=True) # NB: returns sequence
        
        self.GRU2 = GRU(n_rec_units[1],
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        recurrent_dropout=0,
                        unroll=False,
                        use_bias=True,
                        reset_after=True,
                        return_sequences=False) # NB: returns single vector
        
    def call(self, features_sequence):
        x = self.GRU1(features_sequence)
        x = self.GRU2(x)
        return x
    
    def get_config(self):
        return self.config

    
class BidirectionalRecurrentBlock(layers.Layer):
    """ 
    Aggregates a temporal sequence of features vectors into a single vector
    This did not prove to be more efficient than RecurrentBlock but is way heavier.
    Therefore, use RecurrentBlock instead in usual cases.
    """
    def __init__(self, 
                 n_rec_units, # list of 2 integers: [n1, n2]
                 name_suffix, 
                 **kwargs):
        super(BidirectionalRecurrentBlock, self).__init__(name="recurrent_block_"+
                                                          name_suffix, **kwargs)   
        
        self.config = {"n_rec_units":n_rec_units,
                       "name_suffix":name_suffix}

        self.bidirectional_GRU1 = Bidirectional(GRU(n_rec_units[0],
                            # https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
                            # requirements to use the FAST cuDNN implementation
                                                    activation='tanh',
                                                    recurrent_activation='sigmoid',
                                                    recurrent_dropout=0,
                                                    unroll=False,
                                                    use_bias=True,
                                                    reset_after=True, 
                                                    return_sequences=True)) # NB: returns sequence
        self.bidirectional_GRU2 = Bidirectional(GRU(n_rec_units[1],
                                                    activation='tanh',
                                                    recurrent_activation='sigmoid',
                                                    recurrent_dropout=0,
                                                    unroll=False,
                                                    use_bias=True,
                                                    reset_after=True,
                                                    return_sequences=False)) # NB: returns single vector
        
    def call(self, features_sequence):
        x = self.bidirectional_GRU1(features_sequence)
        x = self.bidirectional_GRU2(x)
        return x
    
    def get_config(self):
        return self.config


class BinaryClassifier(layers.Layer):
    """ Binary Classification layer """
    def __init__(self, 
                 dropout_probability, 
                 output_suffix,
                 initial_bias,
                 **kwargs):
        super(BinaryClassifier, self).__init__(name="output_"+
                                               output_suffix, **kwargs)
        
        self.config = {"dropout_probability":dropout_probability,
                       "output_suffix":output_suffix,
                       "initial_bias":initial_bias}
        
        self.flatten = Flatten()
        
        self.dropout = Dropout(dropout_probability)        

        # cfr. https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
        self.dense = Dense(1, 
                           activation="sigmoid",
                           kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                           bias_initializer = tf.keras.initializers.Constant(initial_bias))
        # NB: loss == binary_crossentropy
    
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dropout(x)
        x = self.dense(x)
        return x
    
    def get_config(self):
        return self.config
    
    
class SoftmaxClassifier(layers.Layer):
    """ Softmax Classification layer """
    def __init__(self, 
                 dropout_probability, 
                 output_suffix,
                 n_classes,
                 **kwargs):
        super(SoftmaxClassifier, self).__init__(name="output_"+
                                                output_suffix, **kwargs)
        
        self.config = {"dropout_probability":dropout_probability,
                       "output_suffix":output_suffix,
                       "n_classes":n_classes}
        
        self.flatten = Flatten()
        
        self.dropout = Dropout(dropout_probability)        

        self.dense = Dense(n_classes, 
                           activation="softmax", 
                           kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
        # NB: loss == categorical_crossentropy
    
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dropout(x)
        x = self.dense(x)
        return x
    
    def get_config(self):
        return self.config
    
    
class SigmoidClassifier(layers.Layer):
    """ Sigmoid Classification layer """
    def __init__(self, 
                 dropout_probability, 
                 output_suffix,
                 n_classes,
                 **kwargs):
        super(SigmoidClassifier, self).__init__(name="output_"+
                                                output_suffix, **kwargs)
        
        self.config = {"dropout_probability":dropout_probability,
                       "output_suffix":output_suffix,
                       "n_classes":n_classes}
        
        self.flatten = Flatten()
        
        self.dropout = Dropout(dropout_probability)        

        self.dense = Dense(n_classes, 
                           activation="sigmoid", 
                           kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
        # NB: loss == categorical_crossentropy
    
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dropout(x)
        x = self.dense(x)
        return x
    
    def get_config(self):
        return self.config

    
def _check_architecture_parameters(architecture_parameters):
    
    # convolutional blocks parameters
    n_filters = architecture_parameters["conv_blocks"]["n_filters"]
    kernel_sizes = architecture_parameters["conv_blocks"]["kernel_sizes"]
    pool_sizes = architecture_parameters["conv_blocks"]["pool_sizes"]
    avgpool_flags = architecture_parameters["conv_blocks"]["avgpool_flags"]
    normalization_axes = architecture_parameters["conv_blocks"]["normalization_axes"]
    
    if not len(n_filters)==len(pool_sizes)==len(kernel_sizes)==len(avgpool_flags)==len(normalization_axes):
        raise Exception("All lists in architecture_parameters['conv_blocks'] "+
                        "must have the same length.")
    
    # global parameters
    n_rec_units = architecture_parameters["n_rec_units"]
        
    if n_rec_units is not None and len(n_rec_units)!=2:
        raise Exception("The number of elements in 'n_rec_units' must be 2.")
    
    last_dropout_prob = float(architecture_parameters["last_dropout_prob"])
    if not (last_dropout_prob>=0. and last_dropout_prob<1.):
        raise Exception("'last_dropout_prob' must be in [0,1).")
    
    if architecture_parameters['output_type'] not in ["split","softmax","sigmoid"]:
        raise Exception("Currently supported output types = 'split', 'softmax' or 'sigmoid'.")
    
    return None


def get_model(architecture_parameters, standard_compile=False):
    """ This is the thing! """
    
    # check correctness of the parameters
    _check_architecture_parameters(architecture_parameters)
    
    # initialize input layer
    inputs = Input(shape=architecture_parameters["input_shape"])
    
    # speeds up the trainging, source unknown (?), trust me.
    x = BatchNormalization(axis=2, name="freq_axis_batchnorm")(inputs)

    "CONVOLUTIONAL BLOCKS"
    for idx in range(len(architecture_parameters["conv_blocks"]["n_filters"])):
        x = ConvolutionalBlock(architecture_parameters["conv_blocks"]["n_filters"][idx],
                               architecture_parameters["conv_blocks"]["kernel_sizes"][idx],
                               architecture_parameters["conv_blocks"]["normalization_axes"][idx],
                               architecture_parameters["conv_blocks"]["pool_sizes"][idx],
                               architecture_parameters["conv_blocks"]["avgpool_flags"][idx],
                               idx==len(architecture_parameters["conv_blocks"]["n_filters"])-1, # check if is_last_conv_block
                               str(idx+1))(x)
    
    "RECURRENT BLOCKS"
    if architecture_parameters["n_rec_units"] is not None:
        x = RecurrentBlock(architecture_parameters["n_rec_units"],"1")(x)
    
    "CLASSIFIER"
    if architecture_parameters["output_type"] == "softmax":

        x = SoftmaxClassifier(architecture_parameters["last_dropout_prob"], 
                              "vector", 
                              n_classes = len(list(architecture_parameters["initial_biases"])))(x)
        
        model = Model(inputs=inputs, outputs=x, 
                      name=architecture_parameters["model_name"])
        
        if standard_compile:
            loss = {output_name: "categorical_crossentropy" for output_name in  model.output_names}

            metrics = {output_name:[tf.keras.metrics.CategoricalAccuracy(name="accuracy")] 
                           for output_name in  model.output_names}
            
            
    elif architecture_parameters["output_type"] == "sigmoid":

        x = SigmoidClassifier(architecture_parameters["last_dropout_prob"], 
                              "vector", 
                              n_classes = len(list(architecture_parameters["initial_biases"])))(x)
        
        model = Model(inputs=inputs, outputs=x, 
                      name=architecture_parameters["model_name"])
        
        if standard_compile:
            loss = {output_name: "binary_crossentropy" for output_name in  model.output_names}

            metrics = {output_name:[tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall'),
                                    tf.keras.metrics.AUC(name='auc')] 
                           for output_name in  model.output_names}

        
    elif architecture_parameters["output_type"] == "split":
        outputs = []
        for category in sorted(list(architecture_parameters["initial_biases"])):
            outputs.append(BinaryClassifier(architecture_parameters["last_dropout_prob"],
                                            category,
                                            architecture_parameters["initial_biases"][category])(x))
            
        model = Model(inputs=inputs, outputs=outputs, 
                      name=architecture_parameters["model_name"])

        if standard_compile:
            loss = {output_name: "binary_crossentropy" for output_name in  model.output_names}

            metrics = {output_name:[tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall'),
                                    tf.keras.metrics.AUC(name='auc')] 
                           for output_name in  model.output_names}
            


    if standard_compile:
        initial_lr = 1e-3
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        print("Initial learning rate = %f\n"%initial_lr)
        
    return model



