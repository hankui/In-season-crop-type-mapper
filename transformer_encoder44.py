# Dec 29, 2020
# model builder
# import transformer_encoder44

# refer to
# https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# https://www.tensorflow.org/tutorials/quickstart/advanced

## Provides access to mathematical functions (e.g., trigonometric, logarithmic functions).
import math 
## NumPy is used for numerical computations, such as handling arrays, performing matrix operations, and generating random numbers.
import numpy as np 
## Provides the ability to log messages for debugging or tracking the progress of a program.
import logging 
## Pandas is a popular data manipulation and analysis library, often used to handle structured data, such as data frames for tabular data.
import pandas as pd 

## The following parts primarily related to TensorFlow and Keras, which are widely used libraries for building and training deep learning models.
## TensorFlow is an open-source platform for machine learning, especially deep learning. 
## It provides various tools for building and training neural networks.
import tensorflow as tf                      
## Keras Backend provides lower-level functions to work with tensors and various operations in TensorFlow.
## These include mathematical operations, working with neural network layers, and manipulating models at a fundamental level. 
from keras import backend as K                               

## The following imports are individual layers and utilities commonly used to build neural networks.
## The Sequential model in Keras is a linear stack of layers. 
## It's the simplest way to build a neural network in Keras, where each layer is added sequentially, one after the other.
from tensorflow.keras import Sequential 
## Dense: A fully connected neural network layer where every input is connected to every output.
## Concatenate: A layer that merges two or more input layers by concatenating their outputs along a specified axis.
from tensorflow.keras.layers import Dense, Concatenate 
## BatchNormalization: Normalizes the activations of the previous layer for each batch, improving the training speed and stability.
## LayerNormalization: Similar to batch normalization, but normalizes across features within a layer rather than across batches.
from tensorflow.keras.layers import BatchNormalization,LayerNormalization 
## LSTM (Long Short-Term Memory): A type of recurrent neural network (RNN) layer designed to learn long-term dependencies. 
## Commonly used for sequential data like time series or text.
## GRU (Gated Recurrent Unit): Another type of RNN, similar to LSTM but simpler. 
   ## It is often used in time series or natural language processing tasks.
## Bidirectional: Wraps an RNN (like LSTM or GRU) to run both forwards and backwards in time, 
   ## helping capture dependencies from both directions in a sequence.
from tensorflow.keras.layers import LSTM,GRU,Bidirectional
## Dropout: A regularization technique that randomly "drops out" neurons during training to prevent overfitting.
## Softmax: An activation function that converts raw scores into probabilities, often used in the final layer of a classification model.
from tensorflow.keras.layers import Dropout,Softmax
## MaxPooling1D: A pooling layer that downsamples the input by taking the maximum value over a fixed-size window along the 1D input.
## AveragePooling1D: Similar to max pooling, but takes the average value instead of the maximum.
## Conv1D: A 1D convolutional layer that applies a filter to sequences, commonly used for time series or text data.
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D,Conv1D
## Masking: A layer used to skip over certain time steps in sequences (like padding in sequences of different lengths).
## Embedding: A layer that converts integer-encoded categorical data (like words or tokens) into dense vector representations (embeddings).
from tensorflow.keras.layers import Masking,Embedding
## SimpleRNN: A basic recurrent neural network (RNN) layer.
## Attention: A general attention mechanism that allows the model to focus on specific parts of the input when making predictions.
## AdditiveAttention: A specific attention mechanism that calculates attention weights by adding the inputs, 
   ## often used in sequence-to-sequence models.
## TimeDistributed: Applies a layer (like Dense or Conv) to every time step of a sequence independently.
## MultiHeadAttention: Implements the multi-head attention mechanism, commonly used in Transformer models, 
   ## which allows the model to focus on different parts of the input sequence simultaneously.
from tensorflow.keras.layers import SimpleRNN, Attention, AdditiveAttention, TimeDistributed, MultiHeadAttention
## Input: Specifies the input shape and type for a model.
## Model: The general model class in Keras, allowing for more flexible model architectures compared to Sequential. 
   ## It can handle models with multiple inputs or outputs.
from tensorflow.keras import Input, Model
# import keras_layers.layers as customized 

# N_times = 14
# N_feature = 14
# N_outputs = 8

##*************************************** point_wise_feed_forward_network *********************************
# d_model: The dimensionality of the model, i.e., the number of units (neurons) in the final layer.
# dff: The dimensionality of the hidden layer, or the number of units in the intermediate layer.
# reg=None: This optional argument allows the addition of a regularization technique (like L2 regularization) to prevent overfitting. 
  # None means no regularization by default.
## This function defines a two-layer feed-forward neural network that applies a point-wise transformation to every time step in a sequence. 
## The first layer expands the dimensionality (dff), and the second layer projects it back to d_model. 
## The optional reg argument allows for regularization, which helps prevent overfitting during training. 
## This structure is typical in Transformer models after the attention layers.
def point_wise_feed_forward_network(d_model, dff, reg=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', kernel_regularizer=reg),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, kernel_regularizer=reg)  # (batch_size, seq_len, d_model)
    ])

##*****************************check test_mask.py to see why this is like adding two new axises*************************************************
## The create_padding_mask function creates a mask that identifies the padding in sequences. 
## It converts the positions corresponding to non-padding elements into 1.0 and padding elements into 0.0. 
## This mask is reshaped to (batch_size, 1, 1, seq_len) so it can be easily used in attention-based models like Transformers.
def create_padding_mask(inputs, mask_value=0):
    seq = tf.cast(tf.math.not_equal(inputs[:, :, 0], mask_value), tf.float32) # (batch_size, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len) 

##***************************** create_padding_mask_any *************************************************
## check test_mask.py to see why this is like adding two new axises
## This function, create_padding_mask_any, generates a padding mask similar to the previous example 
## but adds more flexibility by checking if any feature in the input tensor is not equal to the mask_value, 
## rather than just the first feature. This is useful when the padding can exist in different features, not just the first one.
def create_padding_mask_any(inputs0, mask_value=0):
    # seq = tf.cast(tf.math.not_equal(inputs[:, :, 0], mask_value), tf.float32)
    seq = tf.cast(tf.math.reduce_any(tf.math.not_equal(inputs0, mask_value),axis=2), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :],seq[:, :, tf.newaxis]  # (batch_size, 1, 1, seq_len)

## ****************************************** lat/lon encoder ******************************************
## This function, embedding_lon_lat, creates a neural network model for embedding geographical coordinates (longitude and latitude) 
## using a configurable number of layers and units. The function returns a Keras Sequential model made up of fully connected (Dense) layers, 
## layer normalization, and optional dropout.
## layer_n=1: Number of layers in the network.
## units=64: Number of units (neurons) in each Dense layer by default.
## drop=0.1: Dropout rate, which helps with regularization to prevent overfitting.
## reg=None: Optional regularization (such as L2 regularization) to prevent overfitting.
def embedding_lon_lat(layer_n=1, units=64, drop=0.1, reg=None):
    all_layers = list()
    unit_small = units
    # unit_small = units//2 ## tested for v8_75 and v8_76 
    for i in range(layer_n):
        if i==layer_n-1:
            unit_small = units
        
        all_layers.append(Dense(unit_small, activation='relu', kernel_regularizer=reg))
        all_layers.append(LayerNormalization(epsilon=1e-6))
        if drop>0:
            all_layers.append(Dropout(drop))
    
    return tf.keras.Sequential(all_layers)

## *********************** position embeddings (inform relative positions) ******************************
## The purpose of this layer is to create learnable relative position embeddings for sequences. 
## In Transformer-like models, position embeddings are crucial for giving the model information about the order or 
## relative position of tokens in a sequence.
class AddLearnableEmbedding(tf.keras.layers.Layer): # BNAD_N = 7
    def __init__(self, d_model, seq_length):
        super(AddLearnableEmbedding, self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        # Initialize learnable relative position (lrp) bias
        self.relative_position_bias = self.add_weight(
            "relative_position_bias",                 
            shape=[2 * seq_length - 1, self.d_model], ## i.e., from -(seq_length - 1) to +(seq_length - 1)
            ### d_model represents the embedding dimension, or the number of units in each relative position embedding.
            initializer="random_normal", # lrp bias with random values sampled from a normal distribution.              
            trainable=True
        )
    def call(self, inputs):
        return self.relative_position_bias

## ******* position embeddings (inform relative positions) from either learnable or non-learnable *******
## The function create_relative_position_bias can generate either learnable or non-learnable relative positional embeddings, 
## depending on whether is_learn is set to True or False. This flexibility allows the model to either learn relative positions 
## during training or use pre-defined embeddings.
def create_relative_position_bias(seq_length, d_model, is_learn=False): #get this from ChatGPT
    """
    Create learnable relative positional encodings.
    """
    # relative_positions = tf.range(-seq_length + 1, seq_length)
    if is_learn:
        print ("AddLearnableEmbedding")
        relative_positional_encoding = AddLearnableEmbedding(d_model,2 * seq_length - 1)(8)
    else:
        # This creates a range of relative positions starting from 0 to 2 * seq_length - 2 (i.e., 0, 1, ..., 2 * seq_length - 2), 
        # resulting in 2 * seq_length - 1 relative positions.
        relative_positions = tf.range(0, 2 * seq_length - 1)
        embedding_layer = tf.keras.layers.Embedding(input_dim=2 * seq_length - 1, output_dim=d_model)
        relative_positional_encoding = embedding_layer(relative_positions)
    
    # relative_positional_encoding = tf.reshape(relative_positional_encoding, [2 * seq_length - 1, d_model])
    return relative_positional_encoding


#=================== Example usage ## Use a smaller length for simplicity ========================================
# batch_size = 2   ## batch size
# seq_length = 10  ## sequence length
# d_model = 16     ## layer depth

# Create dummy queries, keys, values, and position array
# queries = tf.random.uniform((batch_size, seq_length, d_model))
# keys = tf.random.uniform((batch_size, seq_length, d_model))
# values = tf.random.uniform((batch_size, seq_length, d_model))
# position_array = tf.constant([[1, 2, 3, -9999, 5, 6, -9999, 8, 9, 10], [1, -9999, 3, 4, 5, 6, 7, 8, -9999, 10]], dtype=tf.int32)

# Create relative positional encoding
# relative_position_bias = create_relative_position_bias(seq_length, d_model)

# Compute self-attention with relative position bias
# output = self_attention_with_relative_position_bias(queries, keys, values, position_array, relative_position_bias)
# print(output)
#=================== Example usage ## Use a smaller length for simplicity ========================================

import multi_head_from_ChatGPT
import importlib
importlib.reload(multi_head_from_ChatGPT)

## ****************************************** Transformer block **************************************************
## This transformer_block function is a flexible and customizable implementation of a Transformer block, 
## allowing for the integration of various features and configurations commonly used in Transformer-based models.

## transformer_block
# xL,xL,padding_maskL,units,n_head,drop=drop,is_batch=True,is_att_score=False;
# relative=is_day_input>5
# x_doy=x_doy[:,:MAX_LANDSAT,0]
# queryx = xL
# x = xL 

## queryx: The query tensor for attention, often the same as the input tensor 
## x: The input tensor for the attention mechanism.
## padding_mask: A mask to prevent attention to certain positions (e.g., padding tokens).
## units: The dimensionality of the model's embeddings.
## n_head: The number of attention heads.
## reg: Optional regularization to apply to the attention layers.
## drop: Dropout rate to apply to the attention and feed-forward layers.
## is_batch: Boolean to decide if layer normalization should be applied after the residual connection.
## is_att_score: Boolean to decide if attention scores should be returned.
## relative: Boolean to use relative positional encoding.
## x_doy: Additional input for relative position encoding if used (e.g., day of the year).
def transformer_block (queryx, x, padding_mask, units, n_head, reg=None, drop=0.1, is_batch=True, is_att_score=True, relative=False, x_doy=None):
    if relative:
        attention_layer = multi_head_from_ChatGPT.MultiHeadAttentionWithRelativePositionBias(units, n_head, 366)
        attn_output  = attention_layer(queryx, x, x, x_doy, padding_mask)
    else:
        if is_att_score:
            attn_output, attn4 = MultiHeadAttention(key_dim=units//n_head, num_heads=n_head, kernel_regularizer=reg)(query=queryx, value=x, key=x,
                return_attention_scores=is_att_score, attention_mask=padding_mask)
        else:
            attn_output        = MultiHeadAttention(key_dim=units//n_head, num_heads=n_head, kernel_regularizer=reg)(query=queryx, value=x, key=x,
                return_attention_scores=is_att_score, attention_mask=padding_mask)
    
    if drop > 0: ## Applies dropout to the attention output to help with regularization if drop > 0.
        attn_output = Dropout(drop)(attn_output)
    
    out1 = queryx + attn_output ## Adds the attention output to the query tensor (residual connection).
    if is_batch == True:
        out1 = LayerNormalization(epsilon=1e-6)(out1)
        ## Applies layer normalization if is_batch=True. 
        ## This normalizes the combined tensor to stabilize training and improve convergence.
        
    ffn_output = point_wise_feed_forward_network(units, units * 4, reg=reg)(out1)
    ## Applies a point-wise feed-forward network to out1. 
    ## The FFN typically consists of two dense layers with a ReLU activation in between.
    if drop > 0:
        ffn_output = Dropout(drop)(ffn_output) ## Applies dropout to the FFN output if drop > 0.

    out2 = out1 + ffn_output ## Adds the FFN output to the output of the attention layer (residual connection).
    if is_batch == True:
        out2 = LayerNormalization(epsilon=1e-6)(out2) ## Applies layer normalization if is_batch=True.
    
    ## Updates x with the final output of the Transformer block.
    ## Returns both the output and the attention scores if is_att_score=True. Otherwise, it returns only the output.
    x = out2
    if is_att_score:
        return x, attn4 
    else:
        return x

# ************************************************************************************************************
# ****************************** input is HLS  ***************************************************************
# ************************************************************************************************************
# layern1=3; layern2=3; units=64; n_head=4; drop=0.1; L2=0; concat=4
# is_day_input=1; is_xy=False; is_reflectance=False; active="linear"; is_sensor=True; n_out = 50
# inputs = trainx[:2,:,1:] # no filled data
# https://www.tensorflow.org/text/tutorials/transformer

from config import MAX_LANDSAT,MAX_SENTINEL2,L8_fields,S2_fields,YEARS_DATA
L8_bands_n = len(L8_fields)//MAX_LANDSAT   
S2_bands_n = len(S2_fields)//MAX_SENTINEL2 
mask_value=-9999.0;
def get_transformer_reflectance_cdl(MAX_LANDSAT=14, MAX_SENTINEL2=28, L8_bands_n=2, S2_bands_n=4, YEARS_DATA=3, n_out=9, 
    layern1=3, layern2=3, units=64, n_head=4, drop=0.1, L2=0, 
    is_day_input=1, is_sensor=False, is_xy=False, is_reflectance=True, active="linear", concat=1):
    """using AveragePooling1D with mask"""
    is_batch=True, 
    reg = None
    if L2>0:
        reg = tf.keras.regularizers.l2(l=L2)
    
    ## *******************
    # reflectance
    LENGTH = MAX_LANDSAT+MAX_SENTINEL2
    inputs = Input(shape=(LENGTH*YEARS_DATA, S2_bands_n+2*is_xy,))       
    ## *******************
    
    # positional -> need to change positional to day of year
    x_doy = inputs[:,:,:1]
    mask_multi2 = tf.cast(tf.math.not_equal(x_doy,mask_value), tf.float32)
    if is_day_input==1: ## day of year as position 
        DOY_ARRAY = np.array(range(366))
        x_doy = x_doy*mask_multi2
        # x_doy = (x_doy - DOY_ARRAY.mean() ) * mask_multi2/ DOY_ARRAY.std() 
        x_doy = Dense(units, use_bias=False)(x_doy)
    elif is_day_input==2:
        print ("2nd method doy encoder using positional encoder function sin and cos")
        pos_enc = positional_encoding(366,units)
        ## SHIT I did this using ChatGPT 4 on Jun 12, 2024
        ## I input "In tensorflow, I have a tensor x_doy with shape [a,b], and another tensor pos_enc with shape [366,d], 
        ## the x_doy can be any values between 1 to 366 or filled value -9999. write a piece of code to generate another array with shape [a,b,d], 
        ## where the values are set as if any value in [a,b] if not filled (e.g., n), take the nth d vector from pos_enc as the [a,b,:] value, 
        ## if filled, set [a,b,:] as filled "
        # Create a mask for filled values (-9999)
        mask = tf.not_equal(x_doy, -9999)
        # Use tf.gather to index pos_enc with x_doy
        # gathered = tf.gather(pos_enc, tf.clip_by_value(x_doy, 0, 365))
        gathered = tf.gather(pos_enc, tf.cast(tf.clip_by_value(x_doy[:,:,0]-1, clip_value_min=0, clip_value_max=365),tf.int16))
        # Create the filled tensor with the same shape as the gathered tensor
        # filled_tensor = tf.fill([a, b, d], -9999.0)
        filled_tensor = tf.fill([tf.shape(x_doy)[0],LENGTH,units], 0.0)
        # Use the mask to conditionally combine the tensors
        # result = tf.where(tf.expand_dims(mask, axis=-1), gathered, filled_tensor)
        x_doy = tf.where(mask, gathered, filled_tensor)
        # print(result)
        # tf.cond(condition, true_fn, false_fn)
    elif is_day_input==3:
        print ("3rd method doy encoder with sin and cos")
        n_times = 366
        xp1 = tf.cos((x_doy+0.5)/n_times*np.pi) * mask_multi2
        xp2  = tf.sin((x_doy+0.5)/n_times*np.pi) * mask_multi2
        xpp = tf.concat ([xp1,xp2],axis=2)    
        x_doy = Dense(units, use_bias=False)(xpp)
    elif is_day_input==4:
        print ("4th method doy encoder using positional embedding function (shit, embedding is not learned) ")
        pos_enc = create_relative_position_bias(366,units)
        mask = tf.not_equal(x_doy, -9999)
        gathered = tf.gather(pos_enc, tf.cast(tf.clip_by_value(x_doy[:,:,0]-1, clip_value_min=0, clip_value_max=365),tf.int32))
        filled_tensor = tf.fill([tf.shape(x_doy)[0],LENGTH,units], 0.0)
        x_doy = tf.where(mask, gathered, filled_tensor)
    elif is_day_input==5:
        print ("5th method doy encoder using learnable positional embedding function ")
        # pos_enc = create_relative_position_bias(366,units,is_learn=True) # does not implement learnable 
        pos_enc = AddLearnableEmbedding(units,366)(x_doy)
        mask = tf.not_equal(x_doy, -9999)
        gathered = tf.gather(pos_enc, tf.cast(tf.clip_by_value(x_doy[:,:,0]-1, clip_value_min=0, clip_value_max=365),tf.int32))
        filled_tensor = tf.fill([tf.shape(x_doy)[0],LENGTH,units], 0.0)
        x_doy = tf.where(mask, gathered, filled_tensor)
        # print(result)
        # tf.cond(condition, true_fn, false_fn)    
    else:
        print ("6th method doy encoder using RELATIVE learnable positional embedding function ")
        
    
    ## sensor encoder 
    sensor_embed = Embedding(2, units)
    
    ## Landsat 
    # xL = inputs[:,:MAX_LANDSAT,1:L8_bands_n]
    for yi in range(YEARS_DATA):
        if yi==0:
            xL = inputs[:,:MAX_LANDSAT,1:L8_bands_n]
            xL_doy = x_doy[:,:MAX_LANDSAT,]
        else:
            xL = tf.concat([xL, inputs[:,(LENGTH*yi):(LENGTH*yi+MAX_LANDSAT),1:L8_bands_n] ], axis=1)
            xL_doy = tf.concat([xL_doy, x_doy[:,(LENGTH*yi):(LENGTH*yi+MAX_LANDSAT),] ], axis=1)
    
    # xL = tf.concat([inputs[:,:MAX_LANDSAT,1:L8_bands_n],\
                    # inputs[:,LENGTH:(LENGTH+MAX_LANDSAT),1:L8_bands_n],\
                    # inputs[:,LENGTH*2:(LENGTH*2+MAX_LANDSAT),1:L8_bands_n]], axis=1)
    xLtemp = xL 
    mask_multi0 = tf.cast(tf.math.not_equal(xL,mask_value), tf.float32)
    xL = xL * mask_multi0
    xL = Dense(units, use_bias=False)(xL)
    if is_day_input<=5:
        xL = xL+xL_doy
        # xL = xL+tf.concat([x_doy[:,:MAX_LANDSAT,],\
                           # x_doy[:,LENGTH:(LENGTH+MAX_LANDSAT),], 
                           # x_doy[:,LENGTH*2:(LENGTH*2+MAX_LANDSAT),]], axis=1)
    
    if is_sensor:
        print("sensor Embedding is used");
        xL = xL+sensor_embed(mask_multi0[:,:,0]<-1)
    
    ## Sentinel-2  
    for yi in range(YEARS_DATA):
        if yi==0:
            xS = inputs[:,MAX_LANDSAT:LENGTH,1:S2_bands_n]
            xS_doy = x_doy[:,MAX_LANDSAT:LENGTH,]
        else:
            xS = tf.concat([xS, inputs[:,(yi*LENGTH+MAX_LANDSAT):(yi*LENGTH+LENGTH),1:S2_bands_n] ], axis=1)
            xS_doy = tf.concat([xS_doy, x_doy[:,(yi*LENGTH+MAX_LANDSAT):(yi*LENGTH+LENGTH),] ], axis=1)

    # xS = tf.concat([inputs[:,MAX_LANDSAT:LENGTH,1:S2_bands_n], \
                    # inputs[:,(LENGTH+MAX_LANDSAT):(2*LENGTH),1:S2_bands_n], \
                    # inputs[:,(2*LENGTH+MAX_LANDSAT):,1:S2_bands_n]], axis=1)
    xStemp = xS 
    mask_multi0 = tf.cast(tf.math.not_equal(xS,mask_value), tf.float32)
    xS = xS * mask_multi0
    xS = Dense(units, use_bias=False)(xS)
    if is_day_input<=5:
        xS = xS+xS_doy
        # xS = xS+tf.concat([x_doy[:,MAX_LANDSAT:LENGTH,],\
                           # x_doy[:,(LENGTH+MAX_LANDSAT):(2*LENGTH),],\
                           # x_doy[:,(2*LENGTH+MAX_LANDSAT):,]], axis=1)
    
    if is_sensor:
        xS = xS+sensor_embed(mask_multi0[:,:,0]>-1)
    
    ## *******************
    # lat lon 
    if is_xy:
        xxy = inputs[:,:,S2_bands_n:(S2_bands_n+2)]
        mask_multi2 = tf.cast(tf.math.not_equal(xxy,mask_value), tf.float32)
        xxy = xxy * mask_multi2
        xxy = Dense(units, use_bias=False)(xxy)
        xL = xL+xxy[:,:MAX_LANDSAT,]
        xS = xS+xxy[:,MAX_LANDSAT:,]
        # print(xxy.shape)
        # print (BNAD_N+is_day_input+is_sensor)
        # print (BNAD_N)
        # print (is_day_input)
        # print (is_sensor)
    
    ## *******************
    ## start to encoder and decoder    
    # padding_mask, padding_mask3d = create_padding_mask_any(inputs0=inputs[:,:,:], mask_value=mask_value)
    padding_maskA, _ = create_padding_mask_any(inputs0=inputs[:,:,1:L8_bands_n], mask_value=mask_value) # fix this bug on Feb 18 2023, mask only applied for 6 bands but not for all data
    padding_maskL, _ = create_padding_mask_any(inputs0=xLtemp                  , mask_value=mask_value) # fix this bug on Feb 18 2023, mask only applied for 6 bands but not for all data
    padding_maskS, _ = create_padding_mask_any(inputs0=xStemp                  , mask_value=mask_value) # fix this bug on Feb 18 2023, mask only applied for 6 bands but not for all data
    # if is_mask:                                                              
        # print ("Masking is used")
    # else:
        # print ("Masking is *NOT* used")
        
    # encoder
    for i in range(layern1):
        xL = transformer_block (xL,xL,padding_maskL,units,n_head,drop=drop,is_batch=True,is_att_score=False,relative=is_day_input>5, x_doy=x_doy[:,:MAX_LANDSAT,0]);
        xS = transformer_block (xS,xS,padding_maskS,units,n_head,drop=drop,is_batch=True,is_att_score=False,relative=is_day_input>5, x_doy=x_doy[:,MAX_LANDSAT:,0]);
    
    if concat==1:
        x = tf.concat([xL, xS],axis=1) 
    elif concat==2:
        x = tf.concat([xL, xS],axis=2) # 6.22
        x = Dense(units)(x) # 6.21, 6.2 is Dense only 
        # if drop > 0:
            # x = Dropout(drop)(x)
        
        # padding_maskA = padding_maskL+padding_maskS
        padding_maskA = tf.cast(tf.logical_or(tf.cast(padding_maskL, tf.bool), tf.cast(padding_maskS, tf.bool)), tf.float32)
    elif concat==3:
        x = xL + xS # 6.22
        padding_maskA = tf.cast(tf.logical_or(tf.cast(padding_maskL, tf.bool), tf.cast(padding_maskS, tf.bool)), tf.float32)
    elif concat==4:
        x = tf.concat([xL, xS],axis=2) # 6.22
        # x = Dense(units)(x) # 6.21, 6.2 is Dense only 
        # if drop > 0:
            # x = Dropout(drop)(x)
        
        # padding_maskA = padding_maskL+padding_maskS
        padding_maskA = tf.cast(tf.logical_or(tf.cast(padding_maskL, tf.bool), tf.cast(padding_maskS, tf.bool)), tf.float32)
    
    for i in range(layern2):
        if concat<=3:
            x = transformer_block (x,x,padding_maskA,units,n_head,drop=drop,is_batch=True,is_att_score=False,relative=is_day_input>5, x_doy=x_doy[:,:,0]);
        else:
            x = transformer_block (x,x,padding_maskA,units*2,n_head,drop=drop,is_batch=True,is_att_score=False,relative=is_day_input>5, x_doy=x_doy[:,:,0]);
    
    ## *******************
    ## start to output 
    if is_reflectance==True:
        if concat==1:
            if active=="linear":
                output1 = Dense(S2_bands_n-1, kernel_regularizer=reg)(x[:,:MAX_LANDSAT,:])
                output2 = Dense(S2_bands_n-1, kernel_regularizer=reg)(x[:,MAX_LANDSAT:,:])
            else:
                output1 = Dense(S2_bands_n-1, activation="sigmoid", kernel_regularizer=reg)(x[:,:MAX_LANDSAT,:])
                output2 = Dense(S2_bands_n-1, activation="sigmoid", kernel_regularizer=reg)(x[:,MAX_LANDSAT:,:])
        else:
            if active=="linear":
                output1 = Dense(S2_bands_n-1, kernel_regularizer=reg)(x)
                output2 = Dense(S2_bands_n-1, kernel_regularizer=reg)(x)
            else:
                output1 = Dense(S2_bands_n-1, activation="sigmoid", kernel_regularizer=reg)(x)
                output2 = Dense(S2_bands_n-1, activation="sigmoid", kernel_regularizer=reg)(x)
        
        output  = tf.concat([output1, output2],axis=1)  
    
    else:
        padding_mask3d = padding_maskA[:,0,0,:][:,:,tf.newaxis]
        enc_output2 = tf.math.multiply(x,padding_mask3d)
        enc_output = tf.math.divide(K.sum(enc_output2, axis=1), K.sum(padding_mask3d, axis=1))   
        output = Dense(n_out, kernel_regularizer=reg)(enc_output)
    
    model = Model(inputs, output)    
    return model



# *****************************************************************************************************************************************************
# ****************************** input is daily and 3D  ***************************************************************
# *****************************************************************************************************************************************************
# layern=3; units=64; n_times=23; n_feature=6; n_head=4; is_batch=True; drop=0.1; mask_value = -9999.0;  active="exponential"; n_out=9
# inputs = input_images_train_norm3[:2,:,:] # no filled data
# https://www.tensorflow.org/text/tutorials/transformer
def get_transformer_cls (n_times=14, n_feature=2, n_out=9, layern=3, units=128, n_head=4, drop=0.1, is_batch=True, mask_value=-9999.0, active="softmax", 
        L2=0,is_day_input=False, is_sensor=False, is_sensor_embed=False, is_xy=False, is_reflectance=False):
    """borrow BERT CLS token"""
    
    inputs = Input(shape=(n_times, n_feature+1+is_sensor+is_xy*2,))       
    reg = None
    if L2>0:
        reg = tf.keras.regularizers.l2(l=L2)
    
    ## *******************
    # reflectance
    x0 = inputs[:,:,:6]
    mask_multi0 = tf.cast(tf.math.not_equal(x0,mask_value), tf.float32)
    x0 = x0 * mask_multi0
    
    ## *******************
    # positional -> need to change positional to day of year
    if is_day_input: ## day of year as position 
        xpp = inputs[:,:,6:7]
        mask_multi2 = tf.cast(tf.math.not_equal(xpp,mask_value), tf.float32)
        xpp = xpp * mask_multi2/366
    else: ## position implied in data position 
        xp = np.arange(n_times)[:, np.newaxis] / n_times  # (seq, 1)
        xpp = K.ones_like(inputs[:, :, :1]) * xp        
    
    ## *******************
    # sensor
    if is_sensor:
        xse = inputs[:,:,7:8]
        xse = xse * mask_multi2

    ## *******************
    # lat lon 
    if is_xy:
        xxy = inputs[:,:,8:10]
        xxy = xxy * mask_multi2
    
    ## *******************
    ## embedding inputs 
    embedding_x = Dense(units, use_bias=False)
    embedding_p = Dense(units, use_bias=False)
    embedding_s = Dense(units, use_bias=False)
    embedding_s_em = Embedding(4, units)
    # embedding_xy = embedding_lon_lat(layer_n=3, units=units, drop=0, reg=reg)
    embedding_xy = Dense(units, use_bias=False)
    if is_sensor and is_sensor_embed and is_xy:
        print ("Use Embedding in sensor encoder and use xy encoder ")
        xse_encoder = embedding_s_em(xse)
        x = embedding_x(x0) + embedding_p(xpp) + xse_encoder[:,:,0,:] + embedding_xy(xxy)
    elif is_sensor and is_sensor_embed:
        print ("Use Embedding in sensor encoder")
        xse_encoder = embedding_s_em(xse)
        x = embedding_x(x0) + embedding_p(xpp) + xse_encoder[:,:,0,:]
    elif is_sensor:
        print ("Use Dense in sensor encoder")
        x = embedding_x(x0) + embedding_p(xpp) + embedding_s(xse)
    else:
        x = embedding_x(x0) + embedding_p(xpp)
    
    
    # if drop>0:## tested work not well
    # x=Dropout(drop)(x)
    ## *******************
    ## add token bert method ## add a cls_token
    cls_token = K.zeros_like(x[:, :1, :])
    x2 = tf.concat([x, cls_token],axis=1)
    x = x2
    cls_token_mask = tf.cast(tf.math.not_equal(K.zeros_like(x[:, :1, :1]), 1), tf.float32)
    cls_token_mask = cls_token_mask[:, tf.newaxis, :, :]  # (batch_size, 1, 1, seq_len)
    padding_mask = create_padding_mask(inputs=inputs[:,:,:6], mask_value=mask_value)
    padding_mask = tf.concat([padding_mask, cls_token_mask],axis=3)
    # encoder
    for i in range(layern):
        # temp_mha = MultiHeadAttention(key_dim=units//n_head, num_heads=n_head)
        # attn_output, attn4 = temp_mha(query=x,value=x,key=x,return_attention_scores=True, attention_mask=padding_mask)
        attn_output, attn4 = MultiHeadAttention(key_dim=units // n_head, num_heads=n_head, kernel_regularizer=reg)(query=x, value=x, key=x,
                                                                                           return_attention_scores=True,
                                                                                           attention_mask=padding_mask)
        if drop > 0:
            attn_output = Dropout(drop)(attn_output)
        
        out1 = x + attn_output
        if is_batch == True:
            out1 = LayerNormalization(epsilon=1e-6)(out1)
        
        ffn_output = point_wise_feed_forward_network(units, units * 4, reg=reg)(out1)
        if drop > 0:
            ffn_output = Dropout(drop)(ffn_output)
        
        out2 = out1 + ffn_output
        if is_batch == True:
            out2 = LayerNormalization(epsilon=1e-6)(out2)
        
        x = out2
    
    ## *******************
    ## start to output 
    if is_reflectance:
        output = Dense(6, activation="sigmoid", kernel_regularizer=reg)(x[:,:n_times,:])
    else:
        # enc_output2 = tf.math.multiply(x,padding_mask3d)
        # enc_output = tf.math.divide(K.sum(enc_output2, axis=1), K.sum(padding_mask3d, axis=1))   
        # output = Dense(n_out, activation=active, kernel_regularizer=reg)(enc_output)
        enc_output = x[:,n_times,:]
        output = Dense(n_out, activation=active, kernel_regularizer=reg)(enc_output)
    
    model = Model(inputs, output)
    
    return model



# layern=3; units=64; n_times=23; n_feature=6; n_head=4; is_batch=True; drop=0.1; mask_value = -9999.0;  active="exponential"; n_out=9
# inputs = input_images_train_norm3[:2,:,:] # no filled data
# https://www.freecodecamp.org/news/the-ultimate-guide-to-recurrent-neural-networks-in-python/
def model_Bidirectional_or_GRU_mask(layern=3, units=64, n_times=40, n_feature=1, n_out=9, ReLU=False, drop=0, sequence=False, is_batch=True, mask_value=-9999.0, active="softmax"):
    # rnn = Sequential()  
    
    inputs = Input(shape=(n_times, n_feature,))       
    ## get the elements number in each unit
    if isinstance(units, int):
        units_array = np.repeat(units,100)
    else:
        units_array = np.array(units)
    
    ## add position encoder 
    embedding_x = Dense(units)
    embedding_p = Dense(units)
    
    ## *******************
    # positional
    b = K.ones_like(inputs[:, :, :1])
    xp = np.arange(n_times)[:, np.newaxis] / n_times  # (seq, 1)
    xpp = b * xp
    # x = embedding_x(x0) + embedding_p(xpp)
    
    ## a 
    gn = tf.keras.initializers.GlorotNormal()
    he = tf.keras.initializers.HeNormal()
    gu = tf.keras.initializers.GlorotNormal() if not ReLU else he
    active_gru = 'tanh' if not ReLU else "relu"
    #Adding our first LSTM layer
    if not math.isnan(mask_value): 
        x = Masking(mask_value=mask_value, input_shape=(n_times, n_feature))(inputs)
        # x = embedding_x(x)+embedding_p(xpp)
        x = Bidirectional(LSTM(units=units_array[0], kernel_initializer=gn, return_sequences = True),merge_mode='sum')(x)
    else: 
        x = Bidirectional(LSTM(units=units_array[0], kernel_initializer=gn, return_sequences = True, input_shape = (n_times, n_feature)),merge_mode='sum')(inputs)
    
    # is_batch==True and rnn.add(LayerNormalization())
    if is_batch==True:
        x = LayerNormalization()(x)
    
    #Perform some dropout regularization
    if drop>0:
        # rnn.add(Dropout(drop))
        x = Dropout(drop)(x)
    
    #Adding three more LSTM layers with dropout regularization
    # for i in [True, True, False]:
    for i in range(layern-1):
        return_sequences = True 
        return_sequences = True if i<(layern-2) else sequence
        # print(return_sequences)
        # rnn.add(GRU(units=units_array[i+1], activation=active_gru, kernel_initializer=gu, return_sequences=return_sequences))
        x = Bidirectional(LSTM(units=units_array[i+1], activation=active_gru, kernel_initializer=gn, return_sequences=return_sequences),merge_mode='sum')(x)
        # is_batch==True and rnn.add(LayerNormalization())
        if is_batch==True:
            x = LayerNormalization()(x)
        
        if drop>0:
            # rnn.add(Dropout(drop))
            x = Dropout(drop)(x)
    
    #Adding our output layer This is not helpful
    # x = Dense(units=units*4, activation='relu', kernel_initializer=gu)(x)
    # if is_batch==True:
        # x = LayerNormalization()(x)
    
    # if drop>0:
        # x = Dropout(drop)(x)
    
    # x = Dense(units=n_out, kernel_initializer=gu)(x)
    # if drop>0:
        # x = Dropout(drop)(x)    
    
    # output = Softmax()(x)
    output = Dense(units=n_out, activation=active, kernel_initializer=gu)(x)
    model = Model(inputs, output)    
    return model



# *****************************************************************************************************************************************************
# ****************************** input is daily and 3D with xy as seperate variables  ***************************************************************
# *****************************************************************************************************************************************************
# This is used to include xy positional encoder 
# This can cover get_transformer_new_att0 function 
# 
# layern=3; units=64; n_times=80; n_feature=6; n_head=4; is_batch=True; drop=0.1; mask_value = -9999.0;  active="exponential"; n_out=7
# L2=0; is_day_input=True; is_sensor=True; is_sensor_embed=True
# inputs = [input_images_train_norm3[:2,:,:],training_xy[:2,:]] # no filled data
# is_day_input=False; is_sensor=False; is_sensor_embed=False; xy_layer=1
# is_day_input=True; is_sensor=True; is_sensor_embed=True; xy_layer=1; is_reflectance=False

# https://www.tensorflow.org/text/tutorials/transformer
def get_transformer_new_att0_daily_withsensor_xy(n_times=14, n_feature=2, n_out=9, layern=3, units=128, n_head=4, drop=0.1, is_batch=True, mask_value=-9999.0, active="softmax", 
        L2=0,is_day_input=False, is_sensor=False, is_sensor_embed=False, xy_layer=1, is_reflectance=False):
    """using AveragePooling1D with mask"""
    inputs = [Input(shape=(n_times, n_feature+1+is_sensor,)), Input(shape=(2,))]
    # embedding_x = Dense(units)
    # embedding_p = Dense(units)
    reg = None
    if L2>0:
        reg = tf.keras.regularizers.l2(l=L2)
    
    ## *******************
    # positional -> need to change positional to day of year
    x0 = inputs[0][:,:,:6]
    mask_multi0 = tf.cast(tf.math.not_equal(x0,mask_value), tf.float32)
    x0 = x0 * mask_multi0
    
    ## *******************
    # positional
    if is_day_input:
        xpp = inputs[0][:,:,6:7]
        mask_multi2 = tf.cast(tf.math.not_equal(xpp,mask_value), tf.float32)
        xpp = xpp * mask_multi2/366
    else:
        # b = K.ones_like(inputs[:, :, :1])
        xp = np.arange(n_times)[:, np.newaxis] / n_times  # (seq, 1)
        xpp = K.ones_like(inputs[:, :, :1]) * xp        
    
    ## *******************
    # sensor
    if is_sensor:
        xse = inputs[0][:,:,7:8]
        xse = xse * mask_multi2
        
    ## *******************
    ## embedding inputs 
    embedding_x = Dense(units, use_bias=False)
    embedding_p = Dense(units, use_bias=False)
    embedding_s = Dense(units, use_bias=False)
    embedding_s_em = Embedding(4, units)
    if is_sensor and is_sensor_embed:
        print ("Use Embedding in sensor encoder")
        xse_encoder = embedding_s_em(xse)
        x = embedding_x(x0) + embedding_p(xpp) + xse_encoder[:,:,0,:]
    elif is_sensor:
        print ("Use Dense in sensor encoder")
        x = embedding_x(x0) + embedding_p(xpp) + embedding_s(xse)
    else:
        x = embedding_x(x0) + embedding_p(xpp)
    
    ## *******************
    ## start to encoder and decoder    
    padding_mask, padding_mask3d = create_padding_mask_any(inputs0=inputs[0][:,:,:6], mask_value=mask_value)
    # encoder
    for i in range(layern):
        attn_output, attn4 = MultiHeadAttention(key_dim=units // n_head, num_heads=n_head, kernel_regularizer=reg)(query=x, value=x, key=x,
                                                                                           return_attention_scores=True,
                                                                                           attention_mask=padding_mask)
        if drop > 0:
            attn_output = Dropout(drop)(attn_output)
        
        out1 = x + attn_output
        if is_batch == True:
            out1 = LayerNormalization(epsilon=1e-6)(out1)
        
        ffn_output = point_wise_feed_forward_network(units, units * 4, reg=reg)(out1)
        if drop > 0:
            ffn_output = Dropout(drop)(ffn_output)
        
        out2 = out1 + ffn_output
        if is_batch == True:
            out2 = LayerNormalization(epsilon=1e-6)(out2)
        
        x = out2
    
    ## *******************
    ## start to output 
    embed_lonlat = embedding_lon_lat(layer_n=xy_layer, units=units, drop=0, reg=reg)(inputs[1])
    if is_reflectance:
        enc_output = x + embed_lonlat[:,tf.newaxis,:]
        output = Dense(6, activation="sigmoid", kernel_regularizer=reg)(enc_output)
    else: 
        enc_output2 = tf.math.multiply(x,padding_mask3d)
        enc_output = tf.math.divide(K.sum(enc_output2, axis=1), K.sum(padding_mask3d, axis=1)) + embed_lonlat    
        output = Dense(n_out, activation=active, kernel_regularizer=reg)(enc_output)
    
    model = Model(inputs, output)    
    return model


## copied from tensorflow tutorials https://www.tensorflow.org/text/tutorials/transformer
def positional_encoding(length, depth):
    depth = depth // 2
    
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    
    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    
    return tf.cast(pos_encoding, dtype=tf.float32)


