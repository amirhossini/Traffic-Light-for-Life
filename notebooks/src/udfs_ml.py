import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras import backend

def conv1d(X, conv_layers=2, conv_dropout=0.3, 
           conv_poolsize=2, ls_conv_filters=[8,16], 
           ls_conv_kernels=[13,11], ls_conv_padding=['same','same','same'],
           ls_conv_actvf=['relu','relu','relu'],dense_layers=1, ls_dense_neurns=[16], 
           ls_dense_actvf=['relu'], out_neurns=1, out_actvf='sigmoid',
           loss_func='binary_crossentropy',optim='adam', eval_metric='acc',
           best_model_record='./models/best_model.hdf5'):
    
    backend.clear_session()
    inputs = Input(shape=(X.shape[1],X.shape[2]))
    
    # conv1d layers: 
    for iconv in range(conv_layers):
        if iconv == 0:
            conv = Conv1D(ls_conv_filters[iconv], ls_conv_kernels[iconv], padding=ls_conv_padding[iconv],                       
                          activation=ls_conv_actvf[iconv])(inputs)
        else:
            conv = Conv1D(ls_conv_filters[iconv], ls_conv_kernels[iconv], padding=ls_conv_padding[iconv],                       
                          activation=ls_conv_actvf[iconv])(conv)
            
        conv = Dropout(conv_dropout)(conv)
        conv = MaxPooling1D(conv_poolsize)(conv)
        
    # global Pooling 1D
    conv = GlobalMaxPool1D()(conv)
    
    # dense layers:
    for idense in range(dense_layers):
        if idense == 0:
            dense = Dense(ls_dense_neurns[idense],activation=ls_dense_actvf[idense])(conv)
        else:
            dense = Dense(ls_dense_neurns[idense],activation=ls_dense_actvf[idense])(dense)
            
    outputs = Dense(out_neurns,activation=out_actvf)(dense)
        
    model = Model(inputs,outputs)
    model.compile(loss=loss_func,optimizer=optim,metrics=[eval_metric])
    model_checkpoint = ModelCheckpoint(best_model_record, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    return model, model_checkpoint
   
    
def lstm(X,lstm_layers=3,ls_lstm_units=[128,64,32],lstm_dropout=0.3,
          dense_layers=2, ls_dense_neurns=[32,16], ls_dense_actvf=['relu','relu'],
          out_neurns=1, out_actvf='sigmoid',
           loss_func='binary_crossentropy',optim='adam', eval_metric='acc',
           best_model_record='./models/best_model.hdf5'):
    backend.clear_session()
    inputs = Input(shape=(X.shape[1],X.shape[2]))
    
    # lstm layers
    for ilstm in range(lstm_layers):
        if ilstm == 0:
            lstm_layer = LSTM(ls_lstm_units[ilstm],return_sequences=True)(inputs)
        elif ((ilstm > 0) & (ilstm < lstm_layers-1)):
            lstm_layer = LSTM(ls_lstm_units[ilstm],return_sequences=True)(lstm_layer)
        else:
            lstm_layer = LSTM(ls_lstm_units[ilstm])(lstm_layer)
    
    # dense layers
    for idense in range(dense_layers):
        if idense == 0:
            dense_layer = Dense(ls_dense_neurns[idense],activation=ls_dense_actvf[idense])(lstm_layer)
        else:
            dense_layer = Dense(ls_dense_neurns[idense],activation=ls_dense_actvf[idense])(dense_layer)
    outputs = Dense(out_neurns, activation='sigmoid')(dense_layer)
    
    model = Model(inputs, outputs)
    model.compile(loss=loss_func,optimizer=optim,metrics=[eval_metric])
    model_checkpoint = ModelCheckpoint(best_model_record, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    return model, model_checkpoint 

def log_specgram(audio, sample_rate, eps=1e-10):
    freqs, times, spec = signal.spectrogram(audio,fs=sample_rate, detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)
    
def extract_spectrogram_features(X, sample_rate):
  features=[]
  for i in X:
    _, _, spectrogram = log_specgram(i, sample_rate)
    
    mean = np.mean(spectrogram, axis=0)
    std = np.std(spectrogram, axis=0)
    spectrogram = (spectrogram - mean) / std
    
    features.append(spectrogram)

  return np.array(features)

def freeze_layers(model):
    for i in model.layers:
        i.trainable = False
        if isinstance(i, Model):
            freeze_layers(i)
    return model

def chunk_data(samples, duration=2, r_sample=16000):
  
  data=[]
  for offset in range(0, len(samples), r_sample):
    start = offset
    end   = offset + duration*r_sample
    chunk = samples[start:end]
    
    if(len(chunk)==duration*r_sample):
      data.append(chunk)
    
  return data