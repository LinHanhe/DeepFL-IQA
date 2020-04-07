import numpy as np
import keras
from random import shuffle
import keras.backend as K
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from numpy.random import rand
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications import*
from keras.layers import *
import random
import tensorflow as tf
import os
import pandas as pd
from keras.models import Model


def extract_mlsp_feats(ids,model,data_dir):
    
    feats = []
    i = 1
    for index, row in ids.iterrows():
        im = Image.open(data_dir + str(row[0]))
        x = img_to_array(im)
        x = np.expand_dims(x, axis=0)
        x = x/255
        
        im.close()
        feat = model.predict(x)
        feats.append(feat)
        if i % 1000 == 0:
            print('%d images' % (i))
        i += 1
    print('Done...')   
    return np.squeeze(np.array(feats), axis=1)


def model_inceptionresnet_multigap(input_shape=(None, None, 3), 
                                   return_sizes=False):
    """
    Build InceptionResNetV2 multi-GAP model, that extracts narrow MLSP features.
    
    :param input_shape: shape of the input images
    :param return_sizes: return the sizes of each layer: (model, gap_sizes)
    :return: model or (model, gap_sizes)
    """
    #print 'Loading InceptionResNetV2 multi-gap with input_shape:', input_shape
    
    
    model_base = InceptionResNetV2(weights='imagenet',
                                  include_top=False,
                                  input_shape=input_shape)
    
    model_base.load_weights('model/quality-mlsp-mtl-mse-loss.hdf5')
        
    feature_layers = [l for l in model_base.layers if 'mixed' in l.name]
    gaps = [GlobalAveragePooling2D(name="gap%d" % i)(l.output)
           for i, l in enumerate(feature_layers)]
    concat_gaps = Concatenate(name='concatenated_gaps')(gaps)
    
    model = Model(inputs=model_base.input, outputs=concat_gaps)
    
    if return_sizes:
        gap_sizes = [np.int32(g.get_shape()[1]) for g in gaps]
        return (model, gap_sizes)
    
    else:
        return model
    
    

    
    
def fc_model(input_num=16928):
    input_ = Input(shape=(input_num,))
    x = Dense(2048, kernel_initializer='he_normal', activation='relu')(input_)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(1024, kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(256, kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    pred = Dense(1, activation='linear')(x)

    model = Model(input_,pred)
    return model


def sample_data(filepath,feats,dmos):
    # split data into 60% 20% and 20% according to image content
    
    data = {}
    df_ids = pd.read_csv(filepath)
    list_ref_img = df_ids.ref_img.unique()
    df_ref_img = pd.DataFrame(list_ref_img,columns=['ref_img'])

    
    train_ref_img = df_ref_img.sample(frac=0.6)
    valid_ref_img = df_ref_img.drop(train_ref_img.index).sample(frac=0.5)
    test_ref_img = df_ref_img.drop(train_ref_img.index).drop(valid_ref_img.index)
      
    train_feats = feats[df_ids['ref_img'].isin(train_ref_img.ref_img)]
    valid_feats = feats[df_ids['ref_img'].isin(valid_ref_img.ref_img)]
    test_feats = feats[df_ids['ref_img'].isin(test_ref_img.ref_img)]
        
    train_dmos = dmos[df_ids['ref_img'].isin(train_ref_img.ref_img)]
    valid_dmos = dmos[df_ids['ref_img'].isin(valid_ref_img.ref_img)]
    test_dmos = dmos[df_ids['ref_img'].isin(test_ref_img.ref_img)]

    data['train_feats'] = train_feats
    data['valid_feats'] = valid_feats
    data['test_feats'] = test_feats
    
    
    data['train_dmos'] = train_dmos
    data['valid_dmos'] = valid_dmos
    data['test_dmos'] = test_dmos
    
    return data 









