# -*- coding: utf-8 -*-
"""Parallel_is_all_you_want.ipynb

# Parallel is All You Want: Combining Spatial and Temporal Feature Representions of Speech Emotion by Parallelizing CNNs and Transformer-Encoders
# Abstract
In this notebook, I'm going to build upon my [Intro to Speech Audio Classification repo](https://github.com/IliaZenkov/sklearn-audio-classification) and build two parallel convolutional neural networks (CNN) in parallel with a Transformer encoder network to classify audio data. We're working on the [RAVDESS dataset](https://smartlaboratory.org/ravdess/) to classify emotions from one of 8 classes. We combine the CNN for spatial feature representation and the Transformer for temporal feature representation. We augment the training data by increasing variation in the dataset to reduce overfitting; we use Additive White Gaussian Noise (AWGN) to augment the RAVDESS dataset three-fold for a total of 4320 audio samples.

We harness the image-classification and spatial feature representation power of the CNN by treating MFCC plots as grayscale images; their width is a time scale, their height is a frequency scale. The value of each pixel in the MFCC is the intensity of the audio signal at a particular range of mel frequencies at a time step. 

Because of the sequential nature of the data, we will also use the Transformer to try and model as accurately as possible the temporal relationships between pitch transitions in emotions.  

This notebook takes inspirations from a variety of recent advances in deep learning and network architectures; in particular, stacked and parallel CNN networks combined with multi-head self-attention layers from the Transformer Encoder. I hypothesize that the expansion of CNN filter channel dimensions and reduction of feature maps will provide the most expressive feature representation at the lowest computational cost, while the Transformer-Encoder is used with the hypothesis that the network will learn to predict frequency distributions of different emotions according to the global structure of the MFCC plot (and indirectly, mel spectrogram) of each emotion. **With the strength of the CNN in spatial feature representation and Transformer in sequence encoding, I manage to achieve a 97% accuracy on a hold-out set from the RAVDESS dataset.**

<!--TABLE OF CONTENTS-->
# Table of Contents
- [Introduction](#Introduction)
  - [Define features](#Define-features)
  - [Load Data and Extract Features](#Load-Data-and-Extract-Features)
  - [Augmenting the Data with AWGN: Additive White Gaussian Noise](#Augmenting-the-Data-with-AWGN---Additive-White-Gaussian-Noise)
  - [Format Data into Tensor-Ready 4D Arrays](#Format-Data-into-Tensor-Ready-4D-Arrays)
  - [Split into Train/Validation/Test Sets](#Split-into-Train/Validation/Test-Sets)
  - [Feature Scaling](#Feature-Scaling)
- [Architecture Overview](#Architecture-Overview)
- [CNN Motivation](#CNN-Motivation)
- [Transformer-Encoder Motivation](#Transformer-Encoder-Motivation)
- [Building Model Architecture and Forward Pass](#Build-Model-Architecture-and-Define-Forward-Pass)
  - [Analyzing The Flow of Tensors Through the Network](#Analyzing-The-Flow-of-Tensors-Through-the-Network)
  - [Choosing Loss Function](#Define-Loss/Criterion)
  - [Choosing Optimizer](#Choose-Optimizer)
  - [Build Training Step](#Define-Training-Step)
  - [Build Validation Step](#Define-Validation-Step)
  - [Make Checkpoint Functions](#Make-Checkpoint-Functions)
- [Build Training Loop](#Build-Training-Loop)
  - [Train Model](#Train-It)
- [Check the Loss Curve's Behaviour](#Check-the-Loss-Curve's-Behaviour)
- [Evaluate Performance on Test Set](#Evaluate-Performance-on-Test-Set)
- [Conclusion](#Conclusion)
- [References](#References)


# Introduction
From my previous notebook: "Long-Short-Term-Memory Recurrent Neural Networks (LSTM RNNs) and Convolutional Neural Networks (CNNs) are excellent DNN candidates for audio data classification: LSTM RNNs because of their excellent ability to interpret sequential data such as features of the audio waveform represented as a time series; CNNs because features engineered on audio data such as spectrograms have marked resemblance to images, in which CNNs excel at recognizing and discriminating between distinct patterns." - Me 

I'm going to build on that - CNNs are still the hallmark of image classification today, although even in this domain Transformers are beginning to take the main stage: A [2021 ICLR submission: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy) claims they've implemented a Transformer for image classification that outperforms a state of the art CNN, and at a much lower computational complexity.

In addition to taking inspiration from the above, it's also no longer 2015 - so instead of the LSTM-RNN I'm going to implement its successor the Transformer model in parallel with a CNN to try and get state-of-the-art performance on the RAVDESS dataset. 

Other motivations for the architecture of this model come from a variety of papers from the past few years. **The most notable inspirations are:**
- The Transformer: [Attention is All You Need](https://arxiv.org/abs/1706.03762) for the Transformer
- Inception and GoogLeNet: [Going Deeper with Convolutions](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf) for parallel, stacked CNNs
- AlexNet: [ImageNet Classification with Deep Convolutional
](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) for increasing the complexity of feature maps with deeper CNN networks, as well as data augmentation by adding modified versions of the training data to itself
- VGGNet: [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) for using fixed size kernels throughout stacked CNN layers
- LeNet: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) for the convolution>pool>convolution>pool paradigm
- Self-Attention: [Long Short-Term Memory-Networks for Machine Reading](https://arxiv.org/pdf/1601.06733.pdf) for understanding Transformer architecture
- Dropout regularization: [Dropout: A Simple Way to Prevent Neural Networks from
Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) speaks for itself
- Batch Norm: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) speaks for itself

Let's get to it.

#### Setup
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, glob
import librosa
import librosa.display
import IPython
from IPython.display import Audio
from IPython.display import Image
import warnings; warnings.filterwarnings('ignore') #matplot lib complains about librosa

#google colab has an old version of librosa with missing mel spectrogram args (for MFCC); upgrade to current
!pip install -U librosa

# needed to import dataset from google drive into colab 
from google.colab import drive
drive.mount("/content/gdrive")

# copy RAVDESS dataset from gdrive and unzip
!cp '/content/gdrive/My Drive/DL/RAVDESS.zip' .
!unzip -q RAVDESS.zip

"""## Define features

Define features as in the previous notebook on this task from my ['sklearn-audio-classification' repo](https://github.com/IliaZenkov/sklearn-audio-classification). That notebook explains the motivation behind the Mel Spectrogram and its derivative MFCC, which we use as a feature. In short, we're looking for transitions in audible pitch frequencies. 

**MFCCs alone provide the best accuracy in this model with training considerations in mind - and provide as good an accuracy as using chromagrams + mel spectrograms + MFCCs. We don't want extra complexity in a highly parameterized deep neural net such as this one** (unless we absolutely need it).
"""

# RAVDESS native sample rate is 48k
sample_rate = 48000

# Mel Spectrograms are not directly used as a feature in this model
# Mel Spectrograms are used in calculating MFCCs, which are a higher-level representation of pitch transition
# MFCCs work better - left the mel spectrogram function here in case anyone wants to experiment
def feature_melspectrogram(
    waveform, 
    sample_rate,
    fft = 1024,
    winlen = 512,
    window='hamming',
    hop=256,
    mels=128,
    ):
    
    # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # Using 8khz as upper frequency bound should be enough for most speech classification tasks
    melspectrogram = librosa.feature.melspectrogram(
        y=waveform, 
        sr=sample_rate, 
        n_fft=fft, 
        win_length=winlen, 
        window=window, 
        hop_length=hop, 
        n_mels=mels, 
        fmax=sample_rate/2)
    
    # convert from power (amplitude**2) to decibels
    # necessary for network to learn - doesn't converge with raw power spectrograms 
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
    
    return melspectrogram

def feature_mfcc(
    waveform, 
    sample_rate,
    n_mfcc = 40,
    fft = 1024,
    winlen = 512,
    window='hamming',
    #hop=256, # increases # of time steps; was not helpful
    mels=128
    ):

    # Compute the MFCCs for all STFT frames 
    # 40 mel filterbanks (n_mfcc) = 40 coefficients
    mfc_coefficients=librosa.feature.mfcc(
        y=waveform, 
        sr=sample_rate, 
        n_mfcc=n_mfcc,
        n_fft=fft, 
        win_length=winlen, 
        window=window, 
        #hop_length=hop, 
        n_mels=mels, 
        fmax=sample_rate/2
        ) 

    return mfc_coefficients

def get_features(waveforms, features, samplerate):

    # initialize counter to track progress
    file_count = 0

    # process each waveform individually to get its MFCCs
    for waveform in waveforms:
        mfccs = feature_mfcc(waveform, sample_rate)
        features.append(mfccs)
        file_count += 1
        # print progress 
        print('\r'+f' Processed {file_count}/{len(waveforms)} waveforms',end='')
    
    # return all features from list of waveforms
    return features

def get_waveforms(file):
    
    # load an individual sample audio file
    # read the full 3 seconds of the file, cut off the first 0.5s of silence; native sample rate = 48k
    # don't need to store the sample rate that librosa.load returns
    waveform, _ = librosa.load(file, duration=3, offset=0.5, sr=sample_rate)
    
    # make sure waveform vectors are homogenous by defining explicitly
    waveform_homo = np.zeros((int(sample_rate*3,)))
    waveform_homo[:len(waveform)] = waveform
    
    # return a single file's waveform                                      
    return waveform_homo
    
# RAVDESS dataset emotions
# shift emotions left to be 0 indexed for PyTorch
emotions_dict ={
    '0':'surprised',
    '1':'neutral',
    '2':'calm',
    '3':'happy',
    '4':'sad',
    '5':'angry',
    '6':'fearful',
    '7':'disgust'
}

# Additional attributes from RAVDESS to play with
emotion_attributes = {
    '01': 'normal',
    '02': 'strong'
}

"""## Load Data and Extract Features


We process each file in the dataset and extract its features.

We return the waveforms and the labels (from the file names of the RAVDESS audio samples). We return the raw waveforms because we're going to do some extra processing.
"""

# path to data for glob
data_path = 'RAVDESS dataset/Actor_*/*.wav'

def load_data():
    # features and labels
    emotions = []
    # raw waveforms to augment later
    waveforms = []
    # extra labels
    intensities, genders = [],[]
    # progress counter
    file_count = 0
    for file in glob.glob(data_path):
        # get file name with labels
        file_name = os.path.basename(file)
        
        # get emotion label from the sample's file
        emotion = int(file_name.split("-")[2])

        #  move surprise to 0 for cleaner behaviour with PyTorch/0-indexing
        if emotion == 8: emotion = 0 # surprise is now at 0 index; other emotion indeces unchanged

        # can convert emotion label to emotion string if desired, but
        # training on number is better; better convert to emotion string after predictions are ready
        # emotion = emotions_dict[str(emotion)]
        
        # get other labels we might want
        intensity = emotion_attributes[file_name.split("-")[3]]
        # even actors are female, odd are male
        if (int((file_name.split("-")[6]).split(".")[0]))%2==0: 
            gender = 'female' 
        else: 
            gender = 'male'
            
        # get waveform from the sample
        waveform = get_waveforms(file)
        
        # store waveforms and labels
        waveforms.append(waveform)
        emotions.append(emotion)
        intensities.append(intensity) # store intensity in case we wish to predict
        genders.append(gender) # store gender in case we wish to predict 
        
        file_count += 1
        # keep track of data loader's progress
        print('\r'+f' Processed {file_count}/{1440} audio samples',end='')
        
    return waveforms, emotions, intensities, genders

# load data 
# init explicitly to prevent data leakage from past sessions, since load_data() appends
waveforms, emotions, intensities, genders = [],[],[],[]
waveforms, emotions, intensities, genders = load_data()

"""## Check extracted audio waveforms and labels:"""

print(f'Waveforms set: {len(waveforms)} samples')
# we have 1440 waveforms but we need to know their length too; should be 3 sec * 48k = 144k
print(f'Waveform signal length: {len(waveforms[0])}')
print(f'Emotions set: {len(emotions)} sample labels')

"""Looks good. 1440 samples and 1440 labels in total.

**Waveforms are 144k long because 3 seconds * 48k sample rate = 144k length array representing the 3 second audio snippet.**

## Split into Train/Validation/Test Sets
We'll use an 80/10/10 train/validation/test split to maximize training data and keep a reasonable validation/test set. 

**We're splitting waveforms so we can process train/validation/test waveforms separately and avoid data leakage.** 

**Have to take care to split the sets proportionally w.r.t. emotion.**
"""

# create storage for train, validation, test sets and their indices
train_set,valid_set,test_set = [],[],[]
X_train,X_valid,X_test = [],[],[]
y_train,y_valid,y_test = [],[],[]

# convert waveforms to array for processing
waveforms = np.array(waveforms)

# process each emotion separately to make sure we builf balanced train/valid/test sets 
for emotion_num in range(len(emotions_dict)):
        
    # find all indices of a single unique emotion
    emotion_indices = [index for index, emotion in enumerate(emotions) if emotion==emotion_num]

    # seed for reproducibility 
    np.random.seed(69)
    # shuffle indicies 
    emotion_indices = np.random.permutation(emotion_indices)

    # store dim (length) of the emotion list to make indices
    dim = len(emotion_indices)

    # store indices of training, validation and test sets in 80/10/10 proportion
    # train set is first 80%
    train_indices = emotion_indices[:int(0.8*dim)]
    # validation set is next 10% (between 80% and 90%)
    valid_indices = emotion_indices[int(0.8*dim):int(0.9*dim)]
    # test set is last 10% (between 90% - end/100%)
    test_indices = emotion_indices[int(0.9*dim):]

    # create train waveforms/labels sets
    X_train.append(waveforms[train_indices,:])
    y_train.append(np.array([emotion_num]*len(train_indices),dtype=np.int32))
    # create validation waveforms/labels sets
    X_valid.append(waveforms[valid_indices,:])
    y_valid.append(np.array([emotion_num]*len(valid_indices),dtype=np.int32))
    # create test waveforms/labels sets
    X_test.append(waveforms[test_indices,:])
    y_test.append(np.array([emotion_num]*len(test_indices),dtype=np.int32))

    # store indices for each emotion set to verify uniqueness between sets 
    train_set.append(train_indices)
    valid_set.append(valid_indices)
    test_set.append(test_indices)

# concatenate, in order, all waveforms back into one array 
X_train = np.concatenate(X_train,axis=0)
X_valid = np.concatenate(X_valid,axis=0)
X_test = np.concatenate(X_test,axis=0)

# concatenate, in order, all emotions back into one array 
y_train = np.concatenate(y_train,axis=0)
y_valid = np.concatenate(y_valid,axis=0)
y_test = np.concatenate(y_test,axis=0)

# combine and store indices for all emotions' train, validation, test sets to verify uniqueness of sets
train_set = np.concatenate(train_set,axis=0)
valid_set = np.concatenate(valid_set,axis=0)
test_set = np.concatenate(test_set,axis=0)

# check shape of each set
print(f'X_train:{X_train.shape}, y_train:{y_train.shape}')
print(f'X_valid:{X_valid.shape}, y_valid:{y_valid.shape}')
print(f'X_test:{X_test.shape}, y_test:{y_test.shape}')

# make sure train, validation, test sets have no overlap/are unique
# get all unique indices across all sets and how many times each index appears (count)
uniques, count = np.unique(np.concatenate([train_set,test_set,valid_set],axis=0), return_counts=True)

# if each index appears just once, and we have 1440 such unique indices, then all sets are unique
if sum(count==1) == len(emotions):
    print(f'\nSets are unique: {sum(count==1)} samples out of {len(emotions)} are unique')
else:
    print(f'\nSets are NOT unique: {sum(count==1)} samples out of {len(emotions)} are unique')

"""## Extract Features

Extract the features from unaugmented waveforms first. In the next step, we'll append features from augmented waveforms to these 'native' features.
"""

# initialize feature arrays
# We extract MFCC features from waveforms and store in respective 'features' array
features_train, features_valid, features_test = [],[],[]

print('Train waveforms:') # get training set features 
features_train = get_features(X_train, features_train, sample_rate)

print('\n\nValidation waveforms:') # get validation set features
features_valid = get_features(X_valid, features_valid, sample_rate)

print('\n\nTest waveforms:') # get test set features 
features_test = get_features(X_test, features_test, sample_rate)

print(f'\n\nFeatures set: {len(features_train)+len(features_test)+len(features_valid)} total, {len(features_train)} train, {len(features_valid)} validation, {len(features_test)} test samples')
print(f'Features (MFC coefficient matrix) shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')

"""## Augmenting the Data with AWGN - Additive White Gaussian Noise

### Motivation

Since our dataset is small, it is prone to overfitting - especially with highly parameterized deep neural net models
such as the one we aim to build in this notebook. As such, we're going to want to augment our data. Generating more real samples will be immensely difficult. Instead, we can add white noise to the audio signals - not only to mask the effect of random noise present in the training set - but also **to create pseudo-new training samples and offset the impact of noise intrinsic to the dataset.** 

In addition, the RAVDESS dataset is extremely clean - we will likely want to make predictions on noisy, real-world data - yet another reason to augment the training data.

We're going to use Additive White Gaussian Noise (AWGN). It's Additive because we're adding it to the source audio signal,
**it's Gaussian because the noise vector will be sampled from a normal distribution and have a time average of zero (zero-mean), and it's white because after a whitening transformation the noise will add power to the audio signal uniformly across the frequency distribution.**

We need a good balance of noise - too little will be useless, and too much will make it too difficult for the network to learn from the training data. **Note that this is just for training - we would _not_ need to add AWGN to real-world data on which we make predictions** (although we could). 

### Math
The key parameters in AWGN are the signal to noise ratio (SNR), defining the magnitude of the noise added w.r.t. the audio signal. We parameterize AWGN with the minimum and maximize SNR so we can pick a random SNR to use in augmenting each sample's waveform.

We need to constrain covariance to make it true AWGN. **We make a zero-mean vector of Gaussian noises (np.random.normal) that are statistically dependent. We need to apply a [whitening transformation](https://en.wikipedia.org/wiki/Whitening_transformation)**, a linear transformation taking a vector of random normal (Gaussian) variables with a known covariance matrix and mapping it to a new vector whose covariance is the identity matrix, i.e. the vector is now perfectly uncorrelated with a diaganol covariance matrix, each point of noise having variance == stdev == 1. **The whitening transformation by definition transforms a vector into a white noise vector.**

We're going to add the AWGN augmented waveforms as new samples to our dataset. **Since we generate AWGN which is random for each and every sample - random random noise - we can add multiples of our noise-augmented dataset. I'll add 2 extra identical, randomly noisy datasets with 1440 samples each to get a dataset with 1440 native + 1440x2 == 4320 noisy samples.**
"""

def awgn_augmentation(waveform, multiples=2, bits=16, snr_min=15, snr_max=30): 
    
    # get length of waveform (should be 3*48k = 144k)
    wave_len = len(waveform)
    
    # Generate normally distributed (Gaussian) noises
    # one for each waveform and multiple (i.e. wave_len*multiples noises)
    noise = np.random.normal(size=(multiples, wave_len))
    
    # Normalize waveform and noise
    norm_constant = 2.0**(bits-1)
    norm_wave = waveform / norm_constant
    norm_noise = noise / norm_constant
    
    # Compute power of waveform and power of noise
    signal_power = np.sum(norm_wave ** 2) / wave_len
    noise_power = np.sum(norm_noise ** 2, axis=1) / wave_len
    
    # Choose random SNR in decibels in range [15,30]
    snr = np.random.randint(snr_min, snr_max)
    
    # Apply whitening transformation: make the Gaussian noise into Gaussian white noise
    # Compute the covariance matrix used to whiten each noise 
    # actual SNR = signal/noise (power)
    # actual noise power = 10**(-snr/10)
    covariance = np.sqrt((signal_power / noise_power) * 10 ** (- snr / 10))
    # Get covariance matrix with dim: (144000, 2) so we can transform 2 noises: dim (2, 144000)
    covariance = np.ones((wave_len, multiples)) * covariance

    # Since covariance and noise are arrays, * is the haddamard product 
    # Take Haddamard product of covariance and noise to generate white noise
    multiple_augmented_waveforms = waveform + covariance.T * noise
    
    return multiple_augmented_waveforms

def augment_waveforms(waveforms, features, emotions, multiples):
    # keep track of how many waveforms we've processed so we can add correct emotion label in the same order
    emotion_count = 0
    # keep track of how many augmented samples we've added
    added_count = 0
    # convert emotion array to list for more efficient appending
    emotions = emotions.tolist()

    for waveform in waveforms:

        # Generate 2 augmented multiples of the dataset, i.e. 1440 native + 1440*2 noisy = 4320 samples total
        augmented_waveforms = awgn_augmentation(waveform, multiples=multiples)

        # compute spectrogram for each of 2 augmented waveforms
        for augmented_waveform in augmented_waveforms:

            # Compute MFCCs over augmented waveforms
            augmented_mfcc = feature_mfcc(augmented_waveform, sample_rate=sample_rate)

            # append the augmented spectrogram to the rest of the native data
            features.append(augmented_mfcc)
            emotions.append(emotions[emotion_count])

            # keep track of new augmented samples
            added_count += 1

            # check progress
            print('\r'+f'Processed {emotion_count + 1}/{len(waveforms)} waveforms for {added_count}/{len(waveforms)*multiples} new augmented samples',end='')

        # keep track of the emotion labels to append in order
        emotion_count += 1
        
        # store augmented waveforms to check their shape
        augmented_waveforms_temp.append(augmented_waveforms)
    
    return features, emotions

"""### Compute AWGN-augmented features and add to the rest of the dataset"""

# store augmented waveforms to verify their shape and random-ness
augmented_waveforms_temp = []

# specify multiples of our dataset to add as augmented data
multiples = 2

print('Train waveforms:') # augment waveforms of training set
features_train , y_train = augment_waveforms(X_train, features_train, y_train, multiples)

print('\n\nValidation waveforms:') # augment waveforms of validation set
features_valid, y_valid = augment_waveforms(X_valid, features_valid, y_valid, multiples)

print('\n\nTest waveforms:') # augment waveforms of test set 
features_test, y_test = augment_waveforms(X_test, features_test, y_test, multiples)

# Check new shape of extracted features and data:
print(f'\n\nNative + Augmented Features set: {len(features_train)+len(features_test)+len(features_valid)} total, {len(features_train)} train, {len(features_valid)} validation, {len(features_test)} test samples')
print(f'{len(y_train)} training sample labels, {len(y_valid)} validation sample labels, {len(y_test)} test sample labels')
print(f'Features (MFCC matrix) shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')

"""### Check Augmented Waveforms:"""

# pick a random waveform, but same one from native and augmented set for easier comparison
plt.figure(figsize=(15,4))
plt.subplot(1, 2, 1)
librosa.display.waveplot(waveforms[1], sr=sample_rate)
plt.title('Native')

plt.subplot(1, 2, 2)
librosa.display.waveplot(augmented_waveforms_temp[1], sr=sample_rate)
plt.title('AWGN Augmented')
plt.show()

"""Looks noisy alright. Noise is clearly visible in otherwise-silent regions of the waveform.

**Note that augmentation was only done after splitting data into train, validation, and test sets - and we processed each set separately.**

**When we augmented the data before splitting it, test and validation data leaked into the training set, giving a 97% test accuracy.**

## Format Data into Tensor Ready 4D Arrays
We don't have a colour channel in our MFCC feature array of dim (#samples, #MFC coefficients, time steps). **We have an analog of a black and white image: instead of 3 colour channels, we have 1 signal intensity channel: magnitude of each of 40 mel frequency coefficients at time t.**

**We need an input channel dim to expand to output channels using CNN filters. We create a dummy channel dim to expand features into 2D-CNN-ready 4D tensor format: N x C x H x W.**
"""

# need to make dummy input channel for CNN input feature tensor
X_train = np.expand_dims(features_train,1)
X_valid = np.expand_dims(features_valid, 1)
X_test = np.expand_dims(features_test,1)

# convert emotion labels from list back to numpy arrays for PyTorch to work with 
y_train = np.array(y_train)
y_valid = np.array(y_valid)
y_test = np.array(y_test)

# confiorm that we have tensor-ready 4D data array
# should print (batch, channel, width, height) == (4320, 1, 128, 282) when multiples==2
print(f'Shape of 4D feature array for input tensor: {X_train.shape} train, {X_valid.shape} validation, {X_test.shape} test')
print(f'Shape of emotion labels: {y_train.shape} train, {y_valid.shape} validation, {y_test.shape} test')

# free up some RAM - no longer need full feature set or any waveforms 
del features_train, features_valid, features_test, waveforms, augmented_waveforms_temp

"""## Feature Scaling
Scaling will drastically decrease the length of time the model needs to train to convergence - it will have easier computations to perform on smaller magnitudes. **For reference, scaling reduces the time to convergence from about 500 to 200 epochs for this model.**

**Standard Scaling makes the most sense because we have features whose target distribution we don't know.** When I performed classification on this dataset with an MLP classifier standard scaling was best across a variety of conditions and features.
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#### Scale the training data ####
# store shape so we can transform it back 
N,C,H,W = X_train.shape
# Reshape to 1D because StandardScaler operates on a 1D array
# tell numpy to infer shape of 1D array with '-1' argument
X_train = np.reshape(X_train, (N,-1)) 
X_train = scaler.fit_transform(X_train)
# Transform back to NxCxHxW 4D tensor format
X_train = np.reshape(X_train, (N,C,H,W))

##### Scale the validation set ####
N,C,H,W = X_valid.shape
X_valid = np.reshape(X_valid, (N,-1))
X_valid = scaler.transform(X_valid)
X_valid = np.reshape(X_valid, (N,C,H,W))

#### Scale the test set ####
N,C,H,W = X_test.shape
X_test = np.reshape(X_test, (N,-1))
X_test = scaler.transform(X_test)
X_test = np.reshape(X_test, (N,C,H,W))

# check shape of each set again
print(f'X_train scaled:{X_train.shape}, y_train:{y_train.shape}')
print(f'X_valid scaled:{X_valid.shape}, y_valid:{y_valid.shape}')
print(f'X_test scaled:{X_test.shape}, y_test:{y_test.shape}')

"""### Save and Reload Data as NumPy Arrays 
We can save the training/validation/test data as numpy arrays to enable faster loading in case the notebook kernel crashes / google colab runtime crashes / any number of reasons the training data might be cleared from memory. This is much faster than loading 1440 files and computing their features again - not to mention augmented features.
"""

###### SAVE #########
# choose save file name 
filename = 'features+labels.npy'

# open file in write mode and write data
with open(filename, 'wb') as f:
    np.save(f, X_train)
    np.save(f, X_valid)
    np.save(f, X_test)
    np.save(f, y_train)
    np.save(f, y_valid)
    np.save(f, y_test)

print(f'Features and labels saved to {filename}')

##### LOAD #########
# choose load file name 
filename = 'features+labels.npy'

# open file in read mode and read data 
with open(filename, 'rb') as f:
    X_train = np.load(f)
    X_valid = np.load(f)
    X_test = np.load(f)
    y_train = np.load(f)
    y_valid = np.load(f)
    y_test = np.load(f)

# Check that we've recovered the right data
print(f'X_train:{X_train.shape}, y_train:{y_train.shape}')
print(f'X_valid:{X_valid.shape}, y_valid:{y_valid.shape}')
print(f'X_test:{X_test.shape}, y_test:{y_test.shape}')

"""# Architecture Overview
<img src="reports/cnn-transformer-final.png">
As a whole, the CNN architecture of this network is inspired by a combination of the golden standards in image and sequence processing over the last few years.

Each 3-layer deep 2D convolutional block is extremely similar to the **classic [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) architecture: Conv->Pool>Conv>Pool>FC.** 

**[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) forms the basis for increasing the complexity of feature maps with channel expansion through stacked CNN layers; [Inception and GoogLeNet](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf) are the inspiration for parallelizing CNN layers in the hopes of diverisfying the features learned by the network.** 

**[VGGNet](https://arxiv.org/pdf/1409.1556.pdf) proved the unreasonable efficiency of using fixed sized kernels throughout deeply stacked CNN layers;** I found this to extend to this task. Specifically, VGG saw an improvement over AlexNet largely by replacing large kernels (i.e. 11x11 stride 5) with smaller ones of 3x3 stride 1. One of the motivations that VGG cites for this is that the 3x3 kernel is the smallest kernel size choice in understanding spatial data w.r.t. up/down/left/right (although VGG also uses 1x1 kernels). VGGNet also inspires the maxpool kernel size of 2x2 stride 2, as I have used at the first layer of each convolutional block. 

To be more precise, the motivation to use small stacked filters is two-fold: Computational efficiency and expressivity of feature representation. When we stack 3 3x3 kernels on top of eachother as in this architecture, the second layer has a 5x5 view of the original input volume, and the 3rd layer a 7x7 view. However, the nonlinearities between each smaller layer convey more complex feature representations, whereas a single 7x7 layer would only perform a linear transformation itself. Furthermore, if we keep channel (C) consistent between layers, then 3 3x3 kernels are parameterized by (3(C(3x3xC)) = 27C^2 parameters, while just one 7x7 kernel needs C(7x7xC) = 49C^2 parameters. **Ultimately, small stacked kernels appear to be both more powerful and efficient - although, in [Large Kernel Matters -
Improve Semantic Segmentation by Global Convolutional Network](https://arxiv.org/pdf/1703.02719.pdf), the authors conclude that a larger kernel outperforms smaller stacked kernels for semantic segmentation - however, since we are just doing the semantic part (classification) and don't care about "where" the emotion is - this shouldn't apply.**

Finally, the original [2015 Batch Normalization (BN) paper](https://arxiv.org/abs/1502.03167) suggests that "We add the BN transform immediately before the nonlinearity" i.e. before ReLU; however, **I achieved better performance out of this architechture using BN after ReLU. See Keras author's [Francois Chollet's response on GitHub](https://github.com/keras-team/keras/issues/1802#issuecomment-187966878) regarding the BN order issue: "I can guarantee that recent code written by Christian \[Szegedy\] applies relu before BN".** 

The Transformer architecture is precisely as in Viswani et al, 2017: Attention is All You Need, but I use 4 stacked encoders instead of 6 as in their paper. For more details on the Transformer block: [Appendix B: The Transformer and Self-Attention (is All You Need)](#Appendix-B:-The-Transformer-and-Self-Attention-(is-All-You-Need))

# CNN Motivation
**CNNs with 2D convolutional layers are the gold standard for image processing**, other than the recent advances in the Transformer for images. 2D convolution layers accept input feature maps in a (N,C,H,W) (batch size, channel, height, width) format. We have 4320 mels - 1440 native and 2880 noise augmented - each mel is of shape 128x563 representing 128 mel frequency bands with 563 timesteps each. **We can imagine MFCC plots to be a black and white image with 1 signal intensity channel.** Our MFCC input feature tensor will thus be of shape (4320, 1, 128, 563). I'm going to refer to input/output feature maps and input/output volumes interchangeably, but they have the same meaning. After an activation function operates on a feature map, it produces an activation map.

I'm using 3x3 kernels in all 3 layers in both CNN blocks. The first layer only has a single input channel creating a 1x3x3 filter. The next layer has 16 input channels and 32 output channels , producing 32 unique 16x3x3 filters, *each* filter having 16x3x3 = 144 weights. The second layer is applying 32 differently weighted 16x3x3 filters to an input volume of 16x64x281 (the 2x2 maxpooled output of the first layer), producing an output feature map of 32x8x35 after 8x8 stride 8 maxpooling. The last layer has 32 input channels, so a 32x3x3 filter, and 64 output channels, so 64 unique such filters. The last layer produces an output feature map of 64x1x4 after 8x8 stride 4 maxpooling. **In sum, I hope that the expansion of filter dimensions and reduction of feature maps provides the most expressive feature representation at the lowest computational cost.** 

**The key motivations and operations of a 2D Convolutional Neural Net are described in detail in [Appendix A: Convolutional Neural Nets Dissected](#Appendix-A:-Convolutional-Neural-Nets-Dissected) in order to explain the motivation behind the necessity of the CNN for this task and the considerations to take when optimizing the hyperparameters of each layer in a CNN.**

# Transformer-Encoder Motivation

**I use the Transformer-Encoder layer as introduced in [Attention is All You Need](https://arxiv.org/abs/1706.03762) with the hopes that the network will learn to predict frequency distributions of different emotions according to the global structure of the MFCCs of each emotion.** I could have used LSTM-RNNs to learn the sequence of the spectrogram for each emotion, but the network would only learn to predict frequency changes according to adjacent time steps; in contrast, the multi-head self-attention layers of the transformer enable the network to look at multiple previous time steps when predicting the next. This made sense to me because emotions colour the entire sequence of frequencies, not just at one timestep. 

**I maxpool the input MFCC map to the transformer block to drastically reduce the number of parameters the network needs to learn.**

**The key motivations and operations behind the Transformer architecture are described in detail in [Appendix B: The Transformer](#Appendix-B:-The-Transformer)**

# Build Model Architecture and Define Forward Pass
"""

#change nn.sequential to take dict to make more readable 

class parallel_all_you_want(nn.Module):
    # Define all layers present in the network
    def __init__(self,num_emotions):
        super().__init__() 
        
        ################ TRANSFORMER BLOCK #############################
        # maxpool the input feature map/tensor to the transformer 
        # a rectangular kernel worked better here for the rectangular input spectrogram feature map/tensor
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1,4], stride=[1,4])
        
        # define single transformer encoder layer
        # self-attention + feedforward network from "Attention is All You Need" paper
        # 4 multi-head self-attention layers each with 64-->512--->64 feedforward network
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=40, # input feature (frequency) dim after maxpooling 128*563 -> 64*140 (freq*time)
            nhead=4, # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            dim_feedforward=512, # 2 linear layers in each encoder block's feedforward network: dim 64-->512--->64
            dropout=0.4, 
            activation='relu' # ReLU: avoid saturation/tame gradient/reduce compute time
        )
        
        # I'm using 4 instead of the 6 identical stacked encoder layrs used in Attention is All You Need paper
        # Complete transformer block contains 4 full transformer encoder layers (each w/ multihead self-attention+feedforward)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)
        
        ############### 1ST PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock1 = nn.Sequential(
            
            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, # input volume depth == input channel dim == 1
                out_channels=16, # expand output feature map volume's depth to 16
                kernel_size=3, # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16), # batch normalize the output feature map before activation
            nn.ReLU(), # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2), #typical maxpool kernel size
            nn.Dropout(p=0.3), #randomly zero 30% of 1st layer's output feature map in training
            
            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3), 
            
            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64, # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        ############### 2ND PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock2 = nn.Sequential(
            
            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, # input volume depth == input channel dim == 1
                out_channels=16, # expand output feature map volume's depth to 16
                kernel_size=3, # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16), # batch normalize the output feature map before activation
            nn.ReLU(), # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2), #typical maxpool kernel size
            nn.Dropout(p=0.3), #randomly zero 30% of 1st layer's output feature map in training
            
            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3), 
            
            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64, # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )

        ################# FINAL LINEAR BLOCK ####################
        # Linear softmax layer to take final concatenated embedding tensor 
        #    from parallel 2D convolutional and transformer blocks, output 8 logits 
        # Each full convolution block outputs (64*1*8) embedding flattened to dim 512 1D array 
        # Full transformer block outputs 40*70 feature map, which we time-avg to dim 40 1D array
        # 512*2+40 == 1064 input features --> 8 output emotions 
        self.fc1_linear = nn.Linear(512*2+40,num_emotions) 
        
        ### Softmax layer for the 8 output logits from final FC linear layer 
        self.softmax_out = nn.Softmax(dim=1) # dim==1 is the freq embedding
        
    # define one complete parallel fwd pass of input feature tensor thru 2*conv+1*transformer blocks
    def forward(self,x):
        
        ############ 1st parallel Conv2D block: 4 Convolutional layers ############################
        # create final feature embedding from 1st convolutional layer 
        # input features pased through 4 sequential 2D convolutional layers
        conv2d_embedding1 = self.conv2Dblock1(x) # x == N/batch * channel * freq * time
        
        # flatten final 64*1*4 feature map from convolutional layers to length 256 1D array 
        # skip the 1st (N/batch) dimension when flattening
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1) 
        
        ############ 2nd parallel Conv2D block: 4 Convolutional layers #############################
        # create final feature embedding from 2nd convolutional layer 
        # input features pased through 4 sequential 2D convolutional layers
        conv2d_embedding2 = self.conv2Dblock2(x) # x == N/batch * channel * freq * time
        
        # flatten final 64*1*4 feature map from convolutional layers to length 256 1D array 
        # skip the 1st (N/batch) dimension when flattening
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1) 
        
         
        ########## 4-encoder-layer Transformer block w/ 64-->512-->64 feedfwd network ##############
        # maxpool input feature map: 1*40*282 w/ 1*4 kernel --> 1*40*70
        x_maxpool = self.transformer_maxpool(x)

        # remove channel dim: 1*40*70 --> 40*70
        x_maxpool_reduced = torch.squeeze(x_maxpool,1)
        
        # convert maxpooled feature map format: batch * freq * time ---> time * batch * freq format
        # because transformer encoder layer requires tensor in format: time * batch * embedding (freq)
        x = x_maxpool_reduced.permute(2,0,1) 
        
        # finally, pass reduced input feature map x into transformer encoder layers
        transformer_output = self.transformer_encoder(x)
        
        # create final feature emedding from transformer layer by taking mean in the time dimension (now the 0th dim)
        # transformer outputs 64*140 (freq embedding*time) feature map, take mean of all columns i.e. take time average
        transformer_embedding = torch.mean(transformer_output, dim=0) # dim 40x70 --> 40
        
        ############# concatenate freq embeddings from convolutional and transformer blocks ######
        # concatenate embedding tensors output by parallel 2*conv and 1*transformer blocks
        complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2,transformer_embedding], dim=1)  

        ######### final FC linear layer, need logits for loss #########################
        output_logits = self.fc1_linear(complete_embedding)  
        
        ######### Final Softmax layer: use logits from FC linear, get softmax for prediction ######
        output_softmax = self.softmax_out(output_logits)
        
        # need output logits to compute cross entropy loss, need softmax probabilities to predict class
        return output_logits, output_softmax

"""## Analyzing The Flow of Tensors Through the Network
We zero-pad 1 the input feature map to each convolutional layer to get back from the layer the same shape tensor as we input: zero-pad 1 adds 2 to each of (H, W) dims, and the 3x3, stride 1 kernels cuts off (kernel - stride == 2) dims from each of (H,W). **Zero-pad 1 --> 3x3 stride 1 kernel effectively throws away the zero pads to get same input/output shape from each conv2D block.**

At the end of first convolutional layer in each block we have a maxpool kernel of size 2x2, stride 2 which will take 1 of 4 pixels in its winddow. For the input feature map the maxpool kernel will progress 128/2 = 64 times over the rows and 563/2=281 times over the columns. **Nonoverlapping maxpool kernel reduces each output dim to input dim/kernel size.** We then expand the output channels to 16 making an output feature map of (16x64x281). 

The next two convolutional layers in each block have a maxpool kernel size 8x8, stride 8. Same math as above, maxpool reduces each dim/8. 2nd conv layer takes (16x64x281) --> (32x8x35). 3rd and final conv layer takes (32x8x35) --> (64x1x4).

**Note that in (N,C,H,W) format, for MFCCs H = MFCC (pitch),  W = time step.**

**Complete flow through each convolutional block (C,H,W):**

    Layer 1 ---> 1x128x563 --> PAD-1 --> 1x130x565 --> FILTER --> 16x128x563 --> MAXPOOL 2x2 stride 2 --> 16x64x281

    Layer 2 ---> 16x64x281 --> PAD-1 --> 16x66x283 --> FILTER --> 32x64x281 --> MAXPOOL 8x8 stride 8 --> 32x8x35

    Layer 3 ---> 32x8x35 --> PAD-1 --> 32x10x37 --> FILTER --> 64x8x35 --> MAXPOOL 8x8 stride 8 --> 64x1x4 

    Flatten ---> 64x1x4 --> Final convolutional embedding length 256 1D array


**Complete flow through transformer encoder block (C,H,W):**

    Maxpool 2x4 stride 2x4 ---> 1x128x563 --> 1x64x140

    Drop channel ---> 1x64x140 --> 64x140 (H,W)

    Change dims ---> 64x140 --> 140x64 (W,H)

    4*Transformer encoder ---> 140x64 --> 2x64 (W,H)

    Time average ---> 2x64 --> 1x64 --> Final transformer embedding length 64 1D array

**FC Linear network (C,H,W):**

    Concatenate ---> 256x256x64 --> 576 

    FC Linear layer ---> 576 --> Final linear logits output length 8 1D array

    Softmax layer: 8 ----> 1 predicted emotion / max probability class


We can confirm our network's tensor shapes and flow using the excellent torchsummary package which provides a PyTorch implementation of Keras' model.summary method:
"""

from torchsummary import summary

# need device to instantiate model
device = 'cuda'

# instantiate model for 8 emotions and move to CPU for summary
model = parallel_all_you_want(len(emotions_dict)).to(device)

# include input feature map dims in call to summary()
summary(model, input_size=(1,40,282))

"""## Define Loss/Criterion

We must define the loss function (criterion per PyTorch notation) for the backwards pass of each training iteration. Since our classes our balanced we don't need to specify a class-weight parameter (to balance classes).

**PyTorch nn.CrossEntropyLoss() implements log softmax and negative log likelihood loss (nn.NLLoss() --> nn.LogSoftmax())
We use log softmax for computation benefits and faster gradient optimization. Log softmax heavily penalizes the model when failing to predict the correct class.**
"""

# define loss function; CrossEntropyLoss() fairly standard for multiclass problems 
def criterion(predictions, targets): 
    return nn.CrossEntropyLoss()(input=predictions, target=targets)

"""## Choose Optimizer

I used Adam to train an MLP due to its faster compute and convergence. Adam is great and usually works well with defaults.

**However, a lot of 2018-2020 papers still use SGD. It seems to me that the reason is SGD with properly tuned momentum sometimes (often) converges to lower loss with enough training.**

**Quoting [Wilson et al, 2017](https://arxiv.org/abs/1705.08292):** 

    “We observe that the solutions found by adaptive methods generalize worse (often significantly worse) than SGD, even 
    when these solutions have better training performance. These results suggest that practitioners should reconsider the 
    use of adaptive methods to train neural networks."


"Adaptive methods" refers to the likes of Adam.
   
**I took full advantage of plain old SGD by using the highest momentum leading to convergence, plus a generously long training time.**
"""

optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)

"""## Define Training Step

We define a function to return a single training step defining one iteration of our model.

    Forward pass output logits and softmax probabilities. 

    Record the softmax probabilities to track accuracy. 

    Pass output logits to loss function to compute loss.
    
    Call backwards pass with loss function (backpropogate errors).
    
    Tell optimizer to apply one update step to network parameters.
    
    Zero the accumulated gradient in the optimizer for next iteration.
"""

# define function to create a single step of the training phase
def make_train_step(model, criterion, optimizer):
    
    # define the training step of the training phase
    def train_step(X,Y):
        
        # forward pass
        output_logits, output_softmax = model(X)
        predictions = torch.argmax(output_softmax,dim=1)
        accuracy = torch.sum(Y==predictions)/float(len(Y))
        
        # compute loss on logits because nn.CrossEntropyLoss implements log softmax
        loss = criterion(output_logits, Y) 
        
        # compute gradients for the optimizer to use 
        loss.backward()
        
        # update network parameters based on gradient stored (by calling loss.backward())
        optimizer.step()
        
        # zero out gradients for next pass
        # pytorch accumulates gradients from backwards passes (convenient for RNNs)
        optimizer.zero_grad() 
        
        return loss.item(), accuracy*100
    return train_step

"""## Define Validation Step

Define a function to return a single validation step on the 10% X,y tensor pair to get an idea of our model's generalizibiliy as it trains so we know whether and when to stop it and tune hyperparameters. **Make sure we _do not_ update network parameters during validation by setting model to validation mode. Do not waste resources computing gradients in validation phase by setting torch.no_grad().**
"""

def make_validate_fnc(model,criterion):
    def validate(X,Y):
        
        # don't want to update any network parameters on validation passes: don't need gradient
        # wrap in torch.no_grad to save memory and compute in validation phase: 
        with torch.no_grad(): 
            
            # set model to validation phase i.e. turn off dropout and batchnorm layers 
            model.eval()
      
            # get the model's predictions on the validation set
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax,dim=1)

            # calculate the mean accuracy over the entire validation set
            accuracy = torch.sum(Y==predictions)/float(len(Y))
            
            # compute error from logits (nn.crossentropy implements softmax)
            loss = criterion(output_logits,Y)
            
        return loss.item(), accuracy*100, predictions
    return validate

"""# Make Checkpoint Functions
It's a good idea to save checkpoints of the model state after each epoch. We'll then interrupt training when satisfied with the model's performance and load the appropriate model binary. 

- Resume training if hardware/software fails
- Save compute re-training by training from checkpoint after tuning
- Implement early stopping easily by keeping snapshot of most performant version of model
- Google Colab throttles GPU usage eventually; can't keep re-training from scratch indefinitely
"""

def make_save_checkpoint(): 
    def save_checkpoint(optimizer, model, epoch, filename):
        checkpoint_dict = {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint_dict, filename)
    return save_checkpoint

def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch

"""# Build Training Loop

Build the complete training loop using the training and validation step functions. 

This model is not reasonable to train on CPU, but it's a good way to check if the model compiled successfully. I'm using Google Colab's free GPU (K80 - 24GB RAM ~2.9 TFLOPs). This model is pretty big (383,688 params to learn if I've checked the math correctly) but trains to convergence within ~10 minutes on a K80.

Pick the number of epochs (complete pass of all training samples) to use that is higher than reasonable so the model does not terminate just before convergence - I manually stopped it when it converges. 

**Minibatch size:** [From Yann LeCun's twitter](https://twitter.com/ylecun/status/989610208497360896?lang=en) (with LeCun facebook comment appended) [citing this 2018 minibatch paper](https://arxiv.org/abs/1804.07612):


    
    "Training with large minibatches is bad for your health. More importantly, it's bad for your test error. Friends dont 
    let friends use minibatches larger than 32. Let's face it: the only people have switched to minibatch sizes larger than one since 2012 is because GPUs are inefficient for batch sizes smaller than 32. That's a terrible reason. It just means our hardware sucks." 
    


<br>
That's about it for the logic. Here's the full training loop:

    --Setup--
    
    Instantiate model.

    Instantiate training and validation steps with model, loss function, and optimizer. 
    
    Move model to GPU.
    
        --Epoch--
    
        Set model to train mode after each post-epoch validation phase completes.
    
        Shuffle the training set for each  epoch, reset epoch loss and accuracy. 
        
            --Iteration--
    
            Create X_train, y_train minibatch tensors for each iteration and move tensors to GPU.
            
            Take 1 train step with X_train, y_train minibatch tensors.

            Aggregate accuracy and loss from each iteration, but only record after each epoch. 
            
        --Epoch--

        Compute and record validation accuracy for the entire epoch to keep track of learning progress. 

        Print training metrics after each epoch.
"""

# get training set size to calculate # iterations and minibatch indices
train_size = X_train.shape[0]

# pick minibatch size (of 32... always)
minibatch = 32

# set device to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device} selected')

# instantiate model and move to GPU for training
model = parallel_all_you_want(num_emotions=len(emotions_dict)).to(device) 
print('Number of trainable params: ',sum(p.numel() for p in model.parameters()) )

# encountered bugs in google colab only, unless I explicitly defined optimizer in this cell...
optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)

# instantiate the checkpoint save function
save_checkpoint = make_save_checkpoint()

# instantiate the training step function 
train_step = make_train_step(model, criterion, optimizer=optimizer)

# instantiate the validation loop function
validate = make_validate_fnc(model,criterion)

# instantiate lists to hold scalar performance metrics to plot later
train_losses=[]
valid_losses = []

# create training loop for one complete epoch (entire training set)
def train(optimizer, model, num_epochs, X_train, Y_train, X_valid, Y_valid):

    for epoch in range(num_epochs):
        
        # set model to train phase
        model.train()         
        
        # shuffle entire training set in each epoch to randomize minibatch order
        ind = np.random.permutation(train_size) 
        
        # shuffle the training set for each epoch:
        X_train = X_train[ind,:,:,:] 
        Y_train = Y_train[ind]

        # instantiate scalar values to keep track of progress after each epoch so we can stop training when appropriate 
        epoch_acc = 0 
        epoch_loss = 0
        num_iterations = int(train_size / minibatch)
        
        # create a loop for each minibatch of 32 samples:
        for i in range(num_iterations):
            
            # we have to track and update minibatch position for the current minibatch
            # if we take a random batch position from a set, we almost certainly will skip some of the data in that set
            # track minibatch position based on iteration number:
            batch_start = i * minibatch 
            # ensure we don't go out of the bounds of our training set:
            batch_end = min(batch_start + minibatch, train_size) 
            # ensure we don't have an index error
            actual_batch_size = batch_end-batch_start 
            
            # get training minibatch with all channnels and 2D feature dims
            X = X_train[batch_start:batch_end,:,:,:] 
            # get training minibatch labels 
            Y = Y_train[batch_start:batch_end] 

            # instantiate training tensors
            X_tensor = torch.tensor(X, device=device).float() 
            Y_tensor = torch.tensor(Y, dtype=torch.long,device=device)
            
            # Pass input tensors thru 1 training step (fwd+backwards pass)
            loss, acc = train_step(X_tensor,Y_tensor) 
            
            # aggregate batch accuracy to measure progress of entire epoch
            epoch_acc += acc * actual_batch_size / train_size
            epoch_loss += loss * actual_batch_size / train_size
            
            # keep track of the iteration to see if the model's too slow
            print('\r'+f'Epoch {epoch}: iteration {i}/{num_iterations}',end='')
        
        # create tensors from validation set
        X_valid_tensor = torch.tensor(X_valid,device=device).float()
        Y_valid_tensor = torch.tensor(Y_valid,dtype=torch.long,device=device)
        
        # calculate validation metrics to keep track of progress; don't need predictions now
        valid_loss, valid_acc, _ = validate(X_valid_tensor,Y_valid_tensor)
        
        # accumulate scalar performance metrics at each epoch to track and plot later
        train_losses.append(epoch_loss)
        valid_losses.append(valid_loss)
                  
        # Save checkpoint of the model
        checkpoint_filename = '/content/gdrive/My Drive/DL/models4/checkpoints/parallel_all_you_wantFINAL-{:03d}.pkl'.format(epoch)
        save_checkpoint(optimizer, model, epoch, checkpoint_filename)
        
        # keep track of each epoch's progress
        print(f'\nEpoch {epoch} --- loss:{epoch_loss:.3f}, Epoch accuracy:{epoch_acc:.2f}%, Validation loss:{valid_loss:.3f}, Validation accuracy:{valid_acc:.2f}%')

# choose number of epochs higher than reasonable so we can manually stop training 
num_epochs = 500

# train it!
train(optimizer, model, num_epochs, X_train, y_train, X_valid, y_valid)

"""# Check the Loss Curve's Behaviour
Let's see if we missed something egregious during training.
"""

plt.title('Loss Curve for Parallel is All You Want Model')
plt.ylabel('Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.plot(train_losses[:],'b')
plt.plot(valid_losses[:],'r')
plt.legend(['Training loss','Validation loss'])
plt.show()

"""Looks good. Past the \~78% validation accuracy, the model begins to overfit on the training set. Note that I ran the final model much longer than necessary (~20 minutes vs the 5 minutes it takes to train to convergence; trained on a K80 from Google Colab).

# Load the Trained Model from Checkpoint for Evaluation
"""

# pick load folder  
load_folder = '/content/gdrive/My Drive/DL/models/checkpoints'  

# pick the epoch to load
epoch = '429'
model_name = f'parallel_all_you_wantFINAL-{epoch}.pkl'

# make full load path
load_path = os.path.join(load_folder, model_name)

## instantiate empty model and populate with params from binary 
model = parallel_all_you_want(len(emotions_dict))
load_checkpoint(optimizer, model, load_path)

print(f'Loaded model from {load_path}')

"""# Evaluate the Model on Hold-Out Test Set
Fingers crossed for generalizability.
"""

# reinitialize validation function with model from chosen checkpoint
validate = make_validate_fnc(model,criterion)

# Convert 4D test feature set array to tensor and move to GPU
X_test_tensor = torch.tensor(X_test,device=device).float()
# Convert 4D test label set array to tensor and move to GPU
y_test_tensor = torch.tensor(y_test,dtype=torch.long,device=device)

# Get the model's performance metrics using the validation function we defined
test_loss, test_acc, predicted_emotions = validate(X_test_tensor,y_test_tensor)

print(f'Test accuracy is {test_acc:.2f}%')

"""Not too shabby. For some reference points: [A May 2020 paper](https://www.sciencedirect.com/science/article/abs/pii/S1746809420300501) claims to have achieved a "new SOTA" on RAVDESS with 71.61% accuracy on 8 emotions, while [another May 2020 paper](https://ieeexplore.ieee.org/document/9122698) claims to have achieved the "new SOTA" with an 90% F1 score on 8 emotions. Both use CNN architechtures. However, the paper reporting 90% F1 score is inflated by testing on training samples: [See the GitHub issue on this paper.](https://github.com/marcogdepinto/emotion-classification-from-audio-files/issues/11)

From the author with a 90% score: _"Also, the previous versions of this work used audio features extracted from the videos of the RAVDESS dataset. This particular part of the pipeline has been removed because it was shuffling very similar files in the training and test sets, boosting accuracy of the model as a consequence (overfitting)."_ 

The author has augmented the RAVDESS speech audio dataset with speech extracted from video data from RAVDESS - so the training data contained duplicate samples *before being split into training and test sets.* For reference, I achieved a 97% accuracy with this model when I similarly augmented the dataset with duplicate speech audio before splitting data. I corrected this issue by augmenting data *only after splitting into train/validation/test sets.*

This is an excellent example of "if it's too good to be true". More conservative estimates are more likely to be reproducible, in my opinion.

# Analyze Performance on Test Set
"""

from sklearn.metrics import confusion_matrix
import seaborn as sn

# because model tested on GPU, move prediction tensor to CPU then convert to array
predicted_emotions = predicted_emotions.cpu().numpy()
# use labels from test set
emotions_groundtruth = y_test

# build confusion matrix and normalized confusion matrix
conf_matrix = confusion_matrix(emotions_groundtruth, predicted_emotions)
conf_matrix_norm = confusion_matrix(emotions_groundtruth, predicted_emotions,normalize='true')

# set labels for matrix axes from emotions
emotion_names = [emotion for emotion in emotions_dict.values()]

# make a confusion matrix with labels using a DataFrame
confmatrix_df = pd.DataFrame(conf_matrix, index=emotion_names, columns=emotion_names)
confmatrix_df_norm = pd.DataFrame(conf_matrix_norm, index=emotion_names, columns=emotion_names)

# plot confusion matrices
plt.figure(figsize=(16,6))
sn.set(font_scale=1.8) # emotion label and title size
plt.subplot(1,2,1)
plt.title('Confusion Matrix')
sn.heatmap(confmatrix_df, annot=True, annot_kws={"size": 18}) #annot_kws is value font
plt.subplot(1,2,2)
plt.title('Normalized Confusion Matrix')
sn.heatmap(confmatrix_df_norm, annot=True, annot_kws={"size": 13}) #annot_kws is value font

plt.show()

"""Unsurprising results - the model has trouble differentiating between 'neutral' and 'calm', and between 'disgust' and 'angry'. 

If a human were asked to differentiate disgust and anger, getting 49/60 correct and 11/60 wrong wouldn't be too bad. 

Other predictable results include confusion of 'sad' for 'disgust'. It perhaps surprising that 'fearful' is confused for 'happy' as often as it is for 'sad' or 'disgust' - although fear is indeed a multifaceted emotion. 

Based on this, I would compare in finer detail the features of confused emotions and see if there are any differences at all - and how to capture them. For real-world data, it would be much more productive to perform sentiment analysis on the spoken words translated to text, and consider that in our final evaluation.

# Conclusion
---------------
Advances of the last 5 years involving upgrades to the autoencoder scheme have lead to the RNN, the upgraded LSTM-RNN, bidirectional LSTM-RNNs, and eventually to LSTM-RNNs with Attention layers to give profound temporal expressivity to the latent space of sequentially encoded data. The Transformer has built on this by taking advantage of parallelized self-attention layers to provide an almost truly global temporal representation of sequential data. 

Today, carefully thought out architecture building on these blocks leads to reasonable training times and excellent generalizability. We combine the CNN for spatial feature representation and the Transformer for temporal feature representation, and  augment the training data by increasing variation in the training dataset to reduce overfitting.

CNNs are still the standard for encoding representations of spatial data. A CNN's filters' kernel sizes are important to both performance and accuracy, especially considering recent paradigms using smaller maxpool kernels such as those of the 3x3 strided 1 kernels in [VGGNet](https://arxiv.org/pdf/1409.1556.pdf), in contrast to the 11x11 stride 4 kernels as in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

When we added convolutional and transformer layers beyond what was used here, that actually decreased test accuracy. Only as much complexity is warranted as is needed for generalizability. Although CNNs are good for images and transformers for sequential data, recently emerging paradigms (such as in this notebook) show that these networks are perfectly cross-applicable given careful thought. 

CNNs are powerful. Transformers work beautifully. They're even better together. Gone are the days of the LSTM-RNN.

If you got this far, I sincerely appreciate your taking the time to do so. Feel free to drop me a line at ilzenkov@gmail.com with any feedback or questions you may have.

# References 
--------------
- Ba et al, 2016. Layer Normalization. https://arxiv.org/abs/1607.06450
- Bahdanau et al, 2015. https://arxiv.org/pdf/1409.0473.pdf
- Cheng et al, 2016. Long Short-Term Memory-Networks for Machine Reading. https://arxiv.org/pdf/1601.06733.pdf
- He et al, 2015. Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
- Ioffe, Szegedy, 2015. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. https://arxiv.org/abs/1502.03167
- Krizhevsky et al, 2017. ImageNet Classification with Deep Convolutional Neural Networks. https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
- LeCunn et al, 1998. Gradient-Based Learning Applied to Document Recognition. http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
- Li et al, 2018. Visualizing the Loss Landscape of Neural Nets. https://papers.nips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets.pdf
- Masters and Luschi, 2018. Revisiting Small Batch Training for Deep Neural Networks. https://arxiv.org/abs/1804.07612 
- Peng et al, 2017. Large Kernel Matters —— Improve Semantic Segmentation by Global Convolutional Network. https://arxiv.org/pdf/1703.02719.pdf
- Santurkar et al, 2019. How Does Batch Normalization Help Optimization? https://arxiv.org/pdf/1805.11604.pdf
- Simonyan and Zisserman, 2015. Very Deep Convolutional Networks for Large-Scale Image Recognition. https://arxiv.org/pdf/1409.1556.pdf
- Srivastava et al, 2014. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
- Ulyanov et al, 2017. Instance Normalization: The Missing Ingredient for Fast Stylization. https://arxiv.org/pdf/1607.08022.pdf
- Vaswani et al, 2017. Attention Is All You Need. https://arxiv.org/abs/1706.03762
- Wilson et al, 2017. The Marginal Value of Adaptive Gradient Methods in Machine Learning. https://arxiv.org/abs/1705.08292

- Christopher Olah's Blog; Neural Networks, Types, and Functional Programming: http://colah.github.io/posts/2015-09-NN-Types-FP/
- Lilian Weng's blog on Attention Mechanisms: https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
- Stanford Autoencoder tutorial: http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/
- Stanford CNN Tutorial: http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/
- Stanford's CS231n: https://cs231n.github.io/convolutional-networks/
- U of T CSC2515, Optimization: https://www.cs.toronto.edu/~hinton/csc2515/notes/lec6tutorial.pdf

