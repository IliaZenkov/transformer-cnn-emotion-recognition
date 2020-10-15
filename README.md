
# Parallel is All You Want: Combining Spatial and Temporal Feature Representions of Speech Emotion by Parallelizing CNNs and Transformer-Encoders
# [See Notebook for Code and Explanations](https://nbviewer.jupyter.org/github/IliaZenkov/transformer_cnn_parallel_audio_classification/blob/main/parallel_is_all_you_want1.ipynb)

<img src="reports/cnn-transformer-final.png">
<img src="nicely initialized weights.GIF">

# Abstract
In this notebook, I'm going to build upon my Intro to Speech Audio Classification repo and build two parallel convolutional neural networks (CNN) in parallel with a Transformer encoder network to classify audio data. We're working on the RAVDESS dataset to classify emotions from one of 8 classes. We combine the CNN for spatial feature representation and the Transformer for temporal feature representation. We augment the training data by increasing variation in the dataset to reduce overfitting; we use Additive White Gaussian Noise (AWGN) to augment the RAVDESS dataset three-fold for a total of 4320 audio samples.

We harness the image-classification and spatial feature representation power of the CNN by treating mel spectrograms as grayscale images; their width is a time scale, their height is a frequency scale. The value of each pixel in the mel spectrogram is the intensity of the audio signal at a particular mel frequency at a time step.

Because of the sequential nature of the data, we will also use the Transformer to try and model as accurately as possible the temporal relationships between pitch transitions in emotions.

This notebook takes inspirations from a variety of recent advances in deep learning and network architectures; in particular, stacked and parallel CNN networks combined with multi-head self-attention layers from the Transformer Encoder. I hypothesize that the expansion of CNN filter channel dimensions and reduction of feature maps will provide the most expressive feature representation at the lowest computaitonal cost, while the Transformer-Encoder is used with the hypothesis that the network will learn to predict frequency distributions of different emotions according to the global structure of the mel spectrogram of each emotion. With the strength of the CNN in spatial feature representation and Transformer in sequence encoding, I manage to achieve a 97.01% accuracy on a test set from the RAVDESS dataset.

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

# Appendices
- [Appendix A: Convolutional Neural Nets Dissected](#Appendix-A---Convolutional-Neural-Nets-Dissected)
  - [Kernels and Filters](#Kernels-and-Filters)
  - [Zero Padding](#Zero-Padding)
  - [Max Pooling](#Max-Pooling)
  - [Regularization Using Dropout and a note on Pruning](#Regularization-Using-Dropout-and-a-note-on-Pruning)
  - [Batch Normalization and Optimizing the Optimization Landsape](#Batch-Normalization-and-Optimizing-the-Optimization-Landsape)
  - [ReLU: Non-Saturated Activations are Healthier (for CNNs)](#ReLU---Non-Saturated-Activations-are-Healthier-[for-CNNs])
  - [Turning Feature Maps into Probabilities with a Fully Connected Layer](#Turning-Feature-Maps-into-Probabilities-with-a-Fully-Connected-Layer)
- [Appendix B: The Transformer](#Appendix-B---The-Transformer)
    - [Self-Attention](#Self-Attention)
    - [Multi-Head Self-Attention](#Multi-Head-Self-Attention)
    - [Transformer Architecture](#Transformer-Architecture)
        - [Input](#Input)
        - [Encoder](#Encoder)
        - [Decoder](#Decoder)
        - [Output](#Output)
- [Appendix C: From Autoencoders to LSTMs to Attention](#Appendix-C---From-Autoencoders-to-LSTMs-to-Attention)
  - [Autoencoders](#Autoencoders)
  - [The Sparse Autoencoder](#The-Sparse-Autoencoder)
  - [The Variational Autoencoder, Decoders, and Latent Space](#The-Variational-Autoencoder,-Decoders,-and-Latent-Space)
  - [The RNN, Seq2Seq, and Gradient Problems](#The-RNN,-Seq2Seq,-and-Gradient-Problems)
  - [The LSTM Cell](#The-LSTM-Cell)
  - [Bidirectional RNN](#Bidirectional-RNN)
  - [The Attention Mechanism](#The-Attention-Mechanism)
- [Appendix D: Supplementary Notes](#Appendix-D---Supplementary-Notes)
  - [More CNN Kernel/Filter Math](#More-CNN-Kernel/Filter-Math)
  - [Smoothing the Optimization Surface](#Smoothing-the-Optimization-Surface)
