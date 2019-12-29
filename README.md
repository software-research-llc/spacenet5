# SpaceNet5 and xView2 challenge code

Code used for the SpaceNet5 and Xview2 challenges; chronological order was SpaceNet5 first,
and that code was modified for Xview2 afterward.  See individual commits if desired, as the
resulting code in the master branch is Xview2-specific and there are no releases.

------

## Overview
The model takes RGB input images (satellite imagery) and outputs a one hot encoded
(BATCH_SIZE, HEIGHT * WIDTH, NUM_CLASSES) array so that standard categorical crossentropy
can be used for training.

The project wasn't written to serve as a basis for derivative works, but if you find it useful,
it's released under a BSD license and you're free to incorporate it or build on it as you please.

## File descriptions

### The models
- unet.py: A derivation of Motokimura's winning SpaceNet model (a Unet style fully convolutional network)
- deeplabmodel.py: Implementation of DeepLabV3+ model in Keras

### Using the models
- settings.py: Global settings for the project (neural net input shape, mask shape, and so forth)
- train.py: Code for building and training a semantic segmentation model
- infer.py: A few functions for converting neural net output => displayable images
- score.py: Metrics for training and evaluating
- test.py: Generate solution files (PNG image data, 1024 x 1024, 8-bit grayscale, non-interlaced)
- scansolution.py: Counts pixel values in solution files

### The dataset (see the xBD paper for details)
- flow.py: All the code for the dataset (data objects, a few image deformations, etc.)
- show.py: Code for displaying input samples and model predictions
- damage.py: some miscellaneous code related to damaged buildings
- mkbuildings.py: A little script to dump individual buildings from input samples to their own files

------

### Development environment:
- Linux v5.x
- python v3.7.4
- tensorflow v2.0
- all required packages installed via `pip3 install`
