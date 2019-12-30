# SpaceNet5 and xView2 competition code

Code used when competing in the SpaceNet5 and Xview2 challenges; chronological order was SpaceNet5
first, and that code was modified for Xview2 afterward.  See individual commits if desired, as the
resulting code in the master branch is Xview2-specific and there are no releases (code was meant
for internal use only).

i.e. this repo houses code that generates solutions for two specific semantic segmentation and
object classification tasks.

## Xview2 challenge summary
Xview2, sponsored by the Defense Innovation Unit, sought an automated method of identifying
damaged buildings after natural disasters using only satellite imagery.

The given samples in this case were a pair of pre-disaster and post-disaster satellite images,
and the required output was a pair of grayscale masks classifying each pixel in the input images
that represented a building, as well as the level of damage sustained by said building (if any).

[Xview2 Challenge site](https://xview2.org)


## SpaceNet5 challenge summary
SpaceNet5, sponsored by Cosmiq Works et al., sought an automated method of identifying damaged
road networks so that first responders could reach necessary areas efficiently after natural
disasters, also using only satellite imagery.

The given samples for this one were single satellite images of road networks.  The required
output was a set of coordinates representing a graph of the usable roads in the input image, as
well as the speed at which those roads could be safely traveled.

[Cosmiq Works](https://www.cosmiqworks.org)

------

## Project overview
The model takes RGB input images (satellite imagery) and outputs a one hot encoded
`(BATCH_SIZE, HEIGHT * WIDTH, NUM_CLASSES)` array so that standard categorical crossentropy
could be used for training.

The project wasn't written to serve as a basis for derivative works, but if you find it useful,
it's released under a BSD license and you're free to incorporate it or build on it as you please.

## File descriptions

### The models
- unet.py: Version of Motokimura's winning SpaceNet model (Unet style fully convolutional network)
- deeplabmodel.py: DeepLabV3+ model in Keras by [Bonlime](https://github.com/bonlime/keras-deeplab-v3-plus)

### Using the models
- settings.py: Global settings for the project (neural net input shape, mask shape, and so forth)
- train.py: Code for building and training a semantic segmentation model
- infer.py: A few functions for converting neural net output => displayable images
- score.py: Metrics for training and evaluating
- test.py: Generate solution files (PNG image data, 1024 x 1024, 8-bit grayscale, non-interlaced)
- scansolution.py: Counts pixel values in solution files (sanity checks)

### The dataset (see the xBD paper for details)
- flow.py: All the code for the dataset (data objects, a few image deformations, etc.)
- show.py: Code for displaying input samples and model predictions
- damage.py: some miscellaneous code related to damaged buildings
- mkbuildings.py: A little script to dump individual buildings from input samples to their own files

------

### Development environment (the only platform it was really tested on):
- Linux v5.x
- python v3.7.4
- tensorflow v2.0
- all required packages installed via `pip3 install`
