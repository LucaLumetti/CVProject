## People
The people involved in the project are:
- Luca Lumetti (244577@studenti.unimore.it)
- Matteo Di Bartolomeo (241469@studenti.unimore.it)
- Federico Silvestri (243938@studenti.unimore.it)

## Overview

Our project aims to remove face masks over a person’s face, by reconstructing
the covered part of the face. To have a more precise reconstruction of the missing
parts (mouth and nose) behind the mask, we plan to use a second photo of the
same person without the mask as a reference during the facial reconstruction
process. There are no constraints on the quality of the reference photo, for
instance the face can be taken from a different point of view than the first one.
To sum up, given as input an image containing a person’s face partially covered
by a medical mask and another photo of the same person without any occlusions,
the output will be the first image with the mask-covered parts, mouth and nose,
reconstructed.
Future development could lead to generalizing the occlusion caused by the mask
to any type of occlusion possible.

## Pipeline
The pipeline will be structured as follows:
- Detect the mask in the first image by using classical image-processing operators, like edge detection and/or segmentation algorithms.
- Fix face orientation in the reference image by using a geometric-based algorithm, like the thin-plate spline transformation.
- Reconstruct the missing parts of the face in the first image by using a GAN network with a contrastive learning approach. (Deep learning network with a retrieval component)

## TODO
- [x] Weights should be initialized with Xavier (or similar)
- [ ] Write the script to test the network (test.py)
- [ ] Metrics analisys in training.py could be rewritter in a separated class
- [ ] It would be cool to test the network on the same image while training, in
  order to compare the GAN at different epochs
- [ ] Self attention layer generate CUDA out of memory
- [ ] Low number of channels in generator and discriminator always bc of CUDA out of
  memory
- [ ] Give the possibility to resume a training using the saved model as
- [ ] Give the option to trasform images to 512x512 and 256x256 before training
  starting point
