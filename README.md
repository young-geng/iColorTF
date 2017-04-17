iColorTF
=============

Tensorflow implementation of the iColor user guided image colorization network.

## Setup

### Prerequisites
- Linux
- [Anaconda](https://www.continuum.io/downloads)

### Environment Setup
- Install the required packages by running `setup.sh`.
```bash
./setup.sh
```
- Activate conda environment.
```bash
source activate icolor_tf
```

## Training
- Run `train.sh` to train the model.
- Edit the `train.sh` to set the data source and appropriate batch size.
    - The image list should be a file containing a path to an image in each line,
      and the path will be concatenated with the image root to be the full path for accessing images.
    - On a Titan X with 12GiB memory, use batch size 24.
    - On a Tesla P100 with 16GiB memory, use batch size 32.
    - The training output will be stored in the `output` directory, unless
      otherwise specified.

