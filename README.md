Background
----------

This repository contains code written for the Kaggle competition "[ECML/PKDD 15: Taxi Trajectory Prediction](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i)".
This implementation, based on Tensorflow and Keras, is inspired from the approach followed by the competition's winners. More information about this code is provided on [this blog post]().

For more information about the original competition winner's solution, please refer to:
- The [summary](http://blog.kaggle.com/2015/07/27/taxi-trajectory-winners-interview-1st-place-team-%F0%9F%9A%95/) on Kaggle's blog.
- The [detailed published paper](https://arxiv.org/abs/1508.00021).
- The [original implementation](https://github.com/adbrebs/taxi) using Theano and Blocks.

Code structure
--------------

The code is comprised of three main files inside the `code` folder:

- [data.py](code/data.py): Methods for loading, cleaning and pre-processing the original datasets.
- [training.py](code/training.py): Methods for defining the neural network model and for running the training process.
- [utils.py](code/utils.py): Various mathematical and graphical utility functions.

Getting started
---------------

This implementation is based on Tensforflow version 0.11.0. The training process for the included neural network model
can be quite time-comsuming so it's recommended to use a GPU. A simple GPU-enabled Docker container setup can be found
at: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker

Some extra python libraries, specified in `requirements.txt` file, will also have to be installed.

Once your environment is set up, download the competition's [CSV data files](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data)
in the `datasets` folder.

To run the training process, run the following:

```python
    from code.training import full_train
    full_train(n_epochs=100, batch_size=200, save_prefix='mymodel')
```

The above will run the full training process and save some files to disk inside the `cache` folder:

- `mymodel-history.pickle` as
- 100 files (one for each epoch) named `mymodel-XXX.hdf5` (with `XXX` replaced with each epoch number).