# Active Learning experiments with histology image classification (Fork)

This repository was forked from https://github.com/dsgissin/DiscriminativeActiveLearning in order to carry out active 
learning experiments on histology images data for a class project. I added the necessary modifications in main.py and model.py in order 
to fetch and use the relevant images for my experiments.

## Histology datasets

In my experiment I used the [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
 and [BACH ICIAR 2018](https://iciar2018-challenge.grand-challenge.org/) datsets, which are both publicly available 
 breast histology datasets. I added the data/ directory to the project in order to organize the data.

To work with those datasets you must first download them, and then place the extracted directories as is in the data directory. 
Because I find the BreakHis directory hierarchy somewhat convoluted, I added the break_split.py script, which performs the train/test split 
for each magnification level and copies the files in the breakhis/test/ and breakhis/train/ directories. 
Because the structure of the iciar dataset is much simpler, I used the provided hierarchy as is.

In order to extract the patches for the BreakHis dataset I added the extract_patches_breakhis.ipynb notebook, which extracts patches
per image and then stores the matrices in the numpy array format in breakhis/numpy/ for each magnification while preserving the train/test split.
I also created the extract_patches_iciar.ipynb notebook, which performs the train/test split, extracts the patches, and then save the resulting numpy
arrays in the iciar/ directory.

Patches can be extracted with different "modes": single, five or ten. "Single" mode extracts a center crop, "five" extracts 
four corners and a center crop, "ten" is the same as "five" but augments the data by an horizontal flip. To use different modes, 
please see the usage of load_iciar and load_breakhis_from_np methods that were added in main.py. 

At the end of both notebooks you can pickle a numpy array containing the indices of an initial balanced labeled pool, which you can then 
pass as a parameter -idx to run the experiments. 

The data directory should have this structure after these operations.

```
.
├── data
│   ├── BreakHis_v1
│   │   └── histology_slides
│   │       └── breast
│   │           ├── benign
│   │           ├── malignant
│   │           └── ...
│   ├── ICIAR2018_BACH_Challenge
│   │   └── photos
│   │       ├── Benign
│   │       ├── InSitu
│   │       ├── Invasive
│   │       ├── Normal
│   │       └── ...
│   ├── breakhis
│   │   ├── numpy
│   │   |   ├── 100X
│   │   |   ├── 200X
│   │   |   ├── 400X
│   │   |   └── 40X
│   │   ├── test
│   │   |   ├── 100X
│   │   |   ├── 200X
│   │   |   ├── 400X
│   │   |   └── 40X
│   │   └── train
│   │       ├── 100X
│   │       ├── 200X
│   │       ├── 400X
│   │       └── 40X
│   ├── iciar
│   │   └── ...
│   └── ...
└── ...
```

I left the original readme unchanged below as everything in it still holds true. To use the histology datasets, simply use 
"breakhis" or "iciar" as the dataset command line parameter, like so:

```
python3 main.py 0 "breakhis" 100 100 20 "Random" "/path/to/experiment/folder" -idx "/path/to/folder/with/initial/index/file"
python3 main.py 0 "iciar" 100 100 20 "Random" "/path/to/experiment/folder" -idx "/path/to/folder/with/initial/index/file"
```

# Discriminative Active Learning (Original)

This repository contains the code used to run the deep active learning experiments detailed in our paper - https://arxiv.org/abs/1907.06347

You may use the code in this repository, but note that this isn't a complete active learning library and is not fully generic. Replicating the experiments and using the implementations should be easy, but adapting the code to new datasets and experiment types may take a bit of effort.

For the blog post detailing the thought process and relevant background for the algorithm, click [here](https://dsgissin.github.io/DiscriminativeActiveLearning/).

## Dependencies

In order to run our code, you'll need these main packages:

- [Python](https://www.python.org/)>=3.5
- [Numpy](http://www.numpy.org/)>=1.14.3
- [Scipy](https://www.scipy.org/)>=1.0.0
- [TensorFlow](https://www.tensorflow.org/)>=1.5
- [Keras](https://keras.io/)>=2.2
- [Gurobi](http://www.gurobi.com/documentation/)>=8.0 (for the core set MIP query strategy)
- [Cleverhans](https://github.com/tensorflow/cleverhans)>=2.1 (for the adversarial query strategy)

## Running the Code

The code is run using the main.py file in the following way:

    python3 main.py <experiment_index> <dataset> <batch_size> <initial_size> <iterations> <method> <experiment_folder> -method2 <method2> -idx <indices_folder> -gpu <gpus>

- experiment_index: an integer detailing the number of experiment (since usually many are run in parallel and combined later).
- dataset: a string detailing the dataset for this experiment (one of "mnist", "cifar10" or "cifar100").
- batch_size: the size of the batch of examples to be labeled in every iteration.
- initial_size: the amount of labeled examples to start the experiment with (chosen randomly).
- iteration: the amount of active learning iterations to run in the experiment.
- method: a string for the name of the query strategy to be used in the experiment.
- experiment_folder: the path of the folder where the experiment data and results will be saved.

There are also three optional parameters:
- idx: a path to the folder with the pickle file containing the initial labeled example indices for the experiment.
- method2: the name of the second query strategy (if you want to try and combine two methods together).
- gpu: the number of gpus to use for training the models.

### Possible Method Names
These are the possible names of methods that can be used in the experiments:
- "Random": random sampling
- "CoreSet": the greedy core set approach
- "CoreSetMIP": the core set with the MIP formulation
- "Discriminative": discriminative active learning with raw pixels as the representation
- "DiscriminativeAE": discriminative active learning with an autoencoder embedding as the representation
- "DiscriminativeLearned": discriminative active learning with the learned representation from the model as the representation
- "DiscriminativeStochastic": discriminative active learning with the learned representation as the representation and sampling proportionally to the confidence as being "unlabeled".
- "Uncertainty": uncertainty sampling with minimal top confidence
- "UncertaintyEntropy": uncertainty sampling with maximal entropy
- "Bayesian": Bayesian uncertainty sampling with minimal top confidence
- "BayesianEntropy": Bayesian uncertainty sampling with maximal entropy
- "EGL": estimated gradient length
- "Adversarial": adversarial active learning using DeepFool


## Directory Structure

### main.py

This file contains the logic which runs the active learning experiment and saves the results to the relevant folder.

### models.py

This file contains all of the neural network models and training functions used by the query methods.

### query_methods.py

this file contains the query strategy implementations for all of the methods detailed in the blog.

## Examples


    python3 main.py 0 "mnist" 100 100 20 "Random" "/path/to/experiment/folder" -idx "/path/to/folder/with/initial/index/file"
    python3 main.py 7 "cifar10" 5000 5000 5 "DiscriminativeLearned" "/path/to/experiment/folder" -idx "/path/to/folder/with/initial/index/file"
    python3 main.py 0 "cifar100" 5000 5000 3 "Adversarial" "/path/to/experiment/folder" -method2 "Bayesian" -gpu 2

