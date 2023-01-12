![](docs/airhead_logo.png)
 
## Lightweight Convolutional Neural Networks for Brain Tumor Segmentation  

This repository hosts the code for my thesis at KULeuven, in order to obtain the degree of Master in Artificial Intelligence. Working title: Airhead (because we're dealing with *lightweight* networks for *brain* tumor segmentation, get it? ...I'll see myself out.).

Promotor: [prof. dr. ir. Sabine Van huffel](https://www.esat.kuleuven.be/stadius/person.php?id=17)

Copromotor: [prof. dr. ir. Frederik Maes](https://www.kuleuven.be/wieiswie/nl/person/00007203)

Supervisor: [Pooya Ashtari](https://www.kuleuven.be/wieiswie/nl/person/00129604).
[Check out his GitHub! 🙂](https://github.com/pashtari)

### Components

#### 0. main scripts

* `flops.py` -- FLOP/MAC counting script; also counts network parameters.
* `hpc_generate.py` -- Generates bash scripts that submit a large number of jobs simultaneously.
* `playground.py` -- Scratch script, used to create several plots for chapter 2.
* `train_baseline.py` -- Training script for baseline network.
* `test_baseline.py` -- Testing script for baseline network (load checkpoint and only execute test phase).
* `train_air.py` -- Training script for tensorized network (takes compression rate parameter and tensor network format parameter)
* `test_air.py` -- Testing script for tensorized network (load checkpoint and only execute test phase)


#### 1. [utils](utils/)

* `helper.py` -- Some helper functionality to make my life easier (i.e. timestamped logging, setting global parameters, time functions, etc.).
* `utils.py` -- Utility functions that are used for layer building (e.g., determining tuning parameters based on compression rates).

#### 2. [layers](layers/)

Implementations of low-rank convolutional layers.

* `air_conv.py` -- Implementation of low-rank principle, for various tensor network formats. Includes `comp_friendly` parameter for optimized training (WARNING: don't count FLOPS using optimized computation!)

#### 3. [models](models/)

* `baseline_unet.py` -- A specific instance of the *U-Net model*, based on Isensee, F., Jaeger, P. F., Full, P. M., Vollmuth, P., & Maier-Hein, K. H. (2020). _nnU-Net for Brain Tumor Segmentation_. 1–15. http://arxiv.org/abs/2011.00848. The model is not designed to be *overly* modular, but there is a set of parameters that can be freely selected, such as the number of filters in each level of the hierarchy, and whether to include the final activation layer (or train with logits).
* `air_unet.py` -- A lightweight variant of the baseline network, in which convolutions (in the double convolution blocks) are substituted with low-rank layers of our choosing. Other parameters are identical.

#### 4. [training](training/)

* `data_module.py` -- Custom data module for the BraTS dataset.
* `inference.py` -- Functionality related to model inference, for the test phase and to predict single instances (for visualization).
* `lightning.py` -- Lightning wrapper for all U-Net models, to simplify training.
* `losses.py` -- Loss functions and other measures/metrics.
* `transforms` -- Data transforms (preprocessing/augmentation) for training, validation/testing, and visualization, based on [Monai](https://monai.io/).

#### 5. [evaluation](evaluation/)

* `analysis.py` -- Creating necessary data frames from output files, plot results.
* `stats.py` -- Statistical tests (TNNs versus baseline).

#### 6. [docs](docs/)

My silly logo, and final version of the **[thesis](docs/mai2021_r0745527_wouter_durnez.pdf)**.
