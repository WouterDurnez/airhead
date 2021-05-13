![](docs/airhead_logo.png)
 
## Lightweight Convolutional Neural Networks for Brain Tumor Segmentation  

This repository hosts the code for my thesis at KULeuven, in order to obtain the degree of Master in Artificial Intelligence. Extensive boilerplate code was provided by my supervisor [Pooya Ashtari](https://www.kuleuven.be/wieiswie/nl/person/00129604).

### Components

* A module with helper functions (i.e. timestamped logging, setting global parameters, time functions, etc.).
* A specific instance of the **U-Net model**, based on Isensee, F., Jaeger, P. F., Full, P. M., Vollmuth, P., & Maier-Hein, K. H. (2020). _nnU-Net for Brain Tumor Segmentation_. 1–15. http://arxiv.org/abs/2011.00848. The model is not designed to be *overly* modular, but there is a set of parameters that can be freely selected, such as the number of filters in each level of the hierarchy, and whether to include the final activation layer (or train with logits).
* Lightning wrapper for the model (adding some useful functionality).
* Data module (Pytorch Lightning)
* Transforms for training, validation and test instances, based on [Monai](https://monai.io/).


### TODOs  
  
* Add **operational training script**  
  * (maybe) move parameters to dict-style **config file**  
  * (maybe) include **terminal-based** run-script (see example by Pooya)  
* Visualize some images & masks
* LIGHTWEIGHT LAYERS!