This is a tensorflow wrapper for the LRP Toolbox which provides simple and accessible stand-alone implementations of LRP for artificial neural networks.

To run the given example 

   `cd mnist`
   
   `python mnist_conv.py --relevance_bool=True `
   
   `python mnist_linear.py --relevance_bool=True`
   
   `python mnist_ae.py --relevance_bool=True`
   

It downloads and extract the mnist datset, runs it on a neural netowrk and plots the relevances once the network is optimized. The relvances of the images can be viewed using

   
   `tensorboard --logdir=mnist_conv_logs`
   
   `tensorboard --logdir=mnist_linear_logs`
   
   `tensorboard --logdir=mnist_ae_logs`
   
