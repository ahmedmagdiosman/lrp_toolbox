This is a tensorflow wrapper for the LRP Toolbox which provides simple and accessible stand-alone implementations of LRP for artificial neural networks.

To run the given example 

   `cd mnist`
   
   `python mnist_with_summaries.py --relevance_bool=True`
   

It downloads and extract the mnist datset, runs it on a neural netowrk and plots the relevances once the network is optimized. The relvances of the images can be viewed using

   `tensorboard --log_dir=../mnist_logs`
   
