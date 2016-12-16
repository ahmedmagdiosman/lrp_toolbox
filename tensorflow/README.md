This is a tensorflow wrapper for the LRP Toolbox which provides simple and accessible stand-alone implementations of LRP for artificial neural networks.

This TF-wrapper considers the layers in the neural network to be in the form of a Sequence. A quick way to define a network would be

        net = Sequential([Linear(784,500, input_shape=(FLAGS.batch_size,784)), 
                     Relu(),
                     Linear(500, 100), 
                     Relu(),
                     Linear(100, 10), 
                     Softmax()]) 

This `net` can then be used to visualize the contributions of the input pixels towards the decision by

        relevance = net.lrp(y, 'simple', 1.0)

the different lrp variants available are:

        'simple','flat','w^2','epsilon' and 'alphabeta' 

To run the given example 

          
        python mnist/mnist_conv.py --relevance_bool=True 
   
        python mnist/mnist_linear.py --relevance_bool=True
   
        python mnist/mnist_ae.py --relevance_bool=True
   

It downloads and extract the mnist datset, runs it on a neural netowrk and plots the relevances once the network is optimized. The relvances of the images can be viewed using

   
        tensorboard --logdir=mnist_conv_logs
        
        tensorboard --logdir=mnist_linear_logs
   
        tensorboard --logdir=mnist_ae_logs
   
