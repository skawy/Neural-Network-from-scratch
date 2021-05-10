# Neural-Network-From-Scratch

This is an implementation for neural network without using numpy or any data science or machine learning library.

## Neural Network Structure 

<img src="https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork-WHITEBG.png" alt="What are Neural Networks? | IBM" style="zoom:50%;" />

**Neural Network is A Network of fully connected layers:**

> First layer is the input layer
>
> Last layer is the output layer
>
> in between is just the hidden layers 

- Every Node in The layer is called *Neuron*

We Calculate the Neurons in the hidden layers to the output layers by multiplying every neuron   in the previous layer for its weights connected to this neuron   

<img src="https://i.stack.imgur.com/gzrsx.png" alt="img" style="zoom:67%;" />

### But there are something missing 

- We Cant Solve The problems with this equation because the output is just will be linear 

- then we will apply  non linearity function to every neuron in the layer

  >  In this code it is the *sigmoid*

![Ldapwiki: Sigmoid function](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRuvCWELjFqOlrzWjNJsKBuNL80xEVlb2eJyw&usqp=CAU)



So We will Calculate all neurons through all layers till we get the final layer which is the output layer and calculate the output 

## .

### ..

#### ...



### Is This Output will be accurate with our weights ?

**nope**

 

At First Iteration we initialized the weights at every layer randomly and by applying our activation function to every neuron we will get some output but its not the accurate output 

- For the training after calculate the output we will compare it with the real output for this input 
- we will calculate the error by Mean squared error <img src="https://siegel.work/blog/LossFunctions/img/MSE_Formula.svg" alt="siegel.work - Loss Functions" style="zoom:50%;" />
- this error will help us to modify all weights from the output layer to the input layer **that is why we are calling it backward propagation** 
- the whole Iteration of Forwarding till the output layer and Backwarding with changing the weights it is called an *Epoch* 


## Perfect References To Understand Neural Network 

1. https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65 
   
   *English*

2. https://www.youtube.com/watch?v=twOGEhgYuoI&list=PL5JZLxl_tFCczr7SDjVxZce9lw7ESsaQM&index=5

   *Arabic* 
