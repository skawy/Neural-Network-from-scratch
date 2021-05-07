#!/usr/bin/env python
# coding: utf-8

# ## Libraries 

# In[1]:


import math
import time 
import random 


# ## Creating Layer Class
# 

# In[2]:


class Layer:
    def __init__(self, input_size, output_size , activation_function):
        self.input = None
        self.output = []
        self.weights = []
        self.activation_function = activation_function
        ##nested loop to initailze the weigths with random values
        for i in range(input_size):
          col = []
          for j in range(output_size):
              col.append(random.random())
          self.weights.append(col)
        

    def forward_propagation(self, given_input  ):
        self.input = given_input
        self.output = []
        for col in range(len(self.weights[0])):
            sum = 0 
            for i in range(len(self.input) ):
                sum += self.input[i] * self.weights[i][col]
                
            if self.activation_function == "sigmoid" :
                self.output.append( 1/ (  1+(math.exp(-sum)) ))
            
            elif self.activation_function == "tanh" :
                self.output.append( math.tanh(sum))

        
        return self.output
        
        
    def back_propagation(self, previous_errors, learning_rate , targets):
        
        upcoming_error = []
        for i in range( len(self.input)):
            upcoming_error.append(0)

        
       #change the weights at the output layer
        if len(previous_errors) == 0 :
            for cur_neuron in range ( len(self.output) ) :
                delta_j = ( targets[cur_neuron] - self.output[cur_neuron] ) * self.output[cur_neuron]  * (1-self.output[cur_neuron])

                for upcoming_nueron in range( len( self.input) ) :
                    upcoming_error[upcoming_nueron] += delta_j * self.weights[upcoming_nueron][cur_neuron] 
                    self.weights[upcoming_nueron][cur_neuron] += ( delta_j * self.input[upcoming_nueron] * learning_rate)

      
        #change the weights at the hidden layer
        else :
                
            for cur_neuron in range ( len(self.output) ) :
                delta_j = previous_errors[cur_neuron] * self.output[cur_neuron]  * (1-self.output[cur_neuron]) 

                for upcoming_nueron in range( len ( self.input) ) :
                    
                    upcoming_error[upcoming_nueron] += delta_j * self.weights[upcoming_nueron][cur_neuron] 
                    self.weights[upcoming_nueron][cur_neuron] += delta_j * learning_rate * self.input[upcoming_nueron]
                    
                    
        return upcoming_error
            


# ## Creating Network Class

# In[3]:


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer ):
        self.layers.append(layer )


    def predict(self, input_data):
        
        predicted = []
        for i in range( len(input_data) ):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output )
            predicted.append(output)

        return predicted


    def fit(self, train_data, train_labels, epochs, learning_rate):
        
        samples = len(train_data)

        # training loop
        for epoch in range(epochs):
            
            for sample in range(samples):
        
                cur_deltas = []
                
                output = train_data[sample]
                
                for layer in self.layers:
                    output = layer.forward_propagation(output )


                for layer in reversed(self.layers):
                    cur_deltas = layer.back_propagation(cur_deltas, learning_rate , train_labels[sample])
                
                


# ### Testing With XOR Problem
# #### with tanh as an activation function for forwarding and sigmoid for backwarding
# 

# In[4]:


x_train = [ [0,0] , [0,1] , [1,0] , [1,1] ]
y_train = [[0], [1], [1], [0]]

# neural network init
NN = Network()
NN.add(Layer(2, 3 , "tanh"))
NN.add(Layer(3, 1 , "tanh"))

# train
NN.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
predicted = NN.predict(x_train)

print(predicted)


# ### Another Example Using Sigmod as an Activation Function
# 

# ### Points Belong To The First Class

# In[5]:


train_points_1 = [ [1.0 , 2.] , [ 11.,8. ] , [ 0.5 , 0.2 ] , [ 2. , 0.8 ] , [12. , 6. ] , [11. , 7.] ]
train_labels_1 = [ [0] , [0] , [0] , [0] , [0] , [0] ]


# ### Points Belong To Second Class

# In[6]:


train_points_2 = [ [1,7.5] , [10 , 2 ] , [ 3 , 8.5 ] , [ 2 , 9.2  ] , [11 , 1 ] , [9.6 , 3] ]
train_labels_2 = [ [1] , [1] , [1] , [1] , [1] , [1] ]


# In[7]:


# neural network init

NN_2 = Network()
NN_2.add(Layer(2, 4 , "sigmoid"))
NN_2.add(Layer(4, 4 , "sigmoid"))
NN_2.add(Layer(4, 1 , "sigmoid"))

# train
NN_2.fit(train_points_1, train_labels_1, epochs=100, learning_rate= 0.08)

# test

NN_2.predict(train_points_1)


# In[8]:


# train
NN_2.fit(train_points_2, train_labels_2, epochs=100, learning_rate= 0.08)


# In[9]:


# test
NN_2.predict(train_points_2)

