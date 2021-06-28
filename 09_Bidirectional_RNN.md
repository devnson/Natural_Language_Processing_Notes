# Bidirectional RNN


**What are Bidirectional Recurrent Neural Network?**

Bidirectional recurrent neural networks (BRNN) connects two hidden layers running in opposite directions to a single output, allowing them to receive informationo from both past and future states. This generative deep learning technique is more common in supervised learning approaches, rather than unsupervised or semi-supervised because how difficult it is to calcualte a reliable probabilistic model.

**How are Bidirectional Recurrent Neural Networks Trained?**

BRNNs are trained with similar algorithms as RNNs, since the two directional neurons do not interact with one another. If back-propagation is necessary some additional process is needed, since input and output layers cannot both be updated at once.

In general training, forwared and backward states are processed first in the "forward" pass, before output neurons are passed. For the backward pass, the opposite takes place: output neurons are processed first, then forward and backward states are passed next. Weights are updated only after the forward and backward passes are complete.

**What's the difference between BRNN's and Recurrent Neural Networks?**

Unlike standard recurrent neural networks, BRNN's are trained to predict both the positive and negative directions of time simultaneously. BRNN's split the neurons of a regualr RNN into two directions, one for the forward states (positive time direction), and another for the backward states (negative time direction) Neither of these output states are connected to inputs of the opposite directions. By employing two time directions simultaneously, input data from the past and future of the current time frame can be used to calculate the same output. Which is the opposite of standard recurrent networks that required and extra layer for including future information.

## Understanding Bidirectional RNN

![1_6QnPUSv_t9BY9Fv8_aLb-Q](https://user-images.githubusercontent.com/23405520/116771140-b5e4f800-aa66-11eb-81e5-f256fe67f448.png)

Bidirectional RNN are really just putting two independent RNNs together. The input sequence is fed in normal time order for one network, and in reverse time order for another. The outputs of the two networks are usually concatenated at each time step, through there are other options. eg summation.

The strucutre allows the networks to have both backward and forward information about the sequence at every time step. The concept seems easy enough. But when it comes to actually implementing a neural network which utilizes bidirectional structure, confusioni arises.

### The Confusion

The first confusioni is about the way to forward the outputs of a bidirectional RNN to a dense neural network. For normal RNNs we could just forward the ouputs at the last time step, and the following picture. 

![1_GRQ91HNASB7MAJPTTlVvfw](https://user-images.githubusercontent.com/23405520/116771230-520eff00-aa67-11eb-9845-9fda6aba9317.jpeg)


But wait.....if we pick the output at the last time step, the reverse RNN will have only seen the last input (x3 in the picture). It'll hardly provide any predictive power.

The second confusion is about the **returned hidden states.** In seq2seq models, we'll want hidden states from the encoder to initialize the hidden states of the decoder. Intuitively, if we can only chooose hidden states at one step (as in PyTorch), we'd want the one at which the RNN just consumed the last input iinit the sequence. But if the hidden states of time step n (the last one) are returned, as before, we'll have the hidden states of the reversed RNN with only one step of inputs seen.

