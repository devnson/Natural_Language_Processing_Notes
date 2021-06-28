# Recurrent Neural Network

Recurrent Neural Network (RNN) are a type of Neural Network where the output from previous step are fed as input to the current step. In traditional neural networks, all the inputs and outputs are independent of each other, but in cases like when it is required to predict the next word of a sentence, the previous words are required and hence there is a need to remember the previous word. Thus RNN cam into existence, which solved this issue with the help of a Hidden Layer. The main and most important feature of RNN is Hidden state, which remembers some information about a sequence.

### Need for a Nerual Network dealing with Sequences
The beauty of Recurrent Neural Network lies in their diversity of application. When we are dealing with RNNs they have a great ability to deal with various input and output types.

- <b> Sentiment Classification </b> : This can be a task of simply classifying tweets into positive and negative sentiment. So here the input would be a tweet of varying lengths, while output is of a fixed type and size.

![sentiment-768x202](https://user-images.githubusercontent.com/23405520/115370891-1c585380-a1e7-11eb-8e0e-54f3ed954f5e.png)

- <b> Image Captioning </b> : Here, let's say we have an image for which we need a textual description. So we have a single input - the image, and a series or sequence of words as output. Here the image might be of a fixed size, but the output is a description of varying lengths.

![image-captioning](https://user-images.githubusercontent.com/23405520/115371101-50cc0f80-a1e7-11eb-9cd7-e293b17d5288.png)

- <b> Language Translation </b> : This basically means that we have some text in a particular language let's say English, and we wish to translate it in French. Each language has it's own semantics and would have varying lengths for the same sentence. So here the inputs as well as outputs are of varying lengths.

![English-to-french](https://user-images.githubusercontent.com/23405520/115371353-87a22580-a1e7-11eb-9f22-373aed3c64bf.png)

### RNN vs ANN

RNN's and feed forward neural networks get their names from the way they channel information.

In a feed-forward neural network network, the information only moves in one direction - from the input layer, through the hidden layers, to the output layer. The information moves straight through the network and never touches a node twice.

Feed-forward neural networks have no memory of the input they receive and are bad at predicting what's coming next. Because a feed-forward network only considers the current input, it has no notion of order in time. It simply can't remember anyting about what happend in the past except its training.

In a RNN the information cycles through a loop. When it makes a decision, it considers the current input and also what it has learned from the inputs it received previously.

The two images below illustrates the difference in information flow between a RNN and a feed forward neural network.

![rnn-vs-fnn](https://user-images.githubusercontent.com/23405520/115372270-570ebb80-a1e8-11eb-8236-92392f2de78c.png)


A usual RNN has a short-term memory. In combination with a LSTM they also have a long-term memory (more on that later).

Another good way to illustrate the concept of a recurrent neural network's memory is to explain it with an example.

Imagine you have a normal feed-forward neural network and give it the word "neuron" as an input and it processes the word character by character. By the time it reaches the character "r", it has already forgotten about "n","e" and "u", which makes it almost impossible for this type of neural network to predict which character would come next.

A recurrent neural network, however is able to remember those characters because of its internal memory. It produces output, copies that output and loops it back into the network.

<b> Simply put: recurrent neural networks add the immediate past to the present </b>

Therefore, a RNN has two inputs: the present and the recent past. This is important because the sequence of data contains crucial information about what is coming next, which is why a RNN can do things other algorithms can't

A feed-forward neural network assigns, like all other deep learning algorithms, a weight matrix to its input and then produces the output. Note that RNNs apply weights to the current and also to the previous input. Furthermore, a recurrent neural network will also tweak the weights for both through  gradient descent and backpropagation through time (BPTT).

Also note that while feed-forward neural networks map one input to one output, RNNs can map one to many, many to many (translation) and many to one (classifying a voice).

![Feed-Forward-Neural-Networks](https://user-images.githubusercontent.com/23405520/115373203-4874d400-a1e9-11eb-9e26-6e3e1db4ca5e.png)

### BACKPROPAGATION THROUGH TIME
To understand the concept of backpropagation through time you'll need to understand the concepts of forward and backpropagation first. We could spend an entire article discussing these concepts, so I will attempt to provide as simple a definition as possible. 

In neural networks, you basically do forward-propagation to get the output of your model and check if this output is correct or incorrect, to get the error. Backpropagation is nothing but going backwards through your neural network to find the partial derivatives of the error with respect to the weights, which enables you to subtract this value from the weights.

Those derivatives are then used by gradient descent, an algorithm that can iteratively minimize a given function. Then it adjusts the weights up or down, depending on which decreases the error. That is exactly how a neural network learns during the training process.

So, with backpropagation you basically try to tweak the weights of your model while training.

The image below illustrates the concept of forward propagation and backpropagation in a feed-forward neural network:

![propagatiom-rnn](https://user-images.githubusercontent.com/23405520/115373312-60e4ee80-a1e9-11eb-907c-433c3d51afc3.png)

BPTT is basically just a fancy buzz word for doing backpropagation on an unrolled RNN. Unrolling is a visualization and conceptual tool, which helps you understand what’s going on within the network. Most of the time when implementing a recurrent neural network in the common programming frameworks, backpropagation is automatically taken care of, but you need to understand how it works to troubleshoot problems that may arise during the development process.

You can view a RNN as a sequence of neural networks that you train one after another with backpropagation.

The image below illustrates an unrolled RNN. On the left, the RNN is unrolled after the equal sign. Note there is no cycle after the equal sign since the different time steps are visualized and information is passed from one time step to the next. This illustration also shows why a RNN can be seen as a sequence of neural networks.

![unrolled-rnn_0](https://user-images.githubusercontent.com/23405520/115373360-6d694700-a1e9-11eb-9ff2-7cf3cbc7fd24.png)


If you do BPTT, the conceptualization of unrolling is required since the error of a given timestep depends on the previous time step.

Within BPTT the error is backpropagated from the last to the first timestep, while unrolling all the timesteps. This allows calculating the error for each timestep, which allows updating the weights. Note that BPTT can be computationally expensive when you have a high number of timesteps.


## Understanding a Recurrent Neuron in Detail

Let's take a simple task at first. Let's take a character level RNN where we have a word "Hello". So we provide the first 4 letters i.e h,e,l,l and ask the network to predict the last letter i.e 'o'. So here the vocabulary  of the task is just 4 letters {h,e,l,o}. In real case scenarios involving natural langauge processing, the vocabularies include the words in entire wikipedia database, or all words in a language. Here for simplicity we have taken a very small set of vocabulary.

![rnn-neuron-196x300](https://user-images.githubusercontent.com/23405520/115373965-0009e600-a1ea-11eb-875d-afb29f626fcc.png)

Let's see how the above structure be used to predict the fifth letter in the world "hello". In the above structure, the blue RNN block, applies something called as a recurrence formula to the input vector and also its previous state. In this case, the letter "h" has nothing preceding it, let's take the letter "e". So at the time the letter "e"  is supplied to the network, a recurrence formula is applied to the letter "e" and the previous state which is the letter "h". These are known as various time steps of the input. So if at time t, the input is "e", at time t-1, the input was "h". The recurrence formula is applied to e and h both, and we get a new state.

The formula for the current state can be written as:

![hidden-state](https://user-images.githubusercontent.com/23405520/115374431-773f7a00-a1ea-11eb-9ed4-88c8d44ed10b.png)

Here, 
- ht is the new state.
- ht-1 is the previous state while 
- Xt is the current input.

We now have a state of the previous input instead of the input itself, because the input neuron would have applied the transformations on our previous input. So each successive input is called as a time step.

In this case we have four inputs to be given to the network, during a recurrence formula, the same function and the same weights are applied to the network at each time step.

Taking the simplest form of a recurrent neural network, let's say that the activation function is `tanh`, the weight at the recurrent neuron is `Whh` and the weight at the input neuron is `Wxh`, we can write the equation for the state at time `t` as:

![eq2](https://user-images.githubusercontent.com/23405520/115375016-ffbe1a80-a1ea-11eb-9ab3-25d0cdd63bf2.png)

The Recurrent neuron in this case is just taking the immediate previous state into consideration. For longer sequences the equation can involve multiple such states. Once the final state is calculated we can go on to produce the output.

Now, once the current state is calculated we can calculate the output state as:

![outeq](https://user-images.githubusercontent.com/23405520/115375201-3005b900-a1eb-11eb-8fe6-055138056e67.png)

## Forward Propagation in a Recurrent Neuron in Excel

Let's take a look at the inputs first:

![inputs](https://user-images.githubusercontent.com/23405520/115375328-4f9ce180-a1eb-11eb-8238-9f894a531c4c.png)

The inputs are one hot encoded. Our entire vocabulray is {h,e,l,o} and hence we can easily one hot encode the inputs.

Now the input neuron would transform the input to the hidden state using the weight `w * h`. We have randomly initialized the weigths as a `3 * 4` matrix:

![wxh](https://user-images.githubusercontent.com/23405520/115375534-84a93400-a1eb-11eb-9d88-8b9efed383ab.png)

<b> Step 1: </b>
Now for the letter "h", for the hidden state we would need `W * h * Xt`. By matrix multiplication, we get it as:

![first-state-h](https://user-images.githubusercontent.com/23405520/115375674-a5718980-a1eb-11eb-8d5b-294e3085c7f0.png)

<b> Step 2: </b>

Now moving to the recurrent neuron, we have `Whh` as the weight which is a `1*1` matrix as ![WHH](https://user-images.githubusercontent.com/23405520/115375819-c6d27580-a1eb-11eb-93ae-d4c402ebf1e6.png) and the bias which is also a `1 * 1` matrix as ![bias](https://user-images.githubusercontent.com/23405520/115375896-d782eb80-a1eb-11eb-925c-aabe27594b6a.png)

For the letter 'h', the previous state is [0,0,0] since there is no letter prior to it.

So to calculate -> (whh * ht - 1 + bias)

![WHHT-1-1](https://user-images.githubusercontent.com/23405520/115376036-fa150480-a1eb-11eb-93fc-1823b0d91c00.png)


<b> Step 3: </b>

Now we can get the current state as:

![eq21](https://user-images.githubusercontent.com/23405520/115376111-08fbb700-a1ec-11eb-9017-eacf4b1f27de.png)

Since for h, there is no previous hidden state we apply the tanh function to this output and get the current state -

![ht-h](https://user-images.githubusercontent.com/23405520/115376193-1e70e100-a1ec-11eb-8324-674b0e17979a.png)

<b> Step 4: </b>

Now we go on to the next state. "e" is now supplied to the network. The processed output of `ht`, now becomes `ht-1`, while the one hot encoded `e`, is `xt`. Let's now calculate the current state `ht`

![eq2 (1)](https://user-images.githubusercontent.com/23405520/115376417-4f511600-a1ec-11eb-9472-2705c7b8b50a.png)


`Whh * ht - 1 + bias` will be

![new-ht-1](https://user-images.githubusercontent.com/23405520/115376501-60018c00-a1ec-11eb-8d5b-3ec47e62cdbd.png)

`Wxh * xt` will be:

![state-e](https://user-images.githubusercontent.com/23405520/115376549-6a238a80-a1ec-11eb-9d15-3f3b6e34b595.png)

<b> Step 5: </b>

Now calculating `ht` for the letter `e`.

![htletter-e](https://user-images.githubusercontent.com/23405520/115376615-7a3b6a00-a1ec-11eb-85c9-4404d5bc0a46.png)

Now this would become `ht-1` for the next state and the recurrent neuron would use this along with the new character to predict the next one.

<b> Step 6: </b>

At each state, the recurrent neural network would produce the output as well. Let's calculate `yt` for the letter `e`.

![outeq (1)](https://user-images.githubusercontent.com/23405520/115376874-b4a50700-a1ec-11eb-8ffa-f96a8711bb14.png)
![ytfinal123](https://user-images.githubusercontent.com/23405520/115376894-b969bb00-a1ec-11eb-9aba-bd25e2eb61db.png)

<b> Step 7 : </b>

The probability for a particular letter from the vocabulary can be calculated by applying the softmax function, so we shall have `softmax(yt)`

![classwise-prob](https://user-images.githubusercontent.com/23405520/115377023-da321080-a1ec-11eb-9448-f0f507ffc189.png)

If we convert these probabilities to understand the prediction, we see that the model says that the letter after `e` should be `h` since the highest probability is for the letter `h`. Does this mean we have done something wrong? No, so here we have hardly trained the network. We have just show it two letters. So it pretty much hasn't learnt anything yet.

Now the next BIG question that faces us is how does Back propagation work in case of a Recurrent Neural Network. How are the weights updated while there is a feedback loop?

### Back propagation in a Recurrent Neural Network(BPTT)

To imagine how weights would be updated in case of a recurrent neural network, might be a bit of a challenge. So to understand and visualize the back propagation, let's unroll the network at all the time steps. In an RNN we may or may not have outputs at each time step.

In case of a forward propagation, the inputs enter and move forward at each time step. In case of a backward propagation in this case, we are figuratively going back in time to change the weights, hence we call it the Back propagation through time (BPTT).

![bptt](https://user-images.githubusercontent.com/23405520/115380838-9b05be80-a1f0-11eb-8130-48f2b0adacd6.png)

In case of an RNN, if `yt` is the predicted value `yt` is the actual value, the error is calculated as a cross entropy loss - 

`Et(ȳt,yt) = – ȳt log(yt)`
` E(ȳ,y) = – ∑ ȳt log(yt)`

We typically treat the full sequence (word) as one training example, so the total error is just the sum of the errors at each time step (character). The weights as we can see are the same at each time step. Let's summarize the steps for backpropagation

1. The cross entropy error is first computed using the current output and the actual output.
2. Remember that the network is unrolled for all the time steps.
3. For the unrolled network, the gradient is calculated for each time step with respect to the weigth parameter.
4. Now that the weight is the same for all the time steps the gradient can be combined together for all time steps.
5. The weights are then updated for both current neuron and the dense layers.


The unrolled network looks much like a regular neural network. And the back propagation algorithm is similar to a regular neural network, just that we combine the gradients of the error for all time steps. Now what do you think might happen, if there are 100s of time steps. This would basically take really long for the network to converge since after unrolling the network becomes really huge.

In case you do not wish to deep dive into the math of backpropagation, all you need to understand is that back propagation through time works similar as it does in a regular neural network once you unroll the recurrent neuron in your network. However, I shall be coming up with a detailed article on Recurrent Neural networks with scratch with would have the detailed mathematics of the backpropagation algorithm in a recurrent neural network.



### Training through RNN
1. A single time step of the input is provided to the network.
2. Then calculate its current state using set of current input and the previous state.
3. The current ht becomes ht-1 for the next time step.
4. One can go as many time steps according to the problem and join the information form all the previous states.
5. Once all the time steps are completed the final current state is used to calculate the output.
6. The output is then compared to the actual output i.e the target output and the error is generated.
7. The error is then back-propagated to the network to update the weights and hence the network (RNN) is trained.

### Advantages of Recurrent Neural Network
1. An RNN remembers each and every information through time. It is useful in time series prediction only because of the feature to remember previous inputs as well. This is called Long Short Term Memory.
2. Recurrent neural network are even used with convolutional layers to extend the effective pixel neighborhood.

### Disadvantages of Recurrent Neural Network
1. Gradient vanishing and exploding problems.
2. Training an RNN is a very difficult task.
3. It cannot process very long sequences if using tanh or relu as an activation function.
