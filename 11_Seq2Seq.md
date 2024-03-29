# A Simple Introduction to Sequence to Sequence Models


### Introduction

![Introduction-to-Sequence-to-Sequence-Models](https://user-images.githubusercontent.com/23405520/116772670-4a9f2400-aa6e-11eb-9afc-7dc33f71aa33.jpg)

Sequence to Sequence (often abbreviated to seq2seq) models is a special class of Recurrent Neural Network architecutres that we typically use (but not restricted) to solve complex Language problems like Machine Translation, Question Answering, creating Chatbots, Text Summarizationi, etc.

![0_iDgmgGnrzq65dPXy](https://user-images.githubusercontent.com/23405520/116772753-96ea6400-aa6e-11eb-8c88-234976435d3c.jpg)


### Use Cases of the Sequence to Sequence Models

Sequence to sequence models lies behind numerous systems that you face on a daily basis. For instance, seq2seq model powers applications like Google Translate, voice-enabled devices and online chatbots. The following are some of the applications:

- Machine translation - a 2016 paper form Google shows how the seq2seq model's translation quality "approaches or surpasses all currently published results"

https://arxiv.org/pdf/1409.3215.pdf

![1_s2qQ9RM27O4sa2gC1dJ0fg](https://user-images.githubusercontent.com/23405520/116772796-e92b8500-aa6e-11eb-9f9c-d31f0ce28ddb.png)

- Speech recognition - another Google paper that compares the existing seq2seq models on the speech recognition task.

https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0233.PDF

![0_QsbCJ3lGfcaBlCJf](https://user-images.githubusercontent.com/23405520/116772811-03fdf980-aa6f-11eb-88c4-a37bef7fdddb.jpg)

There are only some applications where seq2seq is seen as the best solution. This model can be used as a solution to any sequences-based problem, especially ones where the inputs and outputs have different sizes and categories.

### Encoder-Decoder Architecture

The most common architecture used to build Seq2Seq models is Encoder-Decoder architecture.

![Encoder-Decoder-Architecture-for-Neural-Machine-Translation](https://user-images.githubusercontent.com/23405520/116772848-4f180c80-aa6f-11eb-99d8-a5565cc10b1c.png)



### Encoder:

- Both encoder and the decoder are LSTM models (or sometimes GRU models)
- Encoder reads the input sequence and summarizes the information in something called the internal state vecotrs and context vector (in case of LSTM these are called the hidden state and cell state vectors). We discard the outputs of the encoder and only preserve the internal states. This context vector aims to encapsulate the information for all input elements in order to help the decoder make accurate predictions.
- The hidden states hi are computed using the formula:

![1_4mHGvQAV6UN_t_PbZuXwhQ](https://user-images.githubusercontent.com/23405520/116772907-aa49ff00-aa6f-11eb-93fc-143e6a4cfc30.png)

![1_aBv8MhvfseL_pTFBqiIrog](https://user-images.githubusercontent.com/23405520/116772909-af0eb300-aa6f-11eb-8480-18f84913e6ba.png)

The LSTM reads the data, one sequence after the other. Thus if the input is a sequence of length 't', we say that LSTM reads it in 't' time steps.

1. Xi  = input sequence at time step i
2. hi and ci = LSTM maintains two states ('h' for hidden state and 'c' for cell state) at each time step.
3. Yi = Output sequence at time step i. Yi is actually a probability distribution over the entire vocabulary which is generated by using a softmax activation. Thus each Yi is a vector of size "vocab_size" representing a probability distribution.

### Decoder

- The decoder is an LSTM whose initial states are initialized to the final states of the Encoder LSTM, i.e the context vector of the encoder's final cell is input to the first cell of the decoder network. Using these initial states, the decoder stats generating the output sequences, and these outputs are also taken into consideration for future outputs.
- A stack of several LSTM units where each predicts an output `yt` at a time step `t`.
- Each recurrent unit accepts a hidden state from the previous unit and produces and output as well as its own hidden state.
- Any hidden state `hi` is computed using the formula

![1_H6p5X-TOLxLfFrwyPTpNwA](https://user-images.githubusercontent.com/23405520/116773057-bc786d00-aa70-11eb-9199-19a39add44b5.png)
![1_p0pQOvo2rVCr_KmBMn6rtw](https://user-images.githubusercontent.com/23405520/116773065-c39f7b00-aa70-11eb-9bd8-1a5550eb6b00.png)


- The output `yt` at time step `t` is computed using the formula:


![Uploading 1_p0pQOvo2rVCr_KmBMn6rtw.png…]()


- We calculate the outputs using the hidden state at the current time step together with the respective weight `W(s)`. Softmax is used to create a probability vector which will help us determine the final output (e.g word in the question-answering problem).

![1_y4D1XNJQmx-Gii1oHeHy_A](https://user-images.githubusercontent.com/23405520/116773069-ca2df280-aa70-11eb-8d70-f7ec82671ee3.png)

We will add two tokens  in the output sequence as follows:

### Example

"**START_** John is hard working **_END**

The most important point is that the initial states (h0, c0) of the decoder are set to the final states of the encoder. The intuitively means that the decoder is trained to start generating the output sequence depending on the information encoded by the encoder.

Finally, the loss is calculated on the predicted outputs from each time step and the errors are backpropagated through time in oder to update the parameters of the network. Training the network over a longer period with a sufficiently large amount of data results in pretty good predictions.

![1_KtWwvLK-jpGPSnj3tStg-Q](https://user-images.githubusercontent.com/23405520/116773437-06625280-aa73-11eb-9744-a6697179de60.png)

- During inference, we generate one word at a time.
- The initial states of the decoder are set to the final states of the encoder.
- The inital input to the decoder is always the START token.
- At each time step, we preserve the states of the decoder and set them as inital states for the next time step.
- At each time step, the predicted output is fed as input in the next time step.
- We break the loop when the decoder predictes the END token.

### Drawbacks of Encoder-Decoder Models :

There are two primary drawbacks to this architecture, both related to length.

1. Firstly, as with humans, this architecture has very limited memory. That final hidden state of the LSTM, which we call S or W, is where you’re trying to cram the entirety of the sentence you have to translate.S or W is usually only a few hundred units (read: floating-point numbers) long — the more you try to force into this fixed dimensionality vector, the lossier the neural network is forced to be. Thinking of neural networks in terms of the “lossy compression” they’re required to perform is sometimes quite useful.


2. Second, as a general rule of thumb, the deeper a neural network is, the harder it is to train. For recurrent neural networks, the longer the sequence is, the deeper the neural network is along the time dimension.This results in vanishing gradients, where the gradient signal from the objective that the recurrent neural network learns from disappears as it travels backward. Even with RNNs specifically made to help prevent vanishing gradients, such as the LSTM, this is still a fundamental problem.

![0_PaGt4fcpHGUUM-NA](https://user-images.githubusercontent.com/23405520/116773564-8ee0f300-aa73-11eb-8580-f160447a9a9b.png)

Furthermore, for more robust and lengthy sentences we have models like **Attention Models** and **Transformers**.



Original Post:
https://www.analyticsvidhya.com/blog/2020/08/a-simple-introduction-to-sequence-to-sequence-models/
