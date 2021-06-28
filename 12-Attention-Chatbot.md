## Part I: Attention understanding

Just like in “Attention” meaning, in real life when we looking at a picture or hearing the song, we usally focus more on some parts and pay less attention in the rest. The Attention mechanism in Deep Learning is also the same flow, paying greater attention to certain parts when processing the data

Attention is one component of a network’s architecture.

Follow the specific tasks, the encoder & decoder will be different. In machine translation, the encoder often set to LSTM/GRU/Bi_RNN, in image captioning, the encoder often set to CNN.

Such as for the task: Translating the sentence: 'le chat est noir' to English sentence (the cat is black)

The input has 4 words, plus EOS token at the end (stop word) corresponding 5 time steps in translating to English. Each time step, Attention is applied by assigning weights to input words, the more important words, the bigger weights will be assigned (Done by backprob gradient process). So There are 5 differrent times weights assigned (coresponding to 5 time steps) The general architecture in seq2seq as follow:

image.png

Without attention, The input in decoder based on 2 component: the initial decoder input (often we set it to EOS token first (start word)) and the last hidden encoder. This way has the drawback in case some informations of very first encoder cell would be loss during the process. To handle this problem, the attention weight is added to all encoder outputs.

Capture.JPG
As we can see, through each decoder output word, the attention weights colors of encoder input is changed differently along itself importance

You may ask how can we appropriately set the weight to encoder outputs. The answer is: we just randomly set the weights, and the backpropagation gradient process will take care about it during the training. What we have to do is correctly build the forward computational graph.

Capture1.JPG

After attention weight was caculated, now we have three components: decoder input, decoder hidden, (attention weights * encoder outputs), we feed them to decoder to return decoder output

There are two primary types of attention: Bahdanau Attention vs Luong Attention. Luong attention is built on top of Bahdanau attention and have proved better scores in several tasks. This kernel is focus on Luong attention.

import torch
import torch.nn as nn
Computational graph for Luong attention
image.png

Step 1: Caculating encoder hidden state

class Encoder_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, drop_prob=0):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True)

    def forward(self, inputs, hidden):
        # Embed input words
        embedded = self.embedding(inputs)
        # Pass the embedded word vectors into LSTM and return all outputs
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden
Step 2-->6

class Luong_Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention, n_layers=1, drop_prob=0.1):
        super(LuongDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

    # The Attention layer is defined in a separate class
        self.attention = attention
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        # Embed input words
        embedded = self.embedding(inputs).view(1,1,-1)
        embedded = self.dropout(embedded)

    # STEP 2: GENERATE NEW HIDDEN STATE FOR DECODER
        lstm_out, hidden = self.lstm(embedded, hidden)

    # STEP 3: Calculating Alignment Scores 
        alignment_scores = self.attention(lstm_out,encoder_outputs)

    # STEP 4: Softmaxing alignment scores to obtain Attention weights
        attn_weights = F.softmax(alignment_scores.view(1,-1), dim=1)

    # STEP 5: CACULATING CONTEXT VECTOR by Multiplying Attention weights with encoder outputs
        context_vector = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs)

    # STEP 6: CACULATING THE FINAL DECODER OUTPUT by Concatenating output from LSTM with context vector
        output = torch.cat((lstm_out, context_vector),-1)
        # Pass concatenated vector through Linear layer acting as a Classifier
        output = F.log_softmax(self.classifier(output[0]), dim=1)
        return output, hidden, attn_weights
Exploring the attention class in STEP 3: Caculating alignment score

In Luong Attention, there are 3 different ways (dot, general, concat) to caculate the alignment score.

1. Dot function
  This is the simplest of the functions: alignment score calculated by multiplying the hidden encoder and the hidden decoder.
  SCORE = H(encoder) * H(decoder)
2. General function
  similar to the dot function, except that a weight matrix is added into the equation
  SCORE = W(H(encoder) * H(decoder))
3. Concat function
  Concating encoder and decoder first, the feed to nn.Linear and activation it, finally we add W2 to get final Score
  SCORE = W2 * tanh(W1(H(encoder) + H(decoder)))
Implementing attention class:

class Luong_attention_layer(nn.Module):
    def __init__(self, method, hidden_size):
        super(Luong_attention_layer, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, 'is not appropriate attention method')
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.weight = nn.Parameter(torch.FloatTensor(hidden_size))

    def get_dot_score(self, hidden, encoder_outputs):
        return torch.sum(hidden*encoder_outputs, dim=2)

    def get_general_score(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)
        return torch.sum(hidden * energy, dim=2)

    def get_concat_score(self, hidden, encoder_outputs):
        concat = torch.cat((hidden.expand(encoder_outputs.size(0),-1,-1), encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(concat))
        return torch.sum(self.weight * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'dot':
            attn_energy = self.get_dot_score(hidden, encoder_outputs)
        elif self.method == 'general':
            attn_energy = self.get_general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energy = self.get_concat_score(hidden, encoder_outputs)

        ## Transpose  attn_energy
        attn_energy = attn_energy.t()

        # Softmanx the attn_energy to return the weight corresponding to each encoder output
        return F.softmax(attn_energy, dim=1).unsqueeze(1)
Part II: Building chatbot seq2seq with Luong attention mechanism
The step by step for building chatbot with attention as follow:Capture%204.JPG

After running this kernel. you can play with chatbot and have some fun with him like this:)) :

Capture6.JPG

The code is based on : https://pytorch.org/tutorials/beginner/chatbot_tutorial.html. I have modified this toturial on something because the Author used some pytorch features that currently depressed. Through this kernel, I added explaination on my own understanding step by step so you might find it friendly to understand all the concepts.
