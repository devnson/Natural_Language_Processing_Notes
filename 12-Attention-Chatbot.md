![image](https://user-images.githubusercontent.com/23405520/123618931-63d30f80-d828-11eb-9a9a-82b471fd270b.png)
![image](https://user-images.githubusercontent.com/23405520/123618985-73525880-d828-11eb-8daf-ea7bb63b513a.png)
![image](https://user-images.githubusercontent.com/23405520/123619045-7fd6b100-d828-11eb-8bf0-3fb166e5bf85.png)
![image](https://user-images.githubusercontent.com/23405520/123619095-8fee9080-d828-11eb-815d-a1dd093ebcec.png)
![image](https://user-images.githubusercontent.com/23405520/123619170-a563ba80-d828-11eb-9e30-9eab058510b4.png)
![image](https://user-images.githubusercontent.com/23405520/123619225-af85b900-d828-11eb-9d83-e30f72ae3112.png)

`Python 
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
        return output, hidden, attn_weights`
`

        
![image](https://user-images.githubusercontent.com/23405520/123619502-f5db1800-d828-11eb-8b0e-95eeb434087e.png)

`Python
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
`

![image](https://user-images.githubusercontent.com/23405520/123619674-215e0280-d829-11eb-96c7-d5b6343b53b7.png)

![image](https://user-images.githubusercontent.com/23405520/123619713-2ae76a80-d829-11eb-9bf5-f5262d148c3d.png)
