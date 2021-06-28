# Understanding LSTM Networks

org: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

### Recurrent Neural Networks

Humans don't start their thinking from scratch every second. As you read this essay, you understand each word based on your understanding of previous words. You don't throw everything away and start thinking from stratch again. Your thought have persistence.

Traditional neural networks can't do this, and it seems like a major shortcoming. For example, you imagine you want to classify what kind of event is happening at every point in a movie. It's unclear how a traditional neural network could use its reasoning about previous events in the film to inform later ones.

Recurrent neural networks address this issue. They are networks with loops in them, allowing information to persist.

![RNN-rolled](https://user-images.githubusercontent.com/23405520/115515684-01e3b000-a2a3-11eb-9413-2e68dd4c718e.png)

      Recurrent Neural Networks have loops
      
In the above diagram, a chunk of neural network, **A**, looks at some input `Xt` and outputs a value `ht`. A loop allows information to be passed from one step of the network to the next.

These loops make recurrent neural networks seem kind of mysterious. However, if you think a bit more, it turns out that they aren't all that different than a normal neural network. A recurrent neural networks can be thought of as multiple copies of the same network, each passing a message to a sucessor. Consider what happens if we unroll the loop:

![RNN-unrolled](https://user-images.githubusercontent.com/23405520/115516150-7880ad80-a2a3-11eb-83b8-db47a89525dc.png)


        An unrolled recurrent neural network
        
This chain-like nature reveals that recurrent neural networks are intimately related to sequences and lists. They're the natural architecture of neural network to use for such data.

And they certainly are used! In the last few years, there ahve been incredible success applying RNNs to a variety of problems: speech recognition, langauage modeling, translation, image captioning.. The list goes on. 

Essential to these successes is the use of “LSTMs,” a very special kind of recurrent neural network which works, for many tasks, much much better than the standard version. Almost all exciting results based on recurrent neural networks are achieved with them. It’s these LSTMs that this essay will explore.

## The Problem of Long-Term Dependencies

One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task, such as using previous video frames might inform the understanding of the present frame. If RNNs could do this, they’d be extremely useful. But can they? It depends.

Sometimes, we only need to look at recent information to perform the present task. For example, consider a language model trying to predict the next word based on the previous ones. If we are trying to predict the last word in **“the clouds are in the ”** we don’t need any further context – it’s pretty obvious the next word is going to be **sky**. In such cases, where the gap between the relevant information and the place that it’s needed is small, RNNs can learn to use the past information.

![RNN-shorttermdepdencies](https://user-images.githubusercontent.com/23405520/115516766-12485a80-a2a4-11eb-92ee-34c168b8846b.png)

But there are also cases where we need more context. Consider trying to predict the last word in the text “I grew up in France… I speak fluent French.” Recent information suggests that the next word is probably the name of a language, but if we want to narrow down which language, we need the context of France, from further back. It’s entirely possible for the gap between the relevant information and the point where it is needed to become very large.

Unfortunately, as that gap grows, RNNs become unable to learn to connect the information.


![RNN-longtermdependencies](https://user-images.githubusercontent.com/23405520/115516854-29874800-a2a4-11eb-9d6c-5f518b4d8485.png)

In theory, RNNs are absolutely capable of handling such “long-term dependencies.” A human could carefully pick parameters for them to solve toy problems of this form. Sadly, in practice, RNNs don’t seem to be able to learn them. The problem was explored in depth by Hochreiter (1991) [German] and Bengio, et al. (1994), who found some pretty fundamental reasons why it might be difficult.

Thankfully, LSTMs don’t have this problem!

## LSTM Networks

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work.1 They work tremendously well on a large variety of problems, and are now widely used.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

![LSTM3-SimpleRNN](https://user-images.githubusercontent.com/23405520/115517037-5dfb0400-a2a4-11eb-882a-8f4818c66d31.png)

        The repeating module in a standard RNN contains a single layer.
        
LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

![LSTM3-chain](https://user-images.githubusercontent.com/23405520/115517125-710dd400-a2a4-11eb-83d5-480c6c8a0f9b.png)

        The repeating module in an LSTM contains four interacting layers.
        

Don't worry about the details of what's going on. We'll walk through the LSTM diagram step by step later. For now, let's just try to get comfortable with the notation we'll be using.

![LSTM2-notation](https://user-images.githubusercontent.com/23405520/115517335-a9adad80-a2a4-11eb-9d74-68a294a85d93.png)

In the above diagram, each line carries an entire vector, from the output of one node to the input of others. The pink circles represent pointwise operations, like vector addition, while the yellow boxes are learned neural network layers. Lines merging denote concatenation, while a line forking denote its content being copied and the copies going to different locations.

## The Core Idea Behind LSTMs

The key to LSTMs is the cell state, the horizontal line running through the top of the diagram.

The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.

![LSTM3-C-line](https://user-images.githubusercontent.com/23405520/115517665-014c1900-a2a5-11eb-9ffc-15848820758c.png)

The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.

Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

![LSTM3-gate](https://user-images.githubusercontent.com/23405520/115517704-090bbd80-a2a5-11eb-8e96-9ed96733d849.png)

The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”

An LSTM has three of these gates, to protect and control the cell state.


## Step-by-Step LSTM Walk through

The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.” It looks at `ht−1` and `xt`, and outputs a number between `0` and `1` for each number in the cell state `Ct−1`. A 1 represents “completely keep this” while a 0 represents “completely get rid of this.”

Let’s go back to our example of a language model trying to predict the next word based on all the previous ones. In such a problem, the cell state might include the gender of the present subject, so that the correct pronouns can be used. When we see a new subject, we want to forget the gender of the old subject.

![LSTM3-focus-f](https://user-images.githubusercontent.com/23405520/115517857-3b1d1f80-a2a5-11eb-96ea-450b296bdc77.png)

The next step is to decide what new information we’re going to store in the cell state. This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values,![image](https://user-images.githubusercontent.com/23405520/115517942-50924980-a2a5-11eb-9da1-e4a989dd71f6.png)
 , that could be added to the state. In the next step, we’ll combine these two to create an update to the state.
 
 In the example of our language model, we’d want to add the gender of the new subject to the cell state, to replace the old one we’re forgetting.
 
 ![LSTM3-focus-i](https://user-images.githubusercontent.com/23405520/115518011-5ee06580-a2a5-11eb-9ce1-f48512b1abf2.png)

It’s now time to update the old cell state, ![image](https://user-images.githubusercontent.com/23405520/115518118-77e91680-a2a5-11eb-94f1-d7baaa6f57ec.png)
, into the new cell state ![image](https://user-images.githubusercontent.com/23405520/115518176-85060580-a2a5-11eb-9ffd-05d2f4c3c033.png). The previous steps already decided what to do, we just need to actually do it.

We multiply the old state by ![image](https://user-images.githubusercontent.com/23405520/115518247-92bb8b00-a2a5-11eb-906c-d88129f20b43.png) forgetting the things we decided to forget earlier. Then we add ![image](https://user-images.githubusercontent.com/23405520/115518296-a0711080-a2a5-11eb-9fc1-62a018cbc3e4.png) This is the new candidate values, scaled by how much we decided to update each state value.

In the case of the language model, this is where we’d actually drop the information about the old subject’s gender and add the new information, as we decided in the previous steps.

![LSTM3-focus-C](https://user-images.githubusercontent.com/23405520/115518336-a961e200-a2a5-11eb-8fb5-15383ab6964b.png)

Finally, we need to decide what we’re going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through `tanh `(to push the values to be between −1 and 1) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.

For the language model example, since it just saw a subject, it might want to output information relevant to a verb, in case that’s what is coming next. For example, it might output whether the subject is singular or plural, so that we know what form a verb should be conjugated into if that’s what follows next.

![LSTM3-focus-o](https://user-images.githubusercontent.com/23405520/115518403-bd0d4880-a2a5-11eb-886a-69e0612e7ce2.png)

## Variants on Long Short Term Memory

What I’ve described so far is a pretty normal LSTM. But not all LSTMs are the same as the above. In fact, it seems like almost every paper involving LSTMs uses a slightly different version. The differences are minor, but it’s worth mentioning some of them.

One popular LSTM variant, introduced by Gers & Schmidhuber (2000), is adding “peephole connections.” This means that we let the gate layers look at the cell state.

![LSTM3-var-peepholes](https://user-images.githubusercontent.com/23405520/115518448-ca2a3780-a2a5-11eb-9c08-04f6a61bd5f0.png)

The above diagram adds peepholes to all the gates, but many papers will give some peepholes and not others.

Another variation is to use coupled forget and input gates. Instead of separately deciding what to forget and what we should add new information to, we make those decisions together. We only forget when we’re going to input something in its place. We only input new values to the state when we forget something older.

![LSTM3-var-tied](https://user-images.githubusercontent.com/23405520/115518473-d2827280-a2a5-11eb-92c8-f2a5d880f585.png)

A slightly more dramatic variation on the LSTM is the Gated Recurrent Unit, or GRU, introduced by Cho, et al. (2014). It combines the forget and input gates into a single “update gate.” It also merges the cell state and hidden state, and makes some other changes. The resulting model is simpler than standard LSTM models, and has been growing increasingly popular.

![LSTM3-var-GRU](https://user-images.githubusercontent.com/23405520/115518506-dc0bda80-a2a5-11eb-9a66-7993f8918552.png)

These are only a few of the most notable LSTM variants. There are lots of others, like Depth Gated RNNs by Yao, et al. (2015). There’s also some completely different approach to tackling long-term dependencies, like Clockwork RNNs by Koutnik, et al. (2014).

Which of these variants is best? Do the differences matter? Greff, et al. (2015) do a nice comparison of popular variants, finding that they’re all about the same. Jozefowicz, et al. (2015) tested more than ten thousand RNN architectures, finding some that worked better than LSTMs on certain tasks.

