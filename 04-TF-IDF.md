# Term Frequency - Inverse Document Frequency (TF-IDF)

" Term frequency-inverse document frequency, is a numerial statistic that is intended to reflect how important a word is to a document in a collection or corpus"

- The **term frequency** of a word in a document. There are several ways of calculating this fequency, with the simplest being a raw count of instances a word appears in a document. Then, there are ways to adjust the frequency, by length of a document, or by the raw frequency of the most frequent word in a document.

- The **inverse document frequency** of the word across a set of documents. This means, how common or rare a word is in the entire document set. The close it is to 0, the more common a word is. This metric can be calculated by taking the total number of documents, dividing it by the number of documents that contain a word, and calculating the logarithm.


### Term Frequency (TF)

Let's first understand Term Frequent (TF). It is a measure of how frequently a tem, t, appears in a document, d:
 
![tf-300x41](https://user-images.githubusercontent.com/23405520/115829450-a5ac9780-a42c-11eb-8e81-e7f3bf062a07.jpg)

**Here, in the numerator, n is the number of times the term "t" appears in the document "d". Thus, each document and term would have its own TF value.**

We wll agian use the same vocabulary we had built in the Bag-of-Words model to show how to calculate the TF for 

- Review#1. **This movie is very scary and long**
- Review#2.  **This movie is not scary and is slow**
- Review#3. **This movie is spooky and good**


Here,

- Vocabulary: 'This', 'movie', 'is', 'very', 'scary', 'and', 'long','not','slow','spooky','good'
- Number of words in Review 2 = 8
- TF for the word 'this' = (number of times 'this' appears in review 2) / (number of terms in review 2) = 1/8

Similary,

- TF('movie') = 1/8
- TF('is') = 2/8 = 1/4
- TF('very') = 0/8 = 0
- TF('scary') = 1/8 
- TF('and') = 1/8
- TF('long') = 0/8 = 0
- TF('not') = 1/8
- TF('slow') = 1/8
- TF('spooky') = 0/8 = 0
- TF('good') = 0/8

We can calculate the term frequencie for all the terms and all the reviews in this manner:

![TF-matrix-1](https://user-images.githubusercontent.com/23405520/115830056-66327b00-a42d-11eb-9e24-68440f166517.png)


### Inverse Document Frequency (IDF)

IDF is a measure of how important a term is. We need the IDF value because computing just the TF alone is not sufficient to understand the importance of words:

![idf-300x44](https://user-images.githubusercontent.com/23405520/115830177-8c581b00-a42d-11eb-8abd-e8c6722f2c1c.jpg)

We can calculate the IDF values for the all the words in Review 2:

**IDF('this') = log(number of documents / number of documents containing the word 'this' )**

= log (3/3) = log(1) = 0

Similarly, 

- IDF('movie') = log(3/3) = 0
- IDF('is') = log(3/3) = 0
- IDF('not') = log(3/1) = log(3) = 0.48
- IDF('scary') = log(3/2) = 0.18
- IDF('and') = log(3/3) = 0
- IDF('slow') = log(3/1) = 0.48

We can calculate the IDF values for each word like this. Thus, the IDF values for the entire vocabulary would be:

![IDF-matrix](https://user-images.githubusercontent.com/23405520/115831680-8c591a80-a42f-11eb-8026-2e198f0a2238.png)

Hence, we see that words like "is", "this","and", etc. are reduced to 0 and have little importance; while words like "scary","long","good",etc. are words with more importance and thus have a higher value.

We can now compute the TF-IDF score for each word in the corpus. Words with a higher score are more important, and those with a lower score are less important.

![tf_idf](https://user-images.githubusercontent.com/23405520/115831902-d3dfa680-a42f-11eb-8d23-6858f278331f.jpg)

We can now calculate the TF-IDF score for every word in Review 2:

TF-IDF(‘this’, Review 2) = TF(‘this’, Review 2) * IDF(‘this’) = 1/8 * 0 = 0

Similarly,

- TF-IDF(‘movie’, Review 2) = 1/8 * 0 = 0
- TF-IDF(‘is’, Review 2) = 1/4 * 0 = 0
- TF-IDF(‘not’, Review 2) = 1/8 * 0.48 = 0.06
- TF-IDF(‘scary’, Review 2) = 1/8 * 0.18 = 0.023
- TF-IDF(‘and’, Review 2) = 1/8 * 0 = 0
- TF-IDF(‘slow’, Review 2) = 1/8 * 0.48 = 0.06

Similary, we can calculate the TF-IDF scores for all the words with respect to all the reviews:

![TF_IDF-matrix](https://user-images.githubusercontent.com/23405520/115831991-f671bf80-a42f-11eb-8cba-e2f5242f4ac2.png)

We have now obtained the TF-IDF scores for our vocabulary. TF-IDF also gives larger values for less frequent words and is high when both IDF and TF values are high i.e the word is rare in all the documents combined but frequent in a single document.


## Applications of TF-IDF

Determining how relevant a word is to a document, or TD-IDF, is useful in many ways, for example:

- **Information retrieval**
TF-IDF was invented for document search and can be used to deliver results that are most relevant to what you’re searching for. Imagine you have a search engine and somebody looks for LeBron. The results will be displayed in order of relevance. That’s to say the most relevant sports articles will be ranked higher because TF-IDF gives the word LeBron a higher score.

It’s likely that every search engine you have ever encountered uses TF-IDF scores in its algorithm.

**Keyword Extraction**
TF-IDF is also useful for extracting keywords from text. How? The highest scoring words of a document are the most relevant to that document, and therefore they can be considered keywords for that document. Pretty straightforward.

