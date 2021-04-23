# Bag of Words (BoW) model

The Bag of Words (BoW) model is the simplest form of text representation in numbers. Like the term itself, we can represent a sentence as a bag of words vector (a string of numbers).

Let's we have three movie reviews:

- **Review 1**: This movie is very scary and long.
- **Review 2**: This movie is not scary and is slow.
- **Review 3**: This movie is spooky and good.


We will first build a vocabulary from all the unique words in the above three reviews. The vocabulary consists of these 11 words: 'This', 'movie', 'is','very','scary','and','long','not','slow','spooky','good'.

We can now take each of these words and mark their occurence in the three movie reviews above with 1s and 0s. This will give us 3 vectors for 3 reviews:

![BoWBag-of-Words-model-2](https://user-images.githubusercontent.com/23405520/115826095-038ab080-a428-11eb-9a25-a234fb787b2f.png)

- **Vector of Review 1**: [1 1 1 1 1 1 1 0 0 0 0]
- **Vector of Review 2**: [1 1 2 0 0 1 1 0 1 0 0]
- **Vector of Review 3**: [1 1 1 0 0 0 1 0 0 1 1]

And that's the core idea behind a Bag of Words (BoW) model.


### Drawbacks

- If the new sentence contain new words, then our vocabulary size would increase and thereby, the legnth of the vectors would increase too.
- Additionally, the vectors would also contain many 0s, thereby resulting in a sparse matrix (which is what we would like to avoid).
- We are rataining no information on the grammar of the sentences nor the ordering of the words in the text.
