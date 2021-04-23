# Stemming

Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words known as lemma. Stemming is important in Natural Language Understanding (NLU) and Natural Language Processing (NLP).

Stemming is a part of linguistic studies in morphology and artificial intelligence (AI) information retrieval and extraction. Stemming and AI knowledge extract meaningful information from vast sources like big data or the Internet since additional forms of a word related to a subject may need to be searched to get the best results. Stemming is also part of queries and Internet search engines.

Stemming is the process of producing morphological variants of a root/base word. Stemming programs are commonly referred to as stemming algorithms or stemmers.

Usually, a word has multple meanings based on its usuage in text, similarly, different forms of words convey related meaning, like "toy" and "toys", indicate indentical meaning.

You would probably find no different objective between a search for "toy" and a search for "toys". This kind of contrast between various forms of words termed as an "infection", however this makes varioius problems in understanding queries.

Suppose another word "came" and "camel", their search intent gives a different meaning, instead of having the same root-word. Similarly, if you search for the word "Love" in the google search option, it sows results in stems of words like "Loves","Loved", and "Loving".

We already know that a word has one root-base form but having different variations, for example "play" is a root-base word and playing, played, plays are the different forms of a single word. So, these words get stripped out, they might get the incorrect meanings or some other sort of errors.

The process of reducing inflection towards their root forms are called Stemming, this occurs in such a way that depicting a group of relatable words under the same stem, even if the root has no appropriate meaning.

**Moreover**;
- Stemming is a rule-based appraoch because it slices the inflected words from prefix or suffix as per the need using a set of commonly underused prefix and suffix, like "-ing", "-ed", "-pre", etc. It results in a word that is actually not a word.
- There are mainly two errors that occur while performing Stemming, Over-Stemming and Under-stemming. Over-stemming occurs when two words are stemmed from the same root of different stems. Under-stemming occurs when two words are stemmed from the same root of not a different stems. Two types of stemmers are:

- **Defining Porter Stemmer**
Porter Stemmer uses suffix striping to produce stems. It does not follow the linguistic set of rules to produce stem for phases in different cases, due to this reason porter stemmer does not generate stems i.e actual English words.

It applies algorithms and rules for producing stems. It also considers the rules to decide whether it is wise to strip the suffix or not. A computer program or subroutine that stems word may be called a stemming program, stemming algorithm or stemmer.

**Defining Snowball Stemmer**
It's an advanced version of Porter Stemmer, also called named as Porter2 Stemmer.
For example, if you print the word **"badly"** with the help of **Snowball** in English and Porter, we get different results. 

Here, the word "badly" is stripped from the English language using Snowball Stemmer and get an ouput as "bad". Now, snowball Stemmer is used for stripping the same word from the Porter language, we get the output as **"badli**.

## Lemmatization

In simpler forms, a method that swtiches any kind of a word to its base root mode is called Lemmatization.

In other words, Lemmatization is a method responsible for grouping different inflected forms of words into the root form, having the same meaning. It is similar to stemming, in turn, it gives the stripped word that has some dictionary meaning. The Morphological analysis would require the extraction of the correct lemma of each word.

For example, Lemmatization clearly identifies the base form of **"trouble"** to **"trouble"** denoting some meaning whereas, Stemming will cut out **"ed"** part and convert it into **"troubl"** which has the wrong meaning and spelling errors.

**troubled** ---> Lemmatization ---> **'trouble'**, and error
**troubled** ---> Stemming ----> **troubl**

## Difference between Stemming and Lematization


![image](https://user-images.githubusercontent.com/23405520/115822278-e226c600-a421-11eb-93d6-3bf070b2b588.png)
![image](https://user-images.githubusercontent.com/23405520/115822295-ebb02e00-a421-11eb-987a-4bc7a2f206e9.png)
