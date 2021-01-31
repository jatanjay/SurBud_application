## Jatan Pandya 

[email](mailto:jatanjay212@gmail.com) | [web](https://jatanjay.github.io/) | [resume](https://drive.google.com/file/d/1b1tp17xBbkccwq57FWOwLbEbKKXXEAs2/view?usp=sharing) | [github](https://github.com/jatanjay)

____

2. I’ve worked with NLP libraries such as nltk, spaCy, Gensim, TextBlob etc. for a variety of applications. From pre-processing/text cleaning/data wrangling etc.to projects such as [Song and dance man](https://jatanjay.github.io/SongandDanceMan/). Though I am relatively new to Snippext, given research papers on both Snippext and OpinionDigest piqued my interest. While further researching about the same, I followed their GitHub documentation to understand more about the same.
   Further, I’ve been closely following a few books such as [this](https://web.stanford.edu/~jurafsky/slp3/), [this](https://www.nltk.org/book/) and [this](https://www.oreilly.com/library/view/natural-language-processing/9781491978221/) and their exercises for some time now. They have not only helped me understand the working of various libraries but also concepts in NLP, ML, and linguistics.

   ______

   

3. I have experience in cloud deployment. I’ve deployed one of the earlier projects, [CorpusGenius](https://jatanjay.github.io/CorpusGenius/) on Heroku. (Frontend is still in progress) As far as this internship is concerned, I recognize the requirements for deploying pytorch ML models on AWS. For the same, I’ve already started by learning through online tutorials and AWS SageMaker documentation. I believe given opportunity and under right mentorship, I’ll be able to carry out tasks in no time.

   ______

   

**<u>A closer look at OpinionDigest</u>**

<u>Key ideas:</u> 

**Abstractive Text Summarization** : Task of generating a concise summary that captures salient and subtle ideas of the source text. Aims to imitate how we humans summarize using deep learning. Summaries are generated from scratch, hence will contain unseen phrases, words etc.

**Aspect-based Sentiment Analysis (ABSA)** : Example, for a hotel if a review says “Food was good but the staff was bad”, a traditional sentiment analysis doesn’t tell you anything, since it will only give a number that will be positive/negative/neutral. This means, even though the review is actually a negative one, (Staff services is not good)  it’s possible the review may slip through as a “positive” or “neutral” review! 
In comes ABSA, simply put, this model will perform sentiment analysis on each aspect. (here food v/s service)

<u>Motivation :</u> 

Applying ABSA on open-ended types of surveys is exciting as it can help gather huge useful insights on a company's performance, user sentiments and relationship etc. Which I believe from where I see is what Survey Buddy specializes in.

It’s a common fact that sometimes reviews online are usually crude, caustic, slang containing. These factors can skew the traditional sentiment analysis models and thus provide little or less insights. A need of a clean, concise, human readable opinion based summary is of utmost important for further insights and analysis. OpinionDigest aims at doing the same.

<u>What is OpinionDigest ?</u> 

OpinionDigest is an abstractive opinion summarization framework that uses existing Aspect Based Sentiment Analysis (ABSA) and seq2seq models to extract opinions/phrases from reviews.

<u>What’s so good about OpinionDigest?</u>

- Although not a completely unsupervised framework, OpinionDigest manages to summarize and extract opinions without the need of a gold-standard dataset. This is a plus point since gold-standard datasets are not always available for all domains hence making it difficult to train the models.

- Also, the framework claims to be indifferent to the number of inputs. In the sense, quality of final summary is unaffected by the number of total input words.

- It also offers a customized summaries based on the user's specific need with respect to sentiment polarity /aspect category. 
- Evaluations on YELP datasets showed the framework to produce promising results than other counterparts. Further, human studies have also verified the same.

<u>Framework :</u> 

 The algorithm starts by defining a dataset that consists of all customer reviews (by various entities (say Hotel, Eatery etc.)). Next, for every element (entity) in that dataset,  a new set is created within that contains all reviews which in turn contains tokenized words.

Further, within the same set (that is set that contains reviews) a new set is defined. This new set will contain :

1. A phrase that will be an "opinion" that contains tokens concatenated (joined together) with a 'special' token called `SEP`. (which will be useful later during summarization) - (a)
2. The sentiment polarity :  I.E out of (positive - neutral - negative) for that phrase.
3. Aspect category: This aspect category can be thought of the 'characteristic/feature' of the entity in question (say cleanliness, service staff, food etc.)

Now, simply put, all OpinionDigest does is, it employs various methods (which we'll be seeing in just a few moments) that finally allows it to generate a summary out of all the "best" features (nothing but appropriate tokens) that it thinks to be good.


Thus, Since:

1. OpinionDigest generates summary by taking an abstractive approach, 
2. OpinionDigest explicitly deals with opinion sets of reviews and makes it the star of their framework,
3. OpinionDigest takes the aspect category into consideration, 

OpinionDigest is an abstract-opinion-summarization-framework that uses Aspect Based Sentiment Analysis (ABSA) to generate reviews.

Further, OpinionDigest’s framework can be broadly classified into three acts:

<u>Act one: Opinion Extraction :</u> 

The opinion extraction is carried out by using and earlier and yet another framework by the same company. The framework is called Snippext. Snippext is also an opinion mining system, but instead is fine-tuned through a semi-supervised learning model on augmented data. Snippext aims to mine opinions specifically where a small amount of label "gold-standard" training data is available. 

Thus, in a way OpinionDigest picks up where Snippext left off. In this sense, OpinionDigest takes those opinions and summarizes in a more human friendly, readable form. Think of Snippext as one of the prequels for OpinionDigest.

**How can we make it better :** Looking at the original opinion, we notice that “... and many other attractions” is extracted as “close to attraction”. I think the word “other” is crucial here since : 
Just saying ‘close to attraction’ creates a little self-referentially (since the wharf, aquatic park are attractions too!) 
“Other attractions” means there are multiple attractions and not only the two mentioned. This is too valuable to be left out since this means the hotel is in a much more prime location.
I propose to include certain adjectives like one above in the extraction process, that can sometimes drastically change the final summary generation outcome.

<u>Act two: Opinion Selection :</u> 

Since the central aim of OpinionDigest is selecting "opinions", following techniques are used for the same:

1. A greedy algorithm, specifically beam search (a heuristic search algorithm that retains only a predetermined number of 'best'  solutions) Thus solving the problem of redundancy while clustering opinions. Since the opinions are "merged", this step is called *Opinion Merging* 
2. Ranking: The ranking process is built upon an assumption that "larger the cluster, more important opinions it contains. If this condition is satisfied, the cluster in question is given an appropriate, high rank. The model is built around a central idea of ‘opinion popularity.’ Since the opinions are "ranked", this step is called *Opinion Ranking*
3. Filtering : This is a pure user experience feature, offering users a tailor made option where the selection can be filtered based on either aspect category or sentiment polarity.

**How can we make it better :** Currently, during opinion merging, top k=15 most popular opinion clusters are selected. I propose to by trial and error reduce or increase the value of k and note its effects on the final summary, since it seems using more opinions in clusters is giving rise to the so-called redundancy (see 3.4 - Human Evaluation) while the opinions are ranked in the next step. I think this is because the ranking system is built upon the assumption that larger clusters contain opinions that are popular.
Additionally, I came across this [paper](https://www.aclweb.org/anthology/E17-2047.pdf) that aims at the reduction of redundant repeating generations. 

<u>Act three: Summary Generation :</u> 

Before we start with summary generation, let's first understand the transformer model, the brain of the project. 
The transformer model is a new model that was introduced in the 2017 Google paper (Attention is all you need - Vaswani et al., 2017) which proposes a new method called “self-attention mechanism. The Transformer uses the representation of all words in context without having to compress all the information into a single fixed-length representation. This is crucial for OpinionDigest for two reasons. First, since context is important when considering opinions and second, the whole model aims at extracting opinions. In essence, the model learns to predict the next word in a sentence by focusing on words that were previously seen in the model and related to predicting the next word.

Training : Next, OpinionDigest uses the earlier generated Opinion sets for each and every review in the corpus (by entity) along with a textual form of the review (see - (a) to train the aforementioned transformer. At the end of this step, a review text is generated. Of course, Being an abstractive model, the resultant text contains new phrases, words etc. 

Summary Generation : Now, remember the set of generated opinions from Act II? They are now again fed into the aforementioned transformer model that again generates a textual summary using those opinions.

**How can we make it better :** 
During the summarization process, the opinions are selected by their frequency. This means opinions that dominate a particular sentiment will appear first. This is all good and dandy if the majority of opinions have negative sentiment. But what If majority opinions are positive with just one or two negative opinions? This will generate an anti-climatic summary! (here for us, negative reviews are of utmost importance, I give out reasons why in a minute)

<u>This can be further exemplified as, say a review says something like</u> 
“Good location, working appliances, good staff, comfy bed, found a fly in my soup”

Now, continuing on what I mean, the review actually tells us about a bigger problem at hand than just comfy beds. It is more or less masquerading as a positive review. As a hotel owner, I’d be more concerned and appreciate a summary that tells me what’s going wrong than letting good reviews take the majority space, that can sometimes push the actually important negative opinion aside.
Thus, I propose to by default generate a summary that first gives out the negative opinion and then the positive opinions. That is, a text summary that displays the most important sentences first. 
(OpionDigest, does claim that any desired ordering can be used, I further propose to by default, sort in a climatic fashion)

Thus in a nutshell,
The framework based on an ABSA model (Snippext) extracts opinion, feeds to the transformer model we just discussed that generates multiple reviews abstractedly (that is from ground up, will contain new phrases/words) Next, while summarizing, new most popular summaries are the generated by extracting from multiple reviews and they are in turn feed again to the transformer. Lastly, the output opinions are then ‘verbalized’ into a more human readable, perceivable summary.

_______

<u>**A closer look at PEGASUS :**</u>

While researching more about opinion mining, I came across Google’s [PEGASUS ](https://arxiv.org/pdf/1912.08777.pdf)Model, based on the earlier paper we discussed, “Attention is all you need.”

Core Model : 

PEGASUS stands for Pre-training with Extracted Gap-Sentences for Abstractive Summarization that implements a seq2seq architecture. Further, the model itself is a self-supervised learning model. Other models with self-supervised pre training are BERT, GPT-2, XLNet, T5 etc. that again use the aforementioned transformer model.

1. <u>Gap Sentence Generation (GSG)</u> : PEGASUS when applied on a text, the complete sentences are ‘removed’ and the model is pre-trained on a large corpora to predict thus removed sentences. 

2. <u>Masked Language Model (MLM)</u> : PEGASUS’s base architecture consists of an encoder and a decoder. Continuing on what we discussed for GSG, words are randomly masked from the sentence and the other remaining words are used from the sentence to predict those masked words. 

**What I liked about it** : At Least for me, the approach PEGASUS takes was unintuitive, that’s what made it exciting. Next, on fine tuning the model on 12 different datasets, PEGASUS managed to surpass the previous benchmarks for 6 datasets (and this too by training on less number of samples!)Further comparing the summaries along with the gold (human generated summary) it was quickly apparent that PEGASUS has beautifully achieved a human-like summary (for 3 datasets). Perhaps the most interesting result was the test on a news article. PEGASUS correctly abstracts the total number of ships in the news article - four, even though there are no traces of the total number anywhere in the article! 

**What can be improved?** : On continuing with the ship example, the researches on testing again (to find if it was fluke or not) found out that the model correctly abstracts numbers only if they lie in the range of 2-5. However, on adding a sixth ship, the “HMS Alphabet”, it miscounts it as “seven”. I think in-order to actually make counting another feature of PEGASUS, perhaps it can in future versions included. This feature will see wide applications in legal document summarization, news article summarizations or say scientific article summarization, i.e. areas that usually deal with multiple entities. 

But what is still worth noting is that the model is able to count even though it wasn’t explicitly programmed, so perhaps future advancement might not even need an explicit feature but will be achieved anyway (like how it ‘happened’ here.)
