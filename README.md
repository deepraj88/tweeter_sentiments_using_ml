# Twitter Sentiment Prediction using Machine Learning

The idea is to make an ML classifier which predicts what kind of reaction a News Tweet will generate and classify them in one of the 9 possible classes ( [low, medium, high]volume * [negative, neutral, positive]sentiment ). 

The input to this model will be a tweet, and the output will be one of the 9 classes. The input tweet gets preprocessed via standard text preprocessing like tokenization, stopword removal, lemmatization etc. to use a bag of word approach containing only useful information words.

To train this model, we also need an output matrix. And output will be generated from popular news channel's tweet related data i.e. tweets text, comments, like count, retweet count etc. We will preprocess this data to assign one of the 9 class to each tweet. Weighted combination of like count and retweet count will be used to decide volume of each tweet's reaction. Whereas, weighted combination of comment's sentiments (analyzed via **NLTK Vader Sentiment Analyzer**) will be used to decide sentiment of each tweet's reaction.

The actual machine learning classifier will use bag of word approach using **Multinomial Naive Bayes classifier with TF-IDF optimization** for better accuracy. 


_**UPDATE**: Since Twitter has put a lot of restriction on it's API regarding the amount of data one can extract, we aren't getting enough data to make a workable machine learning model. More so, associating comments with a particular tweet is challenging and not directly accessible. Hence, **we are stuck at this point.**_
