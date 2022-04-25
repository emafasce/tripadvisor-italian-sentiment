# Sentiment Analysis on Italian TripAdvisor reviews (Data Science Lab exam)

41076 italian reviews were scraped from Tripadvisor and labeled. The goal of this project is experimenting with data science techniques and building a performant classifier from scratch. The grade for this exam was assigned also accordingly to the position on the final leaderboard. 

Here I present two different approaches, one using a standard MLP and the second one using an ensemble model built from a LSTM (due to its temporal pattern recognition capabilities) and a Random Forest (due to its interpretability). 

## MAIN APPROACHES

When feeding the neural networks, all the stemmed words are used as input, while only the nouns and the adjectives are used as input of the random forest.

![alt text](https://github.com/emafasce/tripadvisor-italian-sentiment/blob/master/visualizations/approaches.png)

## RESULTS AND VISUALIZATIONS

The best performing approach is the Approach 1, which scores a F-Score of 0.927 on the test set, while the Approach 2 scores 0.943.

Here are the wordclouds on the reviews predicted as positive and negative by the model built with the Approach 1.

#### Wordcloud positive reviews
![alt text](https://github.com/emafasce/tripadvisor-italian-sentiment/blob/master/visualizations/cloudpositive.png)

#### Wordcloud negative reviews
![alt text](https://github.com/emafasce/tripadvisor-italian-sentiment/blob/master/visualizations/cloudnegative.png)

The Random Forest approach was selected due to its interpretability: indeed, we can select the most important positive or negative words.

![alt text](https://github.com/emafasce/tripadvisor-italian-sentiment/blob/master/visualizations/meaning.png)

If the reader is an italian speaker, he or she can recognize that most of these terms are indeed very expressive, and we can almost guess the writers' opinion from them. The fact that this result was obtained simply by the random forest is very fascinating to me.
