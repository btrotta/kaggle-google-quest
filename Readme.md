# Google Quest Q&A labeling

This is the code for my top 6% solution to the Google Quest Q&A Labeling challenge on Kaggle. This NLP competition requires us to
predict the scores given by human raters to questions and answers on various Stack Exchange Q&A websites. The questions
and answers are scored on 30 dimensions, including whether they are useful, well-written, etc.

My code relies heavily on the following public notebook: https://www.kaggle.com/akensert/bert-base-tf2-0-now-huggingface-transformer
The BERT modelling code is almost identical to that kernel; the only change I made was to insert a special token
between the question title and body. (I'm not sure whether this had any real effect.)

To optimise the Spearman rho metric, I found it was helpful to limit the predicted scores to a smaller number of distinct
values. The way I did this was by post-processing using a LightGBM model with small leaf size (3) and few iterations (20).
The model just takes a single predictor (the output prediction of the BERT model) and optimises the cross-entropy loss. This
gives a boost of around 0.02 on the private leaderboard.
