# LDA_AdML

In this project we re-implemented the Latent Dirichlet Allocation (LDA) as described in the [2003 Blei et. al. paper](https://github.com/Javigsv/LDA_AdML/blob/main/LDA%20paper.pdf). The model was originally created for topic modelling in large corpora in order to find low-dimensional representations of documents while retaining statistical relationships useful for tasks relating to information retrieval.

The model is trained with an Variational Inference - Expectation Maximization (VI-EM) alogirthm. Read more in the [final report](https://github.com/Javigsv/LDA_AdML/blob/main/Report%20-%20Large%20VI%20-%20DD2434.pdf).

This was the final project of the course _DD2434 Machine Learning, Advanced Course_ during the fall of 2020.

## Results

Here are a small sample of the results are presented.

### Topics over time

Using the model parameters one can approximate the "popularity" of topics over time. This can be seen for three topics in the graph below:

![Topics over time](/Images/topicovertime.png)

Is this graph accurate?

_Regarding climate, the peak is subjected around January 2017, which was a period when the Trump administration actively worked to erase ex-president Barack Obama’s climate initiatives. The graph of the Russian Election Allegations which gained large media traction after members of the United States Congress publicly disclosed a potential russian interference in the 2016 election. The media attention continued to increase as more US intelligence agencies confirmed the suspicions and the Office of the Director of National Intelligence published its detailed report in January 2017. Notably, the last peak from 2017-05 to 2017-07 coincides with the news surrounding Trump’s dismissal of James Comey, the FBI director, in may 9th, 2017._

### An example article

Looking at topics with a high number of expected words belonging to a certain article and also making sure that the words have a high probability of belonging to the given topics (>0.9) one can find the "main" topics of articles. Below is an example of this.

![Example article](/Images/ExampleArticle.png)