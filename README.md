# MusicAnalysis
Machine Learning model using sentiment analysis and various learning algorithms on song features to predict artists' gender.
We will investigate the trends in how our model predicts gender over time based on the training data.

In order to run this project, with our prebuilt model, extract best_model.h5 from best_model.zip into the project's primary dirctory

Then run ```python3 buildModels.py```.

If you wish to train your own Sentiment Analysis model, download a pretrained GloVe vector from Stanford at https://nlp.stanford.edu/projects/glove/ and extract it to Databases/glove. We use Common Crawl42B, but any of them should work, just modify the feature dimension variable in buildSA.py. Then run buildSA.py. In our experience it took anywhere from 4 hours (using Tensorflow-gpu on a GTX950M) to 12 hours running on just cpu. buildSA.py will create model checkpoints of the best model to arise during the training process.
