# Fake-News-Classification-by-Kaggle-Competition

The repository contains python file for the Kaggle competition on fake news classification
* **mydataset.py** is the script of the custom Dataset module to be later used by DataLoader. 
  So it first loads the dataset and pre-processes on them, and save the resulting numpy array into dir ./numpy_saved.
* **main.py** is the main program to run either training by the dataset, evaluate on the validation set, or predict the label on the test set.
* **model.py** is the modelling of Convolutional Neural Network, which would be import as package in main.py
* **train.py** is the script of training process, which also includes the evaluation and prediction as they are usually called after training.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites
* Python 3.6.7
* PyTorch 0.4.1
* pre-trained Simplified Chinese Word2Vec file. (.txt as default)
* Dataset used here is downloaded from the Kaggle competition on WSDM2019-Fake News Classification.
* Also, change some hard coded file loading path in the script to load them from local. 
  Otherwise, they are expected to be created locally:
    * ./numpy_saved to save and load the pre-processed numpy array, which is later used in DataLoader.
    * ./word2vec_pretrained to load the pre-trained word vectors.
    * ./models to save the trained model every time after training.
* There are some other arguments can be passed to **main.py**. For details, refer to the ```argparse``` section in it.


## Running the scripts

### Train the model
```
python3 main.py #run training as default, and save the trained model in dir ./model
```
### Evaluate the trained model
```
python3 main.py -test -snapshot './models/saved_model.pth' #evaluate the model loaded from saved state dict of ```.pth```
```
### Predict on test set
```
python3 main.py -predict -snapshot './models/saved_model.pth' #save the predicted output ```submission.csv```
```
Finally, you can submit your output.csv to the Kaggle to get private and public score on leaderboard.


## Built With

* [PyTorch](https://pytorch.org/) - The deep learning platform
* [Gensim](https://radimrehurek.com/gensim/index.html/) - Topic modelling for human


## Authors

* **Robert Chu** - *Initial work* - [robbchu](https://github.com/robbchu)


## Reference

* [WSDM - Fake News Classification](https://www.kaggle.com/c/fake-news-pair-classification-challenge/overview)
