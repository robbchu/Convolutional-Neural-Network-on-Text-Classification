# Fake News Classification by Kaggle Competition

The repository contains files for the Kaggle competition on fake news classification.

In a nutshell, we implement this inference task by Convolutional Neural Network with 2 different design on input. And we find the result achieves the best score by combination of them. For detail, please refer to the [report](https://drive.google.com/open?id=1Ge4FcjNUcLbFVbhMke4qKA7DLaTU8Cxm) of this project.

* **fakedict.py** is the script of building the fake dictionary for later use.
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
    * ```./numpy_saved``` to save and load the pre-processed numpy array, which is later used in DataLoader.
    * ```./word2vec_pretrained``` to load the pre-trained word vectors.
    * ```./models``` to save the trained model every time after training.
* There are some other arguments can be passed to **main.py**. For details, refer to the ```argparse``` section in it.


## Running the scripts

To run the scripts for either training, evaluation, or prediction, you have to specified the model-type. They are ```type1```:3-channel input and ```type2```:1-channel input; also, the combination of them can be specified as ```both```.

### Prepare the data
To implement the method of 3-channel input in the report, we have to build the fake dictionary first from the dataset by running fakedict.py.
```
python3 fakedict.py #save the dictionary as fakedict.npy
```
Also, we have to first transform the raw dataset into numpy and split them into train and validation sets by running mydataset.py.
```
python3 mydataset.py #save the results in dir ./numpy_saved
```

### Train the model
run training as default, and save the trained model in dir ```./model```.
```
python3 main.py -model-type='both' #run training as default, and save the trained model in dir ./model
```
### Evaluate the trained model
evaluate the model loaded from the saved state dict of ```.pth```.
```
python3 main.py -test -model-type='both' -snapshot './models/saved_model.pth' 
```
### Predict on test set
predict and save results as ```submission.csv```.
```
python3 main.py -predict -model-type='both' -snapshot './models/saved_model.pth' 
```
Finally, you can submit your output.csv to Kaggle to get private and public score on leaderboard.


## Built With

* [PyTorch](https://pytorch.org/) - The deep learning platform
* [Gensim](https://radimrehurek.com/gensim/index.html/) - Topic modelling for human


## Authors

* **Robert Chu** - *Initial work* - [robbchu](https://github.com/robbchu)


## Reference

* [WSDM - Fake News Classification](https://www.kaggle.com/c/fake-news-pair-classification-challenge/overview)