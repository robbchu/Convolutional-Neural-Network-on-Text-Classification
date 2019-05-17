'''
Build a fake dictionary from training data:
build the dictionary from fake news title, 
saving those tokens and their appearance count, 
then normalize the weights.
'''
from gensim.models import KeyedVectors

import os
import zipfile
import numpy as np 
import pandas as pd
import jieba
import time

DATAPATH = '/home/robert/Dataset/'
WORD_DIR = '/home/robert/python/'
WV_MODEL = 'word2vec_pretrained/zh_wiki_word2vec_300.txt'

#the column name to index dict for usage of select item when df.itertuples
cols_dict = {'tid1':1, 'tid2':2, 'title1_zh':3, 'title2_zh':4, 'label':5}
#the label to index dict for usage of convert label classes numpy array
label_dict = {'agreed':0, 'disagreed':1, 'unrelated':2}

def loaddata():
	#Load train, test data path, 
	if os.path.isfile(DATAPATH + 'fake-news-pair-classification-challenge.zip'):
		zip_file = zipfile.ZipFile(DATAPATH + 'fake-news-pair-classification-challenge.zip')
		TRAIN_CSV_PATH = zip_file.open('train.csv')
	else:
		TRAIN_CSV_PATH = WORK_DIR+'input/train.csv'

	train = pd.read_csv(TRAIN_CSV_PATH, index_col='id')
	cols = ['tid1', 'tid2', 'title1_zh', 'title2_zh', 'label']
	train = train.loc[:, cols]
	train.fillna('UNKNOWN', inplace=True)

	return train

#by jieba, tokenize the sentence, then store the segments without stopwords according to the title id.
def buildfakedict(data):
	stopwords = []
	fakedict = {}

	with open(WORD_DIR+'jieba_extra/stopword.txt', 'r', encoding='UTF-8') as f:
		for line in f:
			stopwords.append(line.strip('\n'))

	for item in data.itertuples():

		tokens = [token for token in jieba.cut(item[cols_dict['title1_zh']], cut_all=False) if token not in stopwords]
		if item[cols_dict['label']] == 'agreed':
			tokens.extend([token for token in jieba.cut(item[cols_dict['title2_zh']], cut_all=False) if token not in stopwords])

		for tok in tokens:
			if tok not in fakedict:
				fakedict[tok] = 1
			else:
				fakedict[tok] += 1

	return fakedict

if __name__ == '__main__':
	train = loaddata()
	print('\nUse the whole train data of length {}.\n'.format(len(train)))
	fakedict = buildfakedict(train)
	print('\nThere are {} tokens in fakedict, and the total occurence is {}.\n'.format(len(fakedict), sum(fakedict.values())))
	
	np.save('fakedict.npy', fakedict)