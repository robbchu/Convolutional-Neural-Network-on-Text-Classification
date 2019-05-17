'''
make the dataloader for later model usage
'''
import torch 
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors, Word2Vec

import os
import zipfile
import numpy as np 
import pandas as pd
import jieba
import time
import argparse

#the column name to index dict for usage of select item when df.itertuples
cols_dict = {'tid1':1, 'tid2':2, 'title1_zh':3, 'title2_zh':4, 'label':5}
#the label to index dict for usage of convert label classes numpy array
label_dict = {'agreed':0, 'disagreed':1, 'unrelated':2}

class MyDataset(Dataset):
	def __init__(self, data1, data2, target, transform=None):
		self.data1 = data1
		self.data2 = data2
		self.target = target
		self.transform = transform

	def __getitem__(self, index):
		x1 = torch.from_numpy(self.data1[index]).float()
		x2 = torch.from_numpy(self.data2[index]).float()
		y = torch.from_numpy(np.array(self.target[index])).long()

		if self.transform:
			x1 = self.transform(x1)
			x2 = self.transform(x2)

		return x1, x2, y

	def __len__(self):
		return len(self.data1)
'''
class MyDataset(Dataset):
	def __init__(self, data1, target, transform=None):
		self.data1 = torch.from_numpy(data1).float()
		#self.data2 = torch.from_numpy(data2).float()
		self.target = torch.from_numpy(target).long()
		self.transform = transform

	def __getitem__(self, index):
		x1 = self.data1[index]
		#x2 = self.data2[index]
		y = self.target[index]

		if self.transform:
			x1 = self.transform(x1)
			#x2 = self.transform(x2)

		return x1, y

	def __len__(self):
		return len(self.data1)
'''

def loaddata():
	#Load train, test data path, 
	if os.path.isfile(DATAPATH + 'fake-news-pair-classification-challenge.zip'):
		zip_file = zipfile.ZipFile(DATAPATH + 'fake-news-pair-classification-challenge.zip')
		TRAIN_CSV_PATH = zip_file.open('train.csv')
		TEST_CSV_PATH = zip_file.open('test.csv')
		#TOKENIZED_TRAIN_CSV_PATH = "../input/siamese-network-lstm/tokenized_train.csv"
		TOKENIZED_TRAIN_CSV_PATH = ""
	else:
		TRAIN_CSV_PATH = WORK_DIR+'input/train.csv'
		TEST_CSV_PATH = WORK_DIR+'input/test.csv'
		TOKENIZED_TRAIN_CSV_PATH = ""

	train = pd.read_csv(TRAIN_CSV_PATH, index_col='id')
	test = pd.read_csv(TEST_CSV_PATH, index_col='id')
	cols = ['tid1', 'tid2', 'title1_zh', 'title2_zh', 'label']
	train = train.loc[:, cols]
	test = test.loc[:, cols]
	train.fillna('UNKNOWN', inplace=True)
	test.fillna('UNKNOWN', inplace=True)

	#and then split the train into train/validation
	VALIDATION_RATIO = 0.1
	RANDOM_STATE = 9527
	train, val = train_test_split(train, test_size=VALIDATION_RATIO, random_state=RANDOM_STATE)

	return train, val, test

#by jieba, tokenize the sentence, then store the segments without stopwords according to the title id.
def tid2text(data):
	tid2text_dict = {}
	stopwords = []

	with open(WORD_DIR+'jieba_extra/stopword.txt', 'r', encoding='UTF-8') as f:
		for line in f:
			stopwords.append(line.strip('\n'))

	for item in data.itertuples():
		if item[cols_dict['tid1']] not in tid2text_dict.keys():
			#segments = jieba.cut(item[cols_dict['title1_zh']], cut_all=False)
			#tid2text_dict[item[cols_dict['tid1']]] = list(filter(lambda x: x not in stopwords and x != '\n', segments))
			tid2text_dict[item[cols_dict['tid1']]] = \
				[token for token in jieba.cut(item[cols_dict['title1_zh']], cut_all=False) if token not in stopwords]

		if item[cols_dict['tid2']] not in tid2text_dict.keys():
			#segments = jieba.cut(item[cols_dict['title2_zh']], cut_all=False)
			#tid2text_dict[item[cols_dict['tid2']]] = list(filter(lambda x: x not in stopwords and x != '\n', segments))

			tid2text_dict[item[cols_dict['tid2']]] = \
				[token for token in jieba.cut(item[cols_dict['title2_zh']], cut_all=False) if token not in stopwords]

	return tid2text_dict

unique_tid = 161182 
tokens_count = 1592927
mean_token_count = 9.8
var_token_count = 7.6
dev_token_count = 2.7

def tokWV(token, wordvectors, args):
	seg_count = 0
	vector = np.zeros(args.wvdim)
	if token in wordvectors.vocab:
		vector = wordvectors[token]
		seg_count += 1
	else:
		for char in token:
			if char in wordvectors.vocab:
				vector += wordvectors[char]
				seg_count += 1

	if seg_count == 0:
		return vector
	else:
		return vector / seg_count

def df2np(data, wordvectors, fakedict, args, Test=False):

	tid2text_dict = tid2text(data) 

	len_df = len(data)
	channel = 3
	MAX_LEN = 16 #decide from the mean token count and the variance of it by 67-95-98:1-2-3dev
	WV_DIM = args.wvdim
	classes = 3

	input_npdata = np.zeros(len_df*channel*MAX_LEN*MAX_LEN).reshape(len_df, channel, MAX_LEN, MAX_LEN)
	output_npdata = np.zeros(len_df, dtype=int).reshape(len_df)

	index = 0
	for item in data.itertuples():
		if (index % (len_df // 5)) == 0:
			print("Fininshed df2np: {:.2f}% ({}/{})".format(index/len_df*100, index, len_df))
		title1 = tid2text_dict[item[cols_dict['tid1']]]
		title2 = tid2text_dict[item[cols_dict['tid2']]]
		
		if Test is not True:
			label = item[cols_dict['label']]
			output_npdata[index] = label_dict[label]
		elif Test is True:
			pairtid = item[0]
			output_npdata[index] = pairtid
		
		#build the 3-channel figure from text pairs(title1, title2)
		for i, tok1 in enumerate(title1):
			if i >= MAX_LEN: 
				break
			for j, tok2 in enumerate(title2):
				if j >= MAX_LEN: 
					break
				#the 1st channel, whether the tokens are matched
				if (tok1 in fakedict) and (tok2 in fakedict):
					input_npdata[index][0][i][j] = fakedict[tok1] + fakedict[tok2]
				elif tok2 in fakedict:
					input_npdata[index][0][i][j] = fakedict[tok2]
				elif tok1 in fakedict:
					input_npdata[index][0][i][j] = fakedict[tok1]

				#the 2nd channel, the cosine similarity between tokens
				if (tok1 not in wordvectors.vocab) or (tok2 not in wordvectors.vocab):
					input_npdata[index][1][i][j] = 0
					#input_npdata[index][1][j][i] = 0
					input_npdata[index][2][i][j] = 0
					#input_npdata[index][2][j][i] = 0
					continue
				wv1 = tokWV(tok1, wordvectors)
				wv2 = tokWV(tok2, wordvectors)
				cosine_similarity = np.sum(wv1 * wv2) / np.linalg.norm(wv1) / np.linalg.norm(wv2)
				input_npdata[index][1][i][j] = cosine_similarity
				#input_npdata[index][1][j][i] = cosine_similarity
				#the 3rd channel, the distance between tokens
				euclidean_distance = np.linalg.norm(wv1 - wv2)
				input_npdata[index][2][i][j] = euclidean_distance
				#input_npdata[index][2][j][i] = euclidean_distance

		index += 1

	return input_npdata, output_npdata

def df2wvMatrix(data, wordvectors, args):

	tid2text_dict = tid2text(data) 

	len_df = len(data)
	channel = 1
	MAX_LEN = 16 #decide from the mean token count and the variance of it by 67-95-98:1-2-3dev
	WV_DIM = args.wvdim
	classes = 3
	input_npdata2 = np.zeros(len_df*channel*(MAX_LEN*2)*WV_DIM).reshape(len_df, channel, MAX_LEN*2, WV_DIM)

	index = 0
	for item in data.itertuples():
		if (index % (len_df // 5)) == 0:
			print("Fininshed df2wvMatrix: {:.2f}% ({}/{})".format(index/len_df*100, index, len_df))
		title1 = tid2text_dict[item[cols_dict['tid1']]]
		title2 = tid2text_dict[item[cols_dict['tid2']]]
		
		#build the 3-channel figure from text pairs(title1, title2)
		for i, tok1 in enumerate(title1):
			if i >= MAX_LEN: 
				break
			if tok1 not in wordvectors.vocab:
				continue
			wv1 = tokWV(tok1, wordvectors)
			input_npdata2[index][0][i] = wv1
		for j, tok2 in enumerate(title2):
			if j >= MAX_LEN: 
				break
			if tok2 not in wordvectors.vocab:
				continue
			wv2 = tokWV(tok2, wordvectors)
			input_npdata2[index][0][MAX_LEN+j] = wv2

		index += 1

	return input_npdata2

if __name__=="__main__":

	DATAPATH = '/home/robert/Dataset/'
	WORD_DIR = '/home/robert/python/'
	WV_MODEL1 = 'word2vec_pretrained/zh_wiki_word2vec_300.txt'
	WV_MODEL2 = 'word2vec_pretrained/word2vec_yee100/word2vec.model'
	FAKEDICT = 'fakedict/fakedict_sumto1.npy'

	parser = argparse.ArgumentParser(description='Dataset builder')
	parser.add_argument('-wv-dir', type=str, default=WV_MODEL2, help='where to load the word2vec model')
	parser.add_argument('-wvdim', type=int, default=100, help='the dimension of word vectors')
	args = parser.parse_args()

	train, val, test =loaddata()
	print('Loading model from Gensim by pretrained Word2Vec model: {}'.format(args.wv_dir))
	start = time.time()
	if args.wv_dir == WV_MODEL2: #default: load 100-dim word2vec model
		wvmodel = Word2Vec.load(WORD_DIR+WV_MODEL2)
		wordvectors = wvmodel.wv
	if args.wv_dir != WV_MODEL2: #Load 300-dim word2vec model
		wordvectors = KeyedVectors.load_word2vec_format(WORD_DIR+args.wv_dir, binary=False)		
	print('Loaded model costs {} seconds'.format(round(time.time()-start, 2)))

	#transform to the numpy of word vectors
	print('\ndf2wvMatrix of train data\n')
	train_input_npdata2 = df2wvMatrix(train, wordvectors, args)
	np.save('./numpy_saved/train_input2.npy', train_input_npdata2)
	print('\ndf2wvMatrix of eval data\n')
	eval_input_npdata2 = df2wvMatrix(val, wordvectors, args)
	np.save('./numpy_saved/eval_input2.npy', eval_input_npdata2)
	print('\ndf2wvMatrix of test data\n')
	test_input_npdata2 = df2wvMatrix(test, wordvectors, args)
	np.save('./numpy_saved/test_input2.npy', test_input_npdata2)


	#transform to the numpy of 3 channel
	fakedict = np.load(FAKEDICT).item()
	print('\nLoad the fact dictionary of length {}.\n'.format(len(fakedict)))
	#wordvectors = wv_model.wv
	#tid2text_dict,_, _ = tid2text(train)
	print('\ndf2np of train data\n')
	train_input_npdata, train_label_npdata = df2np(train, wordvectors, fakedict, args)
	np.save('./numpy_saved/train_input1.npy', train_input_npdata)
	np.save('./numpy_saved/train_label.npy', train_label_npdata)
	
	print('\ndf2np of eval data\n')
	eval_input_npdata, eval_label_npdata = df2np(val, wordvectors, fakedict, args)
	np.save('./numpy_saved/eval_input1.npy', eval_input_npdata)
	np.save('./numpy_saved/eval_label.npy', eval_label_npdata)

	print('\ndf2np of test data\n')
	test_input_npdata, test_id_npdata = df2np(test, wordvectors, fakedict, args, Test=True)
	np.save('./numpy_saved/test_input1.npy', test_input_npdata)
	np.save('./numpy_saved/test_id.npy', test_id_npdata)