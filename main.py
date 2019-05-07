#pytorch package
import torch 
from torch.utils.data import Dataset, DataLoader
#local python 
from mydataset import MyDataset
import model
import train
#python package
import numpy as np 
import pandas as pd
import argparse
import time

parser = argparse.ArgumentParser(description='CNN text classifier')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=128, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=16, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
parser.add_argument('-num-workers', type=int, default=0, help='how many subprocesses to use for data loading, 0 means that the data will be loaded in the main process')
parser.add_argument('-npdir', type=str, default='./numpy_saved', help='dir name of saved numpy')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=int, default=3, help='kernel size to use for convolution')
parser.add_argument('-channels', type=int, default=3, help='number of channels of input')
parser.add_argument('-classes', type=int, default=3, help='number of classes to predict')
parser.add_argument('-dim0', type=int, default=16, help='size of dimension 0 of input')
parser.add_argument('-dim1', type=int, default=16, help='size of dimension 1 of input')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 means cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', action='store_true', default=False, help='predict the sentence given')
parser.add_argument('-submit', type=str, default='submission', help='filename of predicted output [default: None]')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
#parser.add_argument('-train', action='store_true', default=False, help='train or test')
args = parser.parse_args()

#./main for training
if (args.predict is not True) and (args.test is not True):
	print('\nLoad train data.')
	start = time.time()
	#train_numpy_data = np.load(args.npdir+'/train_input1.npy')
	train_numpy_data2 = np.load(args.npdir+'/train_input2.npy')
	train_numpy_target = np.load(args.npdir+'/train_label.npy')
	train_dataset = MyDataset(train_numpy_data2, train_numpy_target)
	trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
		num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
	print('Spend {} sec to load {} GB np array.'.format\
		(round(time.time()-start, 2), \
		train_numpy_data2.size * train_numpy_data2.itemsize / 1024 / 1024 / 1024))
#./main for evaluation
if (args.test is True) or ((args.predict is not True and args.test is not True) is True):
	print('\nLoad eval data')
	start = time.time()
	#eval_numpy_data = np.load(args.npdir+'/eval_input1.npy')
	eval_numpy_data2 = np.load(args.npdir+'/eval_input2.npy')
	eval_numpy_target = np.load(args.npdir+'/eval_label.npy')
	eval_dataset = MyDataset(eval_numpy_data2, eval_numpy_target)
	evalloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, 
		num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
	print('Spend {} sec to load {} MB np array.'.format\
		(round(time.time()-start, 2), \
		eval_numpy_data2.size * eval_numpy_data2.itemsize / 1024 / 1024))
#./main for prediction
if args.predict is True:
	print('\nLoad test data')
	start = time.time()
	#est_numpy_data = np.load(args.npdir+'/test_input1.npy')
	test_numpy_data2 = np.load(args.npdir+'/test_input2.npy')
	test_numpy_ids = np.load(args.npdir+'/test_id.npy')
	test_dataset = MyDataset(test_numpy_data2, test_numpy_ids)
	testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
		num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
	print('Spend {} sec to load {} MB np array.'.format\
		(round(time.time()-start, 2), \
		test_numpy_data2.size * test_numpy_data2.itemsize / 1024 / 1024))

#Model
cnn = model.Net(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn.cuda(device)

#to load the pre-trained and saved model from snapshot .pn file
if args.snapshot is not None:
	print('\nLoading model from {}\n'.format(args.snapshot))
	cnn.load_state_dict(torch.load(args.snapshot))

#train, test or predict
#to predict the test file without label, which is the submission
if args.predict is True:
	ids, labels = train.predict(testloader, cnn)
	df = pd.DataFrame(columns = ['Id', 'Category'])
	df['Id'] = ids
	df['Category'] = labels
	df.to_csv('./predicted/{}.csv'.format(args.submit), encoding='utf-8', index=False)
	print('\nFinish Prediction, and save the prediction to csv file.\n')

#to eval on the eval data with label, 
#BUT the eval data is the validation set during training
elif args.test is True:
	try:
		train.eval(evalloader, cnn)
	except Exception as e:
		print('\nSorry. The test dataset does not exist.\n')
#to train the model
else:
	print()
	try:
		print('\nrun main.py to train as default....\n')
		r, s, a = train.train(trainloader, evalloader, cnn, args)
		if r == -1:
			print('Early stop training at {} steps with best accuracy: {}.'.format(s, a))
		if r ==1:
			print('Finish training as a whole with beat accuracy: {}%.'.format(a))
	except KeyboardInterrupt:
		print('\n' + '-'*80)
		print('Existing from training early')
		exit(0)

#train.train(trainloader, evalloader, cnn, args)
#train.eval(evalloader, cnn, args)