import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(trainloader, evalloader, model, args):

	model.to(device)

	'''the optional argument weight assigned to each of the classes. 
	This is particularly useful when you have an unbalanced training set.'''	
	weights = torch.tensor([1/15 *5, 1/5 *5, 1/16 *5], device=device)
	criterion = nn.CrossEntropyLoss(weight = weights)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	steps = 0 # total iteration of whole epochs and batches
	best_acc = 0
	last_step = 0
	continued_earlystop = 0
	model.train()
	EPOCH = args.epochs
	BATCH_SIZE = args.batch_size
	batch_count = 0
	for epoch in range(EPOCH):

		running_loss = 0.0

		for i, batch in enumerate(trainloader, 0):
			batch_count += 1
			# get the inputs
			inputs, labels = batch
			inputs, labels = inputs.to(device), labels.to(device)
			#inputs, labels = [x.to(device) for x in inputs], labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(inputs)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			steps += 1
			#print some loss, accuracy satistics on training every 2000 steps
			running_loss += loss.item()
			if i % 1500 == 1499:    # print every 2000 mini-batches
				corrects = (torch.max(outputs, 1)[1] == labels).sum().item()
				accuracy = 100 * corrects / labels.size(0)
				print('[In epoch {}, the {} iteration] - loss: {:.6f}, acc: {:.4f}'.\
					format(epoch + 1, i + 1, running_loss / 1500, accuracy))
				
				'''sys.stdout.write('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format\
					(steps, running_loss / 2000, accuracy, corrects, inputs.size()[0]))
				'''
				running_loss = 0.0
			#save the bset model so far by evaluating on eval data
			if steps % args.test_interval == 0:
				dev_acc = eval(evalloader, model)
				#dev_acc = 0
				if dev_acc > best_acc:
					best_acc = dev_acc
					last_step = steps
					continued_earlystop = 0
					if args.save_best:
						torch.save(model.state_dict(), './models/snapshot_best_{}.pth'.format(steps))
				else:
					if steps - last_step >= args.early_stop:
						continued_earlystop += 1
						if continued_earlystop > (9016 * 10) and epoch > 32:
							return -1, steps, best_acc
						print('early stop at {} steps with continually not increasing eval acc after 10 batched.'.format(steps))
			#save model every interval steps
			elif steps % args.save_interval == 0:
				torch.save(model.state_dict(), './models/snapshot_{}.pth'.format(steps))
			if batch_count == math.ceil(len(trainloader.dataset)//BATCH_SIZE) and epoch ==0:
				print('Batch count {} matches the dataset of len {} with batch size {}.'\
					.format(batch_count, len(trainloader.dataset), BATCH_SIZE))
				batch_count = 0

		#save the latest model snapshot
		if epoch+1 == EPOCH:
			torch.save(model.state_dict(), './models/snapshot_last.pth')   

	return 1, steps, best_acc

def eval(evalloader, model):

	model.eval()

	corrects = 0
	avg_loss = 0
	total = 0
	batch_count = 0
	with torch.no_grad():
		for batch in evalloader:
			batch_count += 1
			inputs, labels = batch
			#only one input figure
			inputs, labels = inputs.to(device), labels.to(device)
			#two input figure, halt bc OOM
			#inputs, labels = [x.to(device) for x in inputs], labels.to(device)

			outputs = model(inputs)
			loss = F.cross_entropy(outputs, labels, reduction='sum')
			avg_loss += loss.item()

			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			corrects += (predicted == labels).sum().item()

	#print('Finished Evaluation')

	avg_loss /= total
	accuracy = 100 * corrects / total
	print('\nEvaluation on {} batches of validation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format\
		(batch_count, avg_loss, accuracy, corrects, total))		

	return accuracy


def predict(testloader, model):

	model.eval()

	label_dict = {0:'agreed', 1:'disagreed', 2:'unrelated', -1:'UNKNOWN'}

	ids_list = []
	predicted_text = []
	with torch.no_grad():
		for batch in testloader:
			inputs, ids = batch
			inputs, ids = inputs.to(device), ids

			outputs = model(inputs)

			_, predicted = torch.max(outputs.data, 1)	
			
			ids_list.extend([x.item() for x in ids])
			predicted_text.extend([label_dict[x.item()] for x in predicted])

	return ids_list, predicted_text