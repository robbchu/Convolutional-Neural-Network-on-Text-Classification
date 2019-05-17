'''
This is the scripts of the models.
There are two types of CNN architecture used here,
the used type can be specified in the passing argument in main.py.
-model-type='type1' is the method b of CNN mentioned in the report, 
 			which is implemented with a 3-channel input matrices, 
 			with 3-layers of convolution and 1 kernel of size 3 in each,
 			and the feature maps output by 3 layers are of size 3, 6, 9.
-model-type='type2' is the method a of CNN mentioned in the report, 
			which is implemented with a 1-channel input matrix, 
			with 1-layer of convolution and 3 kernels used of size 3, 4, 5,
			and the feature maps output by 3 kernels share the same size of 50.
-model-type='both' is the combined of method a and method b, 
			which combine the configuration of type1 and type2 mentioned above,
			then concatenate their output for the final output as a whole.
'''
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
'''
for batch_idx, (data, target) in enumerate(loader):
	print(batch_idx, data.shape, target.shape)
'''

#(batch, 3, 24, 24)
class Net(nn.Module):#subclass in torch.nn.Module
	def __init__(self, args):
		super(Net, self).__init__()
		self.args = args

		Ci = args.channels#channel 3
		C = args.classes#class numbers 3
		W = args.dim0#16
		H = args.dim1#16
		K = args.kernel_sizes#3

		DROPOUT = args.dropout
		#Convolution Layer
		if self.args.model_type == 'type1' or self.args.model_type == 'both':
			#for x1 (type1), batch * 3-channel * 16 *16
			self.bn0 = nn.BatchNorm2d(Ci)
			self.conv1 = nn.Conv2d(Ci, Ci, K, padding=True)#(, 3, 16, 16)#(input_channel, output_channel, kernel_size)
			self.bn1 = nn.BatchNorm2d(Ci)
			self.conv2 = nn.Conv2d(Ci, Ci*2, K, padding=True)#(, 6, 16, 16)
			self.bn2 = nn.BatchNorm2d(Ci*2)
			self.conv3 = nn.Conv2d(Ci*2, Ci*3, K, padding=True)
			self.bn3 = nn.BatchNorm2d(Ci*3)

			#for x1
			self.pool1 = nn.MaxPool2d(2, 1)#(kernel_size, stride)
			self.pool2 = nn.MaxPool2d(2, 2)

			#for x1
			self.fc1 = nn.Linear(Ci*3 * ((W//2//2)-1) * ((H//2//2)-1), 48)
			self.fc2 = nn.Linear(48, 24)
			self.fc3 = nn.Linear(24, C)
		#for x2 (type2), batch * 1-channel(2 title concat) * 32(MAXLEN*2) * 300(wordvector dim)
		if self.args.model_type == 'type2' or self.args.model_type == 'both':
			Co = 50
			Ks = [3, 4, 5]
			D = 100
			self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
			self.bn0_ = nn.BatchNorm2d(Co)

			#for x2
			self.fc1_ = nn.Linear(len(Ks) * Co, C)
			'''
			self.conv1_ = nn.Conv2d(1, 10, (3, 300))
			self.conv2_ = nn.Conv2d(1, 10, (4, 300))
			self.conv3_ = nn.Conv2d(1, 10, (5, 300))
			'''
		#Pooling Layer
		#for x2: call F.maxpool1d() in forward instead, avoiding to forehead declare nn module

		#Linear Layers
		self.dropout = nn.Dropout(DROPOUT)
		
		#for x1+x2 (both)
		if self.args.model_type == 'both':
			#concat 2 CNN output
			self.fc_last = nn.Linear(C*2, C)

	def forward(self, x):
		
		#x1 = x[0]
		#x2 = x[1]

		if self.args.static:
			x1 = Variable(x1)
			x2 = Variable(x2)
		if self.args.model_type == 'type1':
			x1 = x[0]
		elif self.args.model_type == 'type2':
			x2 = x[0]
		elif self.args.model_type == 'both':
			x1 = x[0]
			x2 = x[1]

		if self.args.model_type == 'type1' or self.args.model_type == 'both':
			x1 = self.bn0(x1)
			x1 = self.pool2(F.relu(self.bn1(self.conv1(x1))))
			x1 = self.pool2(F.relu(self.bn2(self.conv2(x1))))
			x1 = self.pool1(F.relu(self.bn3(self.conv3(x1))))
			x1 = x1.view(-1, 3*3 * ((16//2//2)-1) * ((16//2//2)-1))
			#x1 = self.dropout(x1)
			x1 = F.relu(self.fc1(x1))
			x1 = self.dropout(x1)
			x1 = F.relu(self.fc2(x1))
			x1 = self.dropout(x1)
			x1 = self.fc3(x1)

		if self.args.model_type == 'type1':
			x1 = F.softmax(x1, dim=1)
			return x1
		#x1 = F.softmax(self.fc3(x1), dim=1)
		

		#x2 = x2.unsqueeze(1)
		'''
		squeeze() remove the dimension that contains only 1 element
		eg. (batch, 1, MAXLEN, D) --conv2d(1, Co, (k, D))--> (batch, Co, (MAXLEN-k+1), 1)
		squeeze() the 3rd dimension -> (batch, Co, (MAXLEN-k+1)) 
		and then max over time pooling --maxpool1d()-->
		'''
		if self.args.model_type == 'type2' or self.args.model_type == 'both':
			x2 = [F.relu(self.bn0_(conv(x2))).squeeze(3) for conv in self.convs1]
			x2 = [F.max_pool1d(fmap, fmap.size(2)).squeeze(2) for fmap in x2]
			x2 = torch.cat(x2, dim=1)
			x2 = self.dropout(x2)
			x2 = self.fc1_(x2)

		if self.args.model_type == 'type2':
			x2 = F.softmax(x2, dim=1)
			return x2

		if self.args.model_type == 'both':
			y = torch.cat([x1, x2], dim=1)
			y = self.fc_last(y)
			y = F.softmax(y, dim=1)
			return y