import torch 
import torch.nn as nn
import torch.nn.functional as F
'''
for batch_idx, (data, target) in enumerate(loader):
	print(batch_idx, data.shape, target.shape)
'''
classes = ('agreed', 'disagreed', 'unrelated')

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
		self.bn0 = nn.BatchNorm2d(Ci)
		self.conv1 = nn.Conv2d(Ci, Ci, K, padding=True)#(, 3, 16, 16)#(input_channel, output_channel, kernel_size)
		self.bn1 = nn.BatchNorm2d(Ci)
		self.conv2 = nn.Conv2d(Ci, Ci*2, K, padding=True)#(, 6, 16, 16)
		self.bn2 = nn.BatchNorm2d(Ci*2)
		self.conv3 = nn.Conv2d(Ci*2, Ci*3, K, padding=True)
		self.bn3 = nn.BatchNorm2d(Ci*3)

		self.pool1 = nn.MaxPool2d(2, 1)#(kernel_size, stride)
		self.pool2 = nn.MaxPool2d(2, 2)

		self.dropout = nn.Dropout(DROPOUT)
		self.fc1 = nn.Linear(Ci*3 * ((W//2//2)-1) * ((H//2//2)-1), 48)
		self.fc2 = nn.Linear(48, 24)
		self.fc3 = nn.Linear(24, C)

	def forward(self, x):

		if self.args.static:
			x = Variable(x)

		x = self.bn0(x)
		x = self.pool2(F.relu(self.bn1(self.conv1(x))))
		x = self.pool2(F.relu(self.bn2(self.conv2(x))))
		x = self.pool1(F.relu(self.bn3(self.conv3(x))))
		x = x.view(-1, 3*3 * ((16//2//2)-1) * ((16//2//2)-1))
		x = self.dropout(x)
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		x = F.relu(self.fc2(x))
		x = self.dropout(x)
		x = F.softmax(self.fc3(x), dim=1)

		return x