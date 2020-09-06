import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb


# TODO
saver_path = "." # YOUR PATH TO SAVE THE LOG
writer = SummaryWriter(saver_path)


epoch = 20
batch_size = 100
learning_rate = 0.001

# Download Data

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

# Set Data Loader(input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=batch_size,shuffle=True)


# SET DEVICE (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):

	# NOTE there is one bug in this model
	# TODO
	def __init__(self):
		super(Model, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 32, 3, padding=1),
			nn.ReLU(),

			nn.Conv2d(32, 32, 3, padding=1),
			nn.ReLU(),

			nn.Conv2d(32, 32, 3, padding=1),
			nn.ReLU(), 						#Activation function was missing (the bug)
		)

		self.layer2 = nn.Sequential(
			torch.nn.Linear(32*28*28, 256, bias=True),
			nn.ReLU(),
			torch.nn.Linear(256, 10, bias=True),
		)

	def forward(self, x):

		out = self.layer1(x)
		out = out.view(-1,32*28*28) # Flatten
		out = self.layer2(out)

		return out


model = Model().to(device=device)



parameters = list(model.parameters())
loss_func = nn.CrossEntropyLoss()

# TODO : Fill in your favourite optimizer
optimizer = torch.optim.Adam(parameters, lr=learning_rate)



def accuracy_measure(logits,labels):
	"""Accuracy
		Args:
			logits : logits of shape [B, 10]
			labels : labels of shape [B, ]
		Returns:
		   Tensor: accuracy (scalar)
		"""
  # TODO: fill the accuracy implementation
	predicted = torch.argmax(logits, axis=1)
	return (predicted == labels).sum()


for i in range(epoch):


	loss_total = 0.
	loss_test_total = 0.

	accuracy_train = 0.
	accuracy_test = 0.



	## TRAIN
	for idx, dp in enumerate(tqdm(train_loader)):
		# THIS MEANS ALL THE PARAMETERS ARE IN TRAIN MODE
		model.train()

		# NOTE there is one bug in this train loop
		# TODO
		optimizer.zero_grad()			#This was missing (the bug)

		image,label = dp

		input_matrix = image.to(device)					#Another bug(.to(device))

		label_network_prediction = model(input_matrix)
		label_ground_truth = label.to(device)			#(.to(device))

		loss = loss_func(label_network_prediction, label_ground_truth.long())

		# accuracy
		accuracy = accuracy_measure(label_network_prediction,label_ground_truth)

		accuracy_train += accuracy



		loss.backward()
		optimizer.step()

		writer.add_scalar('training loss',
						  loss.item(),
						  i * len(train_loader) + idx)
		writer.add_scalar('training accuracy',
						  accuracy.item(),
						  i * len(train_loader) + idx)

		loss_total +=loss






	## TEST

	for idx_test, dp_test in enumerate(tqdm(test_loader)):
		# THIS MEANS ALL THE PARAMETERS ARE IN EVAL MODE

		optimizer.zero_grad()

		with torch.no_grad():

			image_test,label_test = dp_test

			input_matrix_test = image_test.to(device)

			label_network_prediction_test = model(input_matrix_test)
			label_ground_truth_test = label_test.to(device)

			loss_test = loss_func(label_network_prediction_test, label_ground_truth_test.long())
			accuracy_t = accuracy_measure(label_network_prediction_test,label_ground_truth_test)

			accuracy_test += accuracy_t

			writer.add_scalar('validation loss',
							  loss_test.item(),
							  i * len(test_loader) + idx_test)
			writer.add_scalar('validation accuracy',
							  accuracy_t.item(),
							  i * len(test_loader) + idx_test)

			loss_test_total += loss_test





	print('epoch done: ', i)
	#torch.save([model], saver_path + 'model_1_a.pth')
	#print('Train Loss :',loss_total.item()/len(train_loader))
	#print('Validation Loss :', loss_test_total.item()/len(test_loader))
	print('accuracy train',accuracy_train.item()/len(train_loader))
	print('accuracy test', accuracy_test.item()/len(test_loader))







