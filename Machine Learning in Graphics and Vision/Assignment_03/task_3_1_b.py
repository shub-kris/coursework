import torch
import torch.nn as nn
from torchsummary import summary

# TODO : copy the model structure
class Model(nn.Module):
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Model().to(device)


# TODO : print model summary
print(summary(model, (1,28,28)))
