import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF


labels = {
    0: "Tshirt/Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


# # TODO: declare your model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer2 = nn.Sequential(
            torch.nn.Linear(64 * 14 * 14, 256, bias=True),
            nn.ReLU(),
            torch.nn.Linear(256, 10, bias=True),
        )

    def forward(self, x):

        out = self.layer1(x)
        out = out.view(-1, 64 * 14 * 14)  # Flatten
        out = self.layer2(out)

        return out


# TODO
# load model variables in evaluation mode
model = Model()
chkpt_path = "."
model.load_state_dict(torch.load(chkpt_path + "model_1_c.pth"))
model.eval()

probability = torch.nn.Softmax(dim=1)  # default is dim = 0, I think it should be dim=1

for i in range(10):

    image = Image.open("img0" + str(i) + ".png")
    x = TF.to_tensor(image)
    input_matrix = x.unsqueeze_(0)
    label_network_prediction = model(input_matrix)

    # TODO:
    # 1. Find probability
    # 2. Find labels
    _, predicted = torch.max(probability(label_network_prediction), axis=1)
    prediction = predicted.item()
    print("network prediction for img", i, " : ", {prediction, labels[prediction]})
