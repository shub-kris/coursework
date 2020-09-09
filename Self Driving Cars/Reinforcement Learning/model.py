import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_HEIGHT = 84
# IMG_HEIGHT=96
IMG_WIDTH = 96


class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device
        self.action_size = action_size

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=10, kernel_size=5, padding=2
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, padding=1
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding=1
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=30, out_channels=60, kernel_size=3, padding=1
        )

        self.fc1 = torch.nn.Linear(
            in_features=60 * IMG_WIDTH * IMG_HEIGHT + 1 + 4 + 1 + 1, out_features=400
        )
        # self.fc1 = torch.nn.Linear(in_features=60*IMG_WIDTH*IMG_HEIGHT,
        # out_features=100)
        self.fc2 = torch.nn.Linear(in_features=400, out_features=300)
        self.fc23 = torch.nn.Linear(in_features=300, out_features=200)
        self.fc3 = torch.nn.Linear(in_features=200, out_features=4)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """
        observation = torch.tensor(observation, device=self.device)
        batch_size = observation.shape[0]

        # get sensor data
        speed, abs, steer, gyro = self.extract_sensor_values(
            observation, observation.shape[0]
        )

        # crop image and permute dimensions in correct order
        observation = observation[:, :IMG_HEIGHT, :, :].permute(0, 3, 1, 2)
        # grayscale
        observation = (
            observation[:, 0, :, :] * 0.2989
            + observation[:, 1, :, :] * 0.5870
            + observation[:, 2, :, :] * 0.1140
        )
        observation = observation.unsqueeze(1)

        x = self.conv1(observation)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)

        x = self.conv2(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)

        x = self.conv3(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)

        x = self.conv4(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)

        x = self.fc1(
            torch.cat([x.reshape(batch_size, -1), speed, abs, steer, gyro], dim=1)
        )
        # x = self.fc1(x.reshape(batch_size, -1))
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)

        x = self.fc2(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)

        x = self.fc23(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)

        out = x = self.fc3(x)

        return out

    def extract_sensor_values(self, observation, batch_size):
        """ Extract numeric sensor values from state pixels
        Parameters
        ----------
        observation: list
            python list of batch_size many torch.Tensors of size (96, 96, 3)
        batch_size: int
            size of the batch
        Returns
        ----------
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 4),
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 1)
            Extracted numerical values
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
