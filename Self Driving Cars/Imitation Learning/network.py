import torch


class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        gpu = torch.device("cuda")

        self.classes = [
            [-1, 0.0, 0.0],  # Steer Left
            [-1.0, 0.5, 0.0],
            [-1.0, 0.0, 0.8],  # Steer left with brakes
            [1, 0.0, 0.0],  # Steer right
            [1.0, 0.5, 0.0],
            [1.0, 0.0, 0.8],  # Steer right with brakes
            [0.0, 0.0, 0.0],  # Nothing
            [0.0, 0.5, 0.0],  # Accelerate
            [0.0, 0.0, 0.8],
        ]  # Brakes

        self.features_2d = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 5, stride=2),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm2d(4),
            torch.nn.Conv2d(4, 8, 5, stride=2),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 16, 5, stride=2),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 3, stride=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm2d(32),
        ).to(gpu)

        self.features_1d = torch.nn.Sequential(
            torch.nn.Linear(32 * 7 * 7, 128),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm1d(64)
            # torch.nn.Linear(64,32),
            # torch.nn.LeakyReLU(negative_slope = 0.2),
            # torch.nn.BatchNorm1d(32)
            # torch.nn.Linear(32,16),
            # torch.nn.LeakyReLU(negative_slope = 0.2),
            # torch.nn.Linear(16,7),
            # torch.nn.Softmax(dim = 1)
        ).to(gpu)

        self.scores = torch.nn.Sequential(
            torch.nn.Linear(71, 16),
            # torch.nn.BatchNorm1d(32),
            # torch.nn.LeakyReLU(negative_slope = 0.2),
            # torch.nn.Linear(32,16),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm1d(16),
            torch.nn.Linear(16, 9),
            torch.nn.Softmax(dim=1),
        ).to(gpu)

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """

        batch_size = observation.shape[0]

        # Conversion to gray_scale
        # observation = observation[:,:,:,0] * 0.2989 + observation[:,:,:,1] * 0.5870 + observation[:,:,:,2] * 0.1140
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(
            observation, batch_size
        )
        obs = observation.reshape(batch_size, 3, 96, 96)
        features_2d = self.features_2d(obs).reshape(batch_size, -1)
        features_1d = self.features_1d(features_2d)
        fused_features = torch.cat(
            (features_1d, speed, abs_sensors, steering, gyroscope), 1
        )
        return self.scores(fused_features)

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are number_of_classes
        different classes, then every action is represented by a
        number_of_classes-dim vector which has exactly one non-zero entry
        (one-hot encoding). That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes
        """

        # Figure out which representation means what? ex: [Steer, Gas, Brake] --> [1.,0.,0.8] --> Does what? Does it steer right while applying brake?

        # Outerloop is actions
        # Inner loop is this_class
        # action == class returns a list of boolean values
        return [
            torch.Tensor(
                [
                    int(
                        torch.prod(action == this_class)
                    )  # calculate the product of the elements in boolean tensor and converts them to a list
                    for this_class in torch.Tensor(self.classes)
                ]
            )
            for action in actions
        ]

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """
        x, class_number = torch.max(
            scores[0], dim=0
        )  # Why scores[0] only?? # Finds the max from the 9X1 column vector
        steer, gas, brake = self.classes[
            class_number
        ]  # Uses the classes you defined in init function to get back the appropriate values for steer,gas,brake
        return steer, gas, brake

    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        print(f"Shape is:{speed_crop.shape}")
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
