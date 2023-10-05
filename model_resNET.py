import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_actions):
        super(ResNet50, self).__init__()

        # Load the pre-trained ResNet50 model
        self.resnet50 = models.resnet50(weights=None)

        # Replace the last fully connected layer to output two numerical values
        num_ftrs = self.resnet50.fc.out_features

        self.fc1x_float = nn.Linear(num_ftrs, 32)
        self.relu3x_float = nn.ReLU()
        self.fc2x_mean = nn.Linear(32, 1)
        self.fc2x_std = nn.Linear(32, 1)

        self.fc1y_float = nn.Linear(num_ftrs, 32)
        self.relu3y_float = nn.ReLU()
        self.fc2y_mean = nn.Linear(32, 1)
        self.fc2y_std = nn.Linear(32, 1)

        self.fc1 = nn.Linear(num_ftrs, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_actions)

        self.fcv = nn.Linear(num_ftrs, 64)
        self.relu4 = nn.ReLU()
        self.fc_v_ap = nn.Linear(64, 32)
        self.fc_v_a = nn.Linear(32, 1)
        self.fc_v_cx = nn.Linear(64, 1)
        self.fc_v_cy = nn.Linear(64, 1)


    def a(self, x):
        x = self.resnet50(x)
        x_floatx = self.fc1x_float(x)
        x_floatx = self.relu3x_float(x_floatx)
        mean_x = torch.tanh(self.fc2x_mean(x_floatx))
        std_x = torch.sigmoid(self.fc2x_std(x_floatx)) + 1e-10  # Ensure std is positive

        x_floaty = self.fc1y_float(x)
        x_floaty = self.relu3y_float(x_floaty)
        mean_y = torch.tanh(self.fc2y_mean(x_floaty))
        std_y = torch.sigmoid(self.fc2y_std(x_floaty)) + 1e-10  # Ensure std is positive

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        logits = self.fc3(x)
        action_probs = torch.softmax(logits, dim=-1)

        return action_probs, mean_x, std_x, mean_y, std_y

    def v(self, x):
        x = self.resnet50(x)
        x = self.fcv(x)
        x = self.relu4(x)
        v_cx = self.fc_v_cx(x)
        v_cy = self.fc_v_cy(x)
        v_a = self.fc_v_ap(x)
        v_a = self.fc_v_a(v_a)
        return v_cx, v_cy, v_a 