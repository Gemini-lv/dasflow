import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class Mini(nn.Module):
    def __init__(self):
        super(Mini, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 48, 3, 1, 1)
        self.conv3 = nn.Conv2d(48, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 32, 3, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.conv11 = nn.Conv2d(16, 16, 1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 2)
        self.pool1 = nn.MaxPool2d(2, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool1(torch.relu(self.conv3(x)))
        x = self.pool2(torch.relu(self.conv4(x)))
        #x = nn.functional.interpolate(x, size=(1, 8), mode='bilinear', align_corners=True)
        x = self.global_pool(x)
        x = x.view(-1, 32)
        #x = self.conv11(x).view(-1, 16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        #x = nn.functional.relu(x)
        return x
if __name__ == '__main__':
    model = Mini() # 使用最基础的网络结构进行检测
    model.load_state_dict(torch.load('model_all.pth')) # 使用预训练好的模型权重进行检测