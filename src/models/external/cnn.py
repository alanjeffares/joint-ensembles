import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(-1, 500)

class CNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        learner_network = nn.ModuleList()
        learner_network.append(nn.Conv2d(3, 10, kernel_size=5))
        learner_network.append(nn.Dropout2d(p=0.1))
        learner_network.append(nn.MaxPool2d((2,2)))
        learner_network.append(nn.ReLU())
        learner_network.append(nn.Conv2d(10, 20, kernel_size=5))
        learner_network.append(nn.Dropout2d(p=0.1))
        learner_network.append(nn.MaxPool2d((2,2)))
        learner_network.append(nn.ReLU())
        learner_network.append(Flatten())
        learner_network.append(nn.Linear(500, 50))
        learner_network.append(nn.Dropout(p=0.1))
        learner_network.append(nn.ReLU())
        learner_network.append(nn.Linear(50, num_classes))

        self.model = nn.Sequential(*learner_network)
    
    def forward(self, x):
        return self.model(x)

def cnn(num_classes: int):
    return CNN(num_classes=num_classes)