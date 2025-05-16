import torch
import torch.nn as nn
import torchvision.models as models
# from resnet import resnet18,resnet50
'''
class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
'''
class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim,hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # if self.fc1 is None:
            # self._init_fc_layers(x)

        # x = x.view(x.size(0), -1)
        x = x.view(-1, 16*13*13)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

    # def _init_fc_layers(self, x):
    #     flat_dim = x.size(1) * x.size(2) * x.size(3)

    #     self.fc1 = nn.Linear(flat_dim, self.hidden_dims[0])
    #     self.fc2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])

    #     self.fc1.to(x.device)
    #     self.fc2.to(x.device)


class SimpleCNNMNIST_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST_header, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class ModelFedCon_noheader(nn.Module):
    def __init__(self, base_model, out_dim, n_classes, dataset=None):
        super(ModelFedCon_noheader, self).__init__()
        if base_model == 'resnet18_gn':
            basemodel = models.resnet18(pretrained=False)
            basemodel.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            basemodel.maxpool = torch.nn.Identity()
            basemodel.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            basemodel.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            basemodel.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16*13*13),hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'simple-cnn-mnist':
            self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == "resnet18":
            basemodel = models.resnet18()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-pretrained":
            basemodel = models.resnet18(pretrained=True)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
            for param in self.features.parameters():
                param.requires_grad = False
        elif base_model == "vit-b-16":
            basemodel = models.vit_b_16(pretrained=True)
            num_ftrs = basemodel.heads.head.in_features
            
            basemodel.heads.head = nn.Identity()
            self.features = basemodel
            # num_ftrs = basemodel.heads.head.in_features
            for param in self.features.parameters():
                param.requires_grad = False
        elif  base_model == "resnet50":
            basemodel = models.resnet50()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == 'resnet50-pretrained':
            basemodel = models.resnet50(pretrained=True)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
            for param in self.features.parameters():
                param.requires_grad = False
        elif base_model == "resnet34":
            basemodel = models.resnet34()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet34-pretrained":
            basemodel = models.resnet34(pretrained=True)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
            for param in self.features.parameters():
                param.requires_grad = False
        elif base_model == 'resnet101-pretrained':
            basemodel = models.resnet101(pretrained=True)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
            for param in self.features.parameters():
                param.requires_grad = False
        elif base_model == "vgg16":
            basemodel = models.vgg16()
            self.features = nn.Sequential(*list(basemodel.features.children()),
                                          nn.AdaptiveAvgPool2d(output_size=(7, 7)))
            num_ftrs = 512 * 7 * 7
        elif base_model == "densenet121":
            basemodel = models.densenet121()
            self.features = nn.Sequential(*list(basemodel.features.children()),
                                          nn.ReLU(inplace=True),
                                          nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            num_ftrs = basemodel.classifier.in_features
        # Last layer
        self.l3 = nn.Linear(num_ftrs, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        y = self.l3(h)
        return h, h, y

