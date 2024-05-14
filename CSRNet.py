from torch import nn
from torchvision import models


class CSRNet(nn.Module):
    """CSRNet模型"""

    def __init__(self, init_weights=False, load_vgg=True):
        super(CSRNet, self).__init__()
        channel = 3
        # 前端为vgg16的前10层
        self.frontend_layers = []
        for i in ((2, 64), (2, 128), (3, 256), (3, 512)):
            for j in range(i[0]):
                self.frontend_layers.append(nn.Conv2d(channel, i[1], kernel_size=(3, 3), padding=1))
                self.frontend_layers.append(nn.ReLU(True))
                channel = i[1]
            self.frontend_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.frontend_layers.pop()  # 去掉结尾的MaxPool层
        self.frontend = nn.Sequential(*self.frontend_layers)
        # 后端为空洞卷积层
        self.backend_layers = []
        channels = [512, 512, 512, 256, 128, 64]
        for i in channels:
            self.backend_layers.append(nn.Conv2d(channel, i, kernel_size=(3, 3), dilation=(2, 2), padding=2))
            self.backend_layers.append(nn.ReLU(True))
            channel = i
        self.backend = nn.Sequential(*self.backend_layers)
        # 输出层为1-1卷积层
        self.output = nn.Conv2d(64, 1, kernel_size=(1, 1))
        # 初始化权重
        if init_weights:
            self._initialize_weights()
            if load_vgg:
                # 将预训练模型的权重转到self.frontend上
                mod = models.vgg16(pretrained=True)
                sd = mod.features.state_dict()
                ks = []
                for k in sd:
                    if k not in self.frontend.state_dict():
                        ks.append(k)
                for k in ks:
                    sd.pop(k)
                self.frontend.load_state_dict(sd)

    def forward(self, x):
        x = self.frontend.forward(x)
        x = self.backend.forward(x)
        x = self.output.forward(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
