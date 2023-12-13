import torch
import torch.nn as nn
from swin_transformer import SwinTransformer
from Resnet18 import BasicBlock, ResNet18
import torch.nn.functional as F

class do_swin_transformer(nn.Module):   # Resnet18의 출력값을 swin transformer로 입력한 후 MLP, ReLU 통과 시키는 클래스
    def __init__(self):
        super(do_swin_transformer, self).__init__()

        self.resnet = ResNet18(BasicBlock, [2,2,2,2])

        self.conv = nn.Conv2d(512*3, 512, kernel_size=(1,1))

        self.swin_transformer = SwinTransformer(
            img_size=32,
            patch_size=4,
            in_chans=512,
            num_classes=1,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=4,
            mlp_ratio=4.,
        )

        # self.mlp = nn.Linear(1000, 1)

    def forward(self, x1):
        # resnet 3번
        tensor1 = self.resnet(x1)
        x2 = x1.permute(0,2,1,3)
        tensor2 = self.resnet(x2)
        x3 = x1.permute(0,3,1,2)
        tensor3 = self.resnet(x3)
        # resnet output tensor 3개 concat
        result = torch.cat((tensor1, tensor2, tensor3), dim=1)
        # print(result.shape)
        input_tensor = result
        # 3*512 채널 -> 512 채널로 conv
        output_tensor = self.conv(input_tensor)
        # swin transformer
        output = self.swin_transformer(output_tensor)
        # mlp 후 relu
        output = F.sigmoid(output)
        # output = F.relu(self.mlp(output))

        output = output.squeeze(-1)
        # print(output.shape)
        return output   # B

#model = do_swin_transformer()
#x = torch.randn(100, 128, 128, 128)
#model(x).shape
