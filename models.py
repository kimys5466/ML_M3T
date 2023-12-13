import torch
import torch.nn as nn
import torch.functional as F

from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, swin_t
from einops import repeat
from utils import DEVICE
from Resnet18 import BasicBlock, ResNet18

class CNN_3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(1, 8, 5, padding=5 // 2),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 32, 5, padding=5 // 2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class M3T_model_origin(nn.Module):
    def __init__(self):
        super().__init__()

        # 3D CNN
        self.cnn3d = CNN_3D()

        # 2D CNN
        self.cnn2d = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn2d.conv1 = nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.cnn2d.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # Position and Plane Embedding
        self.cls_token = nn.Parameter(torch.rand(1, 1, 256))
        self.sep_token = nn.Parameter(torch.rand(1, 1, 256))

        # Transformer Encoder
        encoder = nn.TransformerEncoderLayer(256, 8, 768, activation='gelu', batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(encoder, 8)
        self.register_buffer("pos_idx", torch.arange(388))
        self.register_buffer("pln_idx", torch.tensor([0] + [x // 129 for x in range(387)]))
        self.pos_emb = nn.Embedding(388, 256)
        self.pln_emb = nn.Embedding(3, 256)

        # Classification
        self.fc = nn.Sequential(
            nn.Linear(388 * 256, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        # initial_input_shape = B*128*128*128

        # 1. 3D CNN
        x = x.unsqueeze(1)  # B*1*128*128*128
        x = self.cnn3d(x)  # B*32*128*128*128

        # 2. Multi-Slices & 2D CNN
        c = self.cnn2d(x[:, :, 0, :, :]).unsqueeze(1)
        s = self.cnn2d(x[:, :, :, 0, :]).unsqueeze(1)
        a = self.cnn2d(x[:, :, :, :, 0]).unsqueeze(1)
        for i in range(1, 128):
            c = torch.concat([c, self.cnn2d(x[:, :, i, :, :]).unsqueeze(1)], dim=1)  # 2dcnn out: [B*256]
            s = torch.concat([s, self.cnn2d(x[:, :, :, i, :]).unsqueeze(1)], dim=1)  #
            a = torch.concat([a, self.cnn2d(x[:, :, :, :, i]).unsqueeze(1)], dim=1)  # c,saa: [B*128*256]

        # 3. Position and Plane Embedding
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=x.shape[0])
        sep_tokens = repeat(self.sep_token, '() n e -> b n e', b=x.shape[0])
        out = torch.concat([cls_tokens, c, sep_tokens, s, sep_tokens, a, sep_tokens], dim=1)  # B*388*256
        pos_emb = self.pos_emb(self.pos_idx)  # 388*256
        pln_emb = self.pln_emb(self.pln_idx)  # 388*256
        out += pos_emb + pln_emb  # B*388*256

        # 4. Transformer Encoder
        out = self.transformer_enc(out)  # B*388*256

        # 5. Classification
        out = out.flatten(1)  # B*99328
        out = self.fc(out)  # B*1
        return torch.squeeze(out)


class M3T_model(nn.Module):
    def __init__(self):
        super().__init__()

        vec_size = 256
        # 3D CNN
        self.cnn3d = CNN_3D()

        # 2D CNN
        # self.cnn2d = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.cnn2d = ResNet18(BasicBlock, [2,2,2,2])
        self.cnn2d = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn2d.conv1 = nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.cnn2d.fc = nn.Sequential(
            # nn.Linear(2048,512),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Linear(512,vec_size)
            nn.Linear(256, vec_size),
        )

        # Position and Plane Embedding
        self.cls_token = nn.Parameter(torch.rand(1, 1, vec_size))
        self.sep_token = nn.Parameter(torch.rand(1, 1, vec_size))

        # Transformer Encoder
        encoder = nn.TransformerEncoderLayer(vec_size, 8, 384, activation='gelu', batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(encoder, 8)
        self.register_buffer("pos_idx", torch.arange(388))
        self.register_buffer("pln_idx", torch.tensor([0] + [x // 129 for x in range(387)]))
        self.pos_emb = nn.Embedding(388, vec_size)
        self.pln_emb = nn.Embedding(3, vec_size)

        # Classification
        self.fc = nn.Sequential(
            nn.Linear(388 * vec_size, 2),
            # nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        # initial_input_shape = B*128*128*128
        B = x.shape[0]

        # 1. 3D CNN
        x = x.unsqueeze(1)  # B*1*128*128*128
        x = self.cnn3d(x)  # B*32*128*128*128
        c = torch.reshape(x.permute(0, 2, 1, 3, 4), [B * 128, 32, 128, 128])
        s = torch.reshape(x.permute(0, 3, 1, 2, 4), [B * 128, 32, 128, 128])
        a = torch.reshape(x.permute(0, 4, 1, 2, 3), [B * 128, 32, 128, 128])
        # x = torch.concat([x,x.permute(0,1,3,2,4),x.permute(0,1,4,3,2)],dim=2) # [B,32,384,128,128]

        # 2. Multi-Slices & 2D CNN
        c = self.cnn2d(c).view(B, 128, 128)
        s = self.cnn2d(s).view(B, 128, 128)
        a = self.cnn2d(a).view(B, 128, 128)  # [B, 128,128]\
        # y = self.cnn2d(x[:,:,0,:,:]).unsqueeze(1)

        # for i in range(1,4):
        #     y = torch.concat([y,self.cnn2d(x[:,:,i,:,:]).unsqueeze(1)],dim=1) # [B,384,128]
        # c = self.cnn2d(x[:,:,0,:,:]).unsqueeze(1)
        # for i in range(1,128):
        #     c = torch.concat([c,self.cnn2d(x[:,:,i,:,:]).unsqueeze(1)],dim=1)   # 2dcnn out: [B*256]
        # s = self.cnn2d(x[:,:,:,0,:]).unsqueeze(1)
        # for i in range(1,128):
        #     s = torch.concat([s,self.cnn2d(x[:,:,:,i,:]).unsqueeze(1)],dim=1)   #
        # a = self.cnn2d(x[:,:,:,:,0]).unsqueeze(1)
        # for i in range(1,128):
        #     a = torch.concat([a,self.cnn2d(x[:,:,:,:,i]).unsqueeze(1)],dim=1)   # c,saa: [B*128*256]
        # del x
        # 3. Position and Plane Embedding
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=B)
        sep_tokens = repeat(self.sep_token, '() n e -> b n e', b=B)
        # out = torch.concat([cls_tokens,y[:128],sep_tokens,y[128:256],sep_tokens,y[256:],sep_tokens],dim=1)   # B*388*256
        out = torch.concat([cls_tokens, c, sep_tokens, s, sep_tokens, a, sep_tokens], dim=1)  # B*388*256
        del c, s, a
        pos_emb = self.pos_emb(self.pos_idx)  # 388*256
        pln_emb = self.pln_emb(self.pln_idx)  # 388*256
        out += pos_emb + pln_emb  # B*388*256

        # 4. Transformer Encoder
        out = self.transformer_enc(out)  # B*388*256

        # 5. Classification
        out = out.flatten(1)  # B*99328
        out = self.fc(out)  # B*1
        return torch.squeeze(out)


class M3T_model2(nn.Module):
    def __init__(self):
        super().__init__()

        # 2D CNN
        # self.cnn2d = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.cnn2d = resnet18(weights=ResNet18_Weights.DEFAULT)
        # self.cnn2d.conv1 = nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.cnn2d.fc = nn.Sequential(
        #     nn.Linear(512,256),
        #     nn.ReLU(),
        #     nn.Linear(256,vec_size)
        # )
        self.cnn2d = ResNet18(BasicBlock, [2, 2, 2, 2])
        self.conv = nn.Sequential(
            nn.Conv2d(128 * 3, 3, kernel_size=(1, 1)),
            nn.Dropout2d(0.25),
        )
        self.swin = swin_t(**{'num_classes': 2, })
        # self.swin = swin_v2_t(**{'num_classes':2, })

    def forward(self, x: torch.Tensor):
        # initial_input_shape = B*128*128*128
        # 2. Multi-Slices & 2D CNN
        # print(x.shape)
        t1 = self.cnn2d(x)
        t2 = self.cnn2d(x.permute(0, 2, 1, 3))
        t3 = self.cnn2d(x.permute(0, 3, 1, 2))
        # print(t1.shape)
        y = torch.cat((t1, t2, t3), dim=1)
        # print(y.shape)
        y = self.conv(y)

        # 4. Transformer Encoder
        y = self.swin(y)  # B*388*256
        return torch.squeeze(y)


class M3T_model3(nn.Module):
    def __init__(self):
        super().__init__()

        vec_size = 128

        # 2D CNN
        self.cnn2d = nn.Sequential(
            nn.Conv2d(128, 64, 5, padding=5 // 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=5 // 2),
            nn.ReLU(),
        )

        # Position and Plane Embedding
        self.cls_token = nn.Parameter(torch.rand(1, 1, vec_size))
        self.sep_token = nn.Parameter(torch.rand(1, 1, vec_size))

        # Transformer Encoder
        encoder = nn.TransformerEncoderLayer(vec_size, 8, 384, activation='gelu', batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(encoder, 8)
        self.register_buffer("pos_idx", torch.arange(388))
        self.register_buffer("pln_idx", torch.tensor([0] + [x // 129 for x in range(387)]))
        self.pos_emb = nn.Embedding(388, vec_size)
        self.pln_emb = nn.Embedding(3, vec_size)

        # Classification
        self.fc = nn.Sequential(
            nn.Linear(388 * vec_size, 2),
            # nn.Sigmoid()
        )
        # self.swin = swin_v2_t(**{'num_classes':2, })

    def forward(self, x: torch.Tensor):
        # initial_input_shape = B*128*128*128
        B = x.shape[0]

        t1 = self.cnn2d(x)
        t2 = self.cnn2d(x.permute(0, 2, 1, 3))
        t3 = self.cnn2d(x.permute(0, 3, 1, 2))  # [B, 1, 128, 128]
        y = torch.cat((t1, t2, t3), dim=2).squeeze(1)  # [B, 384, 128]

        # 3. Position and Plane Embedding
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=B)
        sep_tokens = repeat(self.sep_token, '() n e -> b n e', b=B)
        out = torch.concat([cls_tokens, y[:, :128], sep_tokens, y[:, 128:256], sep_tokens, y[:, 256:], sep_tokens],
                           dim=1)  # B*388*256
        # out = torch.concat([cls_tokens,c,sep_tokens,s,sep_tokens,a,sep_tokens],dim=1)   # B*388*256
        # del c,s,a
        pos_emb = self.pos_emb(self.pos_idx)  # 388*256
        pln_emb = self.pln_emb(self.pln_idx)  # 388*256
        out += pos_emb + pln_emb  # B*388*256

        # 4. Transformer Encoder
        out = self.transformer_enc(out)  # B*388*256

        # 5. Classification
        out = out.flatten(1)  # B*99328
        out = self.fc(out)  # B*1
        return torch.squeeze(out)


class M3T_model_wResNet(nn.Module):
    def __init__(self):
        super().__init__()

        vec_size = 256
        # 2D CNN
        self.cnn2d = ResNet18(BasicBlock, [2, 2, 2, 2])

        # Position and Plane Embedding
        self.cls_token = nn.Parameter(torch.rand(1, 1, vec_size))
        self.sep_token = nn.Parameter(torch.rand(1, 1, vec_size))

        # Transformer Encoder
        encoder = nn.TransformerEncoderLayer(vec_size, 8, 384, activation='gelu', batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(encoder, 8)
        self.register_buffer("pos_idx", torch.arange(388))
        self.register_buffer("pln_idx", torch.tensor([0] + [x // 129 for x in range(387)]))
        self.pos_emb = nn.Embedding(388, vec_size)
        self.pln_emb = nn.Embedding(3, vec_size)

        # Classification
        self.fc = nn.Sequential(
            nn.Linear(388 * vec_size, 2),
            # nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        # initial_input_shape = B*128*128*128
        B = x.shape[0]

        # 1. 3D CNN
        c = self.cnn2d(x)
        s = self.cnn2d(x.permute(0, 2, 1, 3))
        a = self.cnn2d(x.permute(0, 3, 1, 2))  # [B, 128, 128]

        # 2. Multi-Slices & 2D CNN
        # c = self.cnn2d(c).view(B,128,128)
        # s = self.cnn2d(s).view(B,128,128)
        # a = self.cnn2d(a).view(B,128,128) # [B, 128,128]
        # c = self.cnn2d(c).view(B,128,128)
        # s = self.cnn2d(s).view(B,128,128)
        # a = self.cnn2d(a).view(B,128,128) # [B, 128,128]
        # y = self.cnn2d(x[:,:,0,:,:]).unsqueeze(1)

        # for i in range(1,4):
        #     y = torch.concat([y,self.cnn2d(x[:,:,i,:,:]).unsqueeze(1)],dim=1) # [B,384,128]
        # c = self.cnn2d(x[:,:,0,:,:]).unsqueeze(1)
        # for i in range(1,128):
        #     c = torch.concat([c,self.cnn2d(x[:,:,i,:,:]).unsqueeze(1)],dim=1)   # 2dcnn out: [B*256]
        # s = self.cnn2d(x[:,:,:,0,:]).unsqueeze(1)
        # for i in range(1,128):
        #     s = torch.concat([s,self.cnn2d(x[:,:,:,i,:]).unsqueeze(1)],dim=1)   #
        # a = self.cnn2d(x[:,:,:,:,0]).unsqueeze(1)
        # for i in range(1,128):
        #     a = torch.concat([a,self.cnn2d(x[:,:,:,:,i]).unsqueeze(1)],dim=1)   # c,saa: [B*128*256]
        # del x
        # 3. Position and Plane Embedding
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=B)
        sep_tokens = repeat(self.sep_token, '() n e -> b n e', b=B)
        # out = torch.concat([cls_tokens,y[:128],sep_tokens,y[128:256],sep_tokens,y[256:],sep_tokens],dim=1)   # B*388*256
        out = torch.concat([cls_tokens, c, sep_tokens, s, sep_tokens, a, sep_tokens], dim=1)  # B*388*256
        del c, s, a
        pos_emb = self.pos_emb(self.pos_idx)  # 388*256
        pln_emb = self.pln_emb(self.pln_idx)  # 388*256
        out += pos_emb + pln_emb  # B*388*256

        # 4. Transformer Encoder
        out = self.transformer_enc(out)  # B*388*256

        # 5. Classification
        out = out.flatten(1)  # B*99328
        out = self.fc(out)  # B*1
        return torch.squeeze(out)


class M3T_model_wSw(nn.Module):
    def __init__(self):
        super().__init__()

        # 2D CNN
        self.cnn3d = CNN_3D()

        # 2D CNN
        self.cnn2d = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.cnn2d = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn2d.conv1 = nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.cnn2d.fc = nn.Sequential(
            nn.Linear(2048, 512),
            # nn.Linear(512,256),
            nn.ReLU(),
            # nn.Linear(512,vec_size)
            nn.Linear(512, 256),
        )

        # self.cnn2d = ResNet18(BasicBlock, [2,2,2,2])
        self.conv = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=(1, 1)),
            nn.Conv2d(16, 1, kernel_size=(1, 1)),
            # nn.Dropout2d(0.25),
        )
        self.swin = swin_t(**{'num_classes': 2, })
        # self.swin = swin_v2_t(**{'num_classes':2, })

    def forward(self, x: torch.Tensor):
        # initial_input_shape = B*128*128*128
        B = x.shape[0]
        # 1. 3D CNN
        x = x.unsqueeze(1)  # B*1*128*128*128
        x = self.cnn3d(x)  # B*32*128*128*128
        c = torch.reshape(x.permute(0, 2, 1, 3, 4), [B * 128, 32, 128, 128])
        s = torch.reshape(x.permute(0, 3, 1, 2, 4), [B * 128, 32, 128, 128])
        a = torch.reshape(x.permute(0, 4, 1, 2, 3), [B * 128, 32, 128, 128])
        # print(x.shape)
        # 2. Multi-Slices & 2D CNN
        c = self.cnn2d(c).view(B, 128, 16, 16)
        s = self.cnn2d(s).view(B, 128, 16, 16)
        a = self.cnn2d(a).view(B, 128, 16, 16)  # [B, 128,128]
        c = self.conv(c)
        s = self.conv(s)
        a = self.conv(a)
        y = torch.cat((c, s, a), dim=1)
        # print(y.shape)
        # 4. Transformer Encoder
        y = self.swin(y)  # B*388*256
        # print(y.shape)
        return y