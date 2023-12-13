import torch
import torch.optim as optim
import datetime
import nibabel as nib
import nilearn as nl
import os


from utils import *
from models import *
# from do_swin_transformer import do_swin_transformer
from torch.utils.data import WeightedRandomSampler

# Jang, Jinseong, and Dosik Hwang.
# "M3T: three-dimensional Medical image classifier using Multi-plane and Multi-slice Transformer."
# Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

# set random seed
ManualSeed(2222)

# hyper-parameters
num_batch = 2
num_epoch = 25
learning_rate = 1e-4

# data => this code block need to be edited with actual data
# data = torch.rand((100,128,128,128))
# label = torch.zeros((100))
# label[:50] = 1
# train_set = CustomDataSet(data[:80],label[:80])
# test_set = CustomDataSet(data[80:],label[80:])
# train_loader = DataLoader(train_set,batch_size=num_batch,shuffle=False)
# test_loader = DataLoader(test_set,batch_size=num_batch,shuffle=False)

path = 'C:/Users/kimys/PycharmProjects/pythonProject/ADNI/'
ads1 = os.listdir(path+'AD_resize/')
data = []
for i, a in enumerate(ads1):
    data.append(nib.load(path+'AD_resize/'+a).get_fdata())
ads = os.listdir(path+'CN_resize/')
for i, a in enumerate(ads):
    data.append(nib.load(path+'CN_resize/'+a).get_fdata())

data = torch.tensor(np.stack(data,0)).float()
label = torch.zeros((len(data))).long()
# label[:len(ads1)] = 1
label[:len(ads1)] = 1
dataset = CustomDataSet(data,label)
test_len = int(len(dataset) * 0.2)
train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset) - test_len, test_len])
l = train_set[:][1]
class_sample_count = np.array([sum(l), len(l)-sum(l)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in l])
samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
train_loader = DataLoader(train_set,batch_size=num_batch,shuffle=False, sampler=sampler)
test_loader = DataLoader(test_set,batch_size=num_batch,shuffle=True)

# train-test
# model = do_swin_transformer().to(DEVICE)
model = M3T_model_wSw().to(DEVICE)
tr_acc, tr_loss = doTrain(model=model,
                          train_loader=train_loader,
                          num_epoch=num_epoch,
                          optimizer=optim.Adam(model.parameters(),lr=learning_rate,betas=(0.9,0.999)))
acc, predictions, targets = doTest(model,test_loader)

# predictions, targets = 0,0
# save result
cur_time = datetime.datetime.now().strftime('%m%d_%H%M')
SaveResults_mat(f'result_{cur_time}',acc,predictions,targets,tr_acc,tr_loss,num_batch,num_epoch,learning_rate)
torch.save(model,f'./results/models/model_{cur_time}.pth')