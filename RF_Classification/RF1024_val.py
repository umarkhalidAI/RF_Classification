import os,torch,csv
import os,torch,csv
import torch.nn as nn
import numpy
import dataloader_rf1024 as dataloader
from rfml.nn.model.resnet_CT import resnet18, resnetx
import time
device = torch.device("cuda:0" if torch.cuda.is_available()
                                  else "cpu")
import argparse
parser = argparse.ArgumentParser(description='RF1024 Validation')
parser.add_argument('--model', default='resnet18', type=str,
                    help='resnet18 or conv5 or CNN or CLDNN')
parser.set_defaults(augment=True)
global args
args=parser.parse_args()

print("Loading Data")
val_dataset=dataloader.val_data()

print("Loading Model")
if args.model == 'resnet18':
    model = resnet18()
elif args.model == 'conv5':
    model = resnetx()
model.to(device)
val_loss = []
val_acc = []
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
total_t = 0
correct_t = 0
confusion_matrix = torch.zeros(8,8)
start=time.time()
for data_t, target_t in (val_dataset):
    data_t, target_t = data_t.to(device), target_t.to(device)# on GPU
    target_t=target_t.long()
    outputs_t = model(data_t.float())
    _, pred_t = torch.max(outputs_t,1)#, dim=1)
    correct_t += torch.sum(pred_t == target_t).item()
    for t, p in zip(target_t.view(-1), pred_t.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    total_t += target_t.size(0)
end=time.time()
print(end-start)
numpy.set_printoptions(suppress=True)
print(100 * correct_t / total_t)
print(confusion_matrix.cpu().detach().numpy())
print(confusion_matrix.diag()/confusion_matrix.sum(1))