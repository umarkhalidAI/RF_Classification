import os,torch,csv
import numpy as np
from rfml.nn.model.resnet_CT import resnet18, resnetx
import dataloader_rf1024 as dataloader
import torch.optim as optim
import torch.nn as nn
import argparse
parser = argparse.ArgumentParser(description='RF1024 Training')
parser.add_argument('--model', default='resnet18', type=str,
                    help='resnet18 or conv5 or CNN or CLDNN')
parser.add_argument('--lr', default=0.001, type=int,
                    help='Learning rate for training')
parser.add_argument('--epoch', default=100, type=int,
                    help='Number of training epochs')
parser.add_argument('--model_save', default=True, type=bool,
                    help='resnet18 or conv5')
parser.set_defaults(augment=True)

global args
args=parser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available()
                                  else "cpu")
def write_csv(file, newrow):
    with open(file, mode='a') as f:
        f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(newrow)
valid_loss_min = np.Inf

if __name__ == '__main__':
    train_dataset = dataloader.train_data()
    val_dataset=dataloader.val_data()
    if args.model == 'resnet18':
        model = resnet18()
    elif args.model == 'conv5':
        model = resnetx()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    model.to(device)
    total_step = len(train_dataset)
    print('total_step: ', total_step)
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_dataset, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            labels=labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == labels).item()
            total += labels.size(0)
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / total_step)
        print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
        write_csv(f'log/train.csv', [epoch, (100 * correct / total), np.mean(train_loss)])
        batch_loss = 0
        total_t = 0
        correct_t = 0
        with torch.no_grad():
            model.eval()
            for data_t, target_t in (val_dataset):
                data_t, target_t = data_t.to(device), target_t.to(device)# on GPU
                target_t=target_t.long()
                outputs_t = model(data_t.float())
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss / len(val_dataset))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
            write_csv(f'log/valid.csv', [epoch, (100 * correct_t / total_t), np.mean(val_loss)])
            # Saving the best weight
            if network_learned:
                valid_loss_min = batch_loss
                torch.save(model.state_dict(), 'best_model.pt')
                print('Detected network improvement, saving current model')
        model.train()




