'''

eeg_training - Combines CNN and RNN with late fusion
Copyright (C) 2025 - Chinry Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch import optim
from torch.utils.data import Subset, DataLoader

from eeg_model import *
from Utils import *
import time
import warnings
import os
from datetime import datetime
import torch.nn.functional as F

os.makedirs('Output\\eeg_model', exist_ok=True)
os.makedirs('Output', exist_ok=True)

training_start_time=datetime.now().strftime("%m-%d_%H %M %S")

warnings.simplefilter("ignore")

t = time.time()
device = torch.device("cpu")

array_path="Input\\multi_array_eeg.npy"
img_path="Input\\multi_img.npy"
label_path="Input\\multi_label.npy"

from samples_intercept import *
sample_size=10000
X_images = np.load(img_path) # place here the images representation of EEG
X_array = np.load(array_path) # place here the array representation of EEG features
Label = np.load(label_path) # place here the label for each EEG
indices=intercept(Label,sample_size)
X_images=X_images[indices]
X_array=X_array[indices]
Label=Label[indices]

n_epoch = 150

#Label=label_corrupted(Label,corrupted_rate=0.1)

Dataset = CombDataset(label=Label, image=X_images, array=X_array)

indices = np.arange(len(Dataset))
np.random.shuffle(indices)
split = int(0.8 * len(indices))

Train = Subset(Dataset, indices[:split])
Test = Subset(Dataset, indices[split:])

Trainloader = DataLoader(Train, batch_size=128, shuffle=True, pin_memory=False)
Testloader = DataLoader(Test, batch_size=128, shuffle=False, pin_memory=False)

net = MultiModel().to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
#optimizer=optim.SGD(net.parameters(),lr=1e-3,momentum=0.9,weight_decay=1e-4)
lr="1e-3"
weight_decay="1e-4"
Res = []

Res.append({'samples':sample_size,'lr':lr,'weight_decay':weight_decay})

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

best_val_loss = float('inf')
best_val_acc = 0
patience = 8
counter = 0
patience_counter = 0
train_loss_plt=[]
val_loss_plt=[]
val_acc_plt=[]



for epoch in range(n_epoch):
    # 训练阶段
    net.train()
    running_loss = 0.0
    evaluation = []
    for i, data in enumerate(Trainloader):
        source_img, source_arr, label = data
        img = source_img.to(device)
        arr = source_arr.to(device)
        label = label.to(device)

        feat_img = net.FeatCNN(img.float())
        feat_arr = net.FeatRNN(arr.float().to(device))  # 确保设备一致
        feat = torch.cat((feat_img, feat_arr), axis=1)

        label_pred = net.ClassifierFC(feat)
        label_loss = F.cross_entropy(label_pred, label.long())

        optimizer.zero_grad()
        label_loss.backward()
        optimizer.step()

        running_loss += label_loss.item()
        _, predicted = torch.max(label_pred, 1)
        accuracy = (predicted == label).sum().item() / label.size(0)
        evaluation.append(accuracy)

    train_loss = running_loss / (i + 1)
    train_acc = sum(evaluation) / len(evaluation)

    net.eval()
    validation_loss = 0.0
    validation_acc = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for j, data in enumerate(Testloader):
            img, arr, label = data
            img, arr, label = img.to(device), arr.to(device), label.to(device)

            feat_img = net.FeatCNN(img.float())
            feat_arr = net.FeatRNN(arr.float())
            feat = torch.cat((feat_img, feat_arr), axis=1)
            pred = net.ClassifierFC(feat)

            loss = F.cross_entropy(pred, label.long())
            validation_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            validation_acc.extend((predicted == label).cpu().tolist())


    val_loss = validation_loss / (j + 1)
    val_acc = sum(validation_acc) / len(validation_acc)

    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)

    scheduler.step(val_loss)


    # 保存结果
    Res.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'f1_score': f1,
        'confusion_matrix': cm,
        'class_report': class_report,

    })

    train_loss_plt.append(train_loss)
    val_loss_plt.append(val_loss)
    val_acc_plt.append(val_acc)

    print(f'Epoch {epoch+1}/{n_epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {f1:.4f}')

    if val_acc > best_val_acc+1e-6:
        best_val_acc = val_acc
        train_loss_output=train_loss
        epoch_output=epoch
        train_acc_output=train_acc
        val_loss_output=val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"Early stopping counter increment：{patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered, terminating training")
            break

print('Confusion Matrix:')
print(cm)
print('Classification Report:')
print(class_report)

Res.append({'ouput_epoch': epoch_output,'train_loss': train_loss_output,'train_acc': train_acc_output,'val_loss':val_loss_output,'val_acc':best_val_acc})


log_save_path=f"Output\\eeg_model\\eeg_model_{training_start_time}.npy"
np.save(log_save_path, Res)
model_save_path=f"Output\\eeg_model\\eeg_model_{training_start_time}.pth"
torch.save(net.state_dict(), model_save_path)