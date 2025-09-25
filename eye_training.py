'''

eye_training
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


from torch.utils.data import DataLoader
from samples_intercept import *
from eye_model import *
from Utils import *
import sys
from datetime import datetime
import warnings
import os
import numpy as np

os.makedirs('Output',exist_ok=True)
os.makedirs('Output\\eye_model',exist_ok=True)

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=sys.maxsize)

training_start_time=datetime.now().strftime("%m-%d_%H %M %S")

samples_path = "Input\\multi_array_eye.npy"
labels_path = "Input\\multi_label.npy"
sample_size=10000
samples_input=np.load(samples_path)
labels_input=np.load(labels_path)
indices=intercept(labels_input,sample_size)
samples=samples_input[indices]
labels=labels_input[indices]

labels=labels.astype(np.int64)

#labels=label_corrupted(labels,corrupted_rate=0.1)

indices = np.arange(len(samples))
np.random.shuffle(indices)
split = int(0.8 * len(indices))

train_indices = indices[:split]
val_indices = indices[split:]

processor = EyeFeatureProcessor()
processor.fit(samples[train_indices])

train_dataset = EyeMotionDataset(
    samples[train_indices],
    labels[train_indices],
    processor=processor
)
val_dataset = EyeMotionDataset(
    samples[val_indices],
    labels[val_indices],
    processor=processor
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=128,shuffle=False,pin_memory=False)

device = torch.device("cpu")
#device = torch.device('cuda:0')
model = DeepFFNN()
model = model.to(device)

training_log=[]

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds, targets):
        n_classes = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        ce_loss = -log_preds.gather(1,targets.unsqueeze(1)).squeeze().mean()
        smooth_loss = -log_preds.mean(dim=-1).mean()
        return (1 - self.epsilon) * ce_loss + self.epsilon * smooth_loss


optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        amsgrad=True
    )
criterion = LabelSmoothingCrossEntropy(epsilon=0.05)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        eta_min=1e-5
    )

best_val_acc = 0.0
early_stop_counter = 0
patience = 14

for epoch in range(150):
    model.train()
    train_loss, correct = 0.0, 0
    train_dataset.set_training_mode(True)
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * inputs.size(0)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 4
    class_total = [0] * 4

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = 100 * correct / total

    class_acc = [100 * class_correct[i] / class_total[i] if class_total[i] != 0 else 0
                 for i in range(4)]

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        train_loss_output = train_loss
        epoch_output = epoch
        val_loss_output = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), f"Output\\eye_model\\best_model_{training_start_time}.pth")
    else:
        early_stop_counter += 1


    print(f"Epoch {epoch + 1:03d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.2f}% | "
          f"Class Acc: {[f'{x:.1f}%' for x in class_acc]}")
    training_log.append({'epoch':epoch, 'train_loss': train_loss, 'val_loss': val_loss,'val_acc': val_acc,'class_acc':class_acc})

    if early_stop_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

training_log.append({'best_acc':best_val_acc})
os.makedirs('Output\\eye_model', exist_ok=True)
np.save(f"Output\\eye_model\\training_log_{training_start_time}.npy", training_log)
