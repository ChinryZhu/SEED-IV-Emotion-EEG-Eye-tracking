from torch.utils.data import DataLoader

from multi_model import *
from Utils import *
import time
import warnings
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F

training_start_time=datetime.now().strftime("%m-%d_%H %M %S")

warnings.simplefilter("ignore")

t = time.time()
device = torch.device("cpu")

eeg_array_path="Input\\multi_array_eeg.npy"
img_path="Input\\multi_img.npy"
label_path="Input\\multi_label.npy"
eye_array_path="Input\\multi_array_eye.npy"

from samples_intercept import *

eeg_array=np.load(eeg_array_path)
img=np.load(img_path)
label=np.load(label_path).astype(np.int64)
eye_array=np.load(eye_array_path)

sample_size=10000

indices=intercept(label,sample_size)

eeg_array=eeg_array[indices]
img=img[indices]
label=label[indices]
eye_array=eye_array[indices]

indices = np.arange(len(label))
np.random.shuffle(indices)
split = int(0.8 * len(indices))

train_indices = indices[:split]
val_indices = indices[split:]

train_dataset=MultimodalDataset_DEEP(label[train_indices],img[train_indices],eeg_array[train_indices],eye_array[train_indices],augment=True)
val_dataset=MultimodalDataset_DEEP(label[val_indices],img[val_indices],eeg_array[val_indices],eye_array[val_indices],augment=False)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=128,shuffle=False,pin_memory=False)

net = MultiModel().to(device)

criterion_cls = nn.CrossEntropyLoss()
criterion_domain = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=1e-3,
    weight_decay=1e-4
    )


n_epoch=150
best_val_loss = float('inf')
best_val_acc = 0
patience = 15
counter = 0
patience_counter = 0

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
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

criterion = LabelSmoothingCrossEntropy(epsilon=0.05)

for epoch in range(n_epoch):
    net.train()
    train_loss = 0.0
    evaluation = []
    for images, arrays, eyes, labels in train_loader:
        images = images.float().to(device)
        arrays = arrays.float().to(device)
        eyes = eyes.float().to(device)
        labels = labels.long().to(device)

        cls_logits, domain_logits = net(images, arrays, eyes)
        loss = criterion_cls(cls_logits, labels) + 0.5 * criterion_domain(domain_logits,
                                                                          torch.zeros_like(domain_logits))

        optimizer.zero_grad()
        loss.backward()

        total_norm = 0.0
        for p in net.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

        optimizer.step()

        train_loss += loss.item() * labels.size(0)

    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    class_stats = [{"correct": 0, "total": 0} for _ in range(4)]

    with torch.no_grad():
        for images, arrays, eyes, labels in val_loader:
            images = images.float().to(device)
            arrays = arrays.float().to(device)
            eyes = eyes.float().to(device)
            labels = labels.long().to(device)

            outputs = net(images, arrays, eyes)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                lbl = labels[i].item()
                class_stats[lbl]["total"] += 1
                if predicted[i] == lbl:
                    class_stats[lbl]["correct"] += 1

    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    val_acc = 100 * correct / total

    scheduler.step(val_loss)

    # 早停判断
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        train_loss_output = train_loss
        epoch_output = epoch
        val_loss_output = val_loss
        early_stop_counter = 0
        torch.save(net.state_dict(), f"Output/best_model_{training_start_time}.pth")
    else:
        early_stop_counter += 1

    class_acc = [
        100 * s["correct"] / s["total"] if s["total"] > 0 else 0
        for s in class_stats
    ]

    print(f"Epoch {epoch + 1:03d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.2f}% | "
          f"Class Acc: {[f'{x:.1f}%' for x in class_acc]}")

    training_log.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'class_acc': class_acc
    })

    if early_stop_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
training_log.append({'best_acc': best_val_acc})
np.save(f"Output/training_log_{training_start_time}.npy", training_log)