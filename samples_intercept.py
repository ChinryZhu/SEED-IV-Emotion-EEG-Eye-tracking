'''

samples_intercept - Shuffle and sample data proportionally
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

import numpy as np

def intercept(label,sample_size,random_seed=None,class_num=4):
    np.random.seed(random_seed)
    sample_counts=label.shape[0]
    classes, counts = np.unique(label, return_counts=True)
    ratios=counts/sample_counts
    sample_per_class=(ratios * sample_size).astype(int)
    #print(classes,counts,sample_counts,sample_per_class)
    remainder = sample_size - sample_per_class.sum()
    if remainder > 0:
        # Distribute the remainder to the top few categories with the largest proportions
        sorted_indices = np.argsort(-ratios)  # Sort by proportion in descending order
        for i in range(remainder):
            sample_per_class[sorted_indices[i]] += 1
    #print(sample_per_class)

    selected_indices = []

    for i in range(class_num):
        class_indices = np.flatnonzero(label == classes[i])
        selected = np.random.choice(class_indices, sample_per_class[i], replace=False)
        selected_indices.extend(selected)
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    return selected_indices

def label_corrupted(label,corrupted_rate,random_seed=None):
    np.random.seed(random_seed)
    sample_counts = label.shape[0]
    classes, counts = np.unique(label, return_counts=True)
    ratios = counts/sample_counts
    sample_corrupted = int(round(float(corrupted_rate) * sample_counts))
    sample_per_class = (ratios * sample_corrupted).astype(int)
    remainder = sample_corrupted - sample_per_class.sum()
    if remainder > 0:
        sorted_indices = np.argsort(-ratios)
        for i in range(remainder):
            sample_per_class[sorted_indices[i]] += 1

    corrupted_label=label.copy()
    corrupt_idx = np.array([], dtype=int)

    for i in range(len(classes)):
        cls = classes[i]
        indices = np.where(label == cls)[0]
        selected = np.random.choice(indices, size=sample_per_class[i], replace=False)
        corrupt_idx = np.concatenate([corrupt_idx, selected])

    shuffled_labels = corrupted_label[corrupt_idx]
    np.random.shuffle(shuffled_labels)
    corrupted_label[corrupt_idx] = shuffled_labels

    return corrupted_label





