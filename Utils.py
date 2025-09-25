'''
Modified from original work by Victor Delvigne
ISIA Lab, Faculty of Engineering University of Mons, Mons (Belgium)
victor.delvigne@umons.ac.be
Source: SEEN SOON
Copyright (C) 2019 - UMons

Original license terms (GNU LGPL 2.1+) below:
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

EyeMotionDataset - Robust feature processor and dataset loader for eye-tracking data
MultimodalDataset -
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
from torch.utils.data.dataset import Dataset
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
import cv2

class CombDataset(Dataset):
    """EEGLearn Images Dataset from EEG."""

    def __init__(self, label, image, array):
        self.label = label
        self.array = array
        self.Images = image

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.Images[idx]
        label = self.label[idx]
        array = self.array[idx]
        sample = (image, array, label)

        return sample

from torch.utils.data import Dataset
import torch


class EyeFeatureProcessor:
    """Robust feature standardized processor"""

    def __init__(self):
        self.feature_median = None
        self.feature_iqr = None

    def fit(self, data: np.ndarray):
        """Fit the data statistics  (called on the training set)"""
        # 计算中位数和四分位距
        self.feature_median = np.median(data, axis=0)
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        self.feature_iqr = q3 - q1
        # 防止除零
        self.feature_iqr[self.feature_iqr == 0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply standardization transformation"""
        return (data - self.feature_median) / self.feature_iqr

    def save(self, path: str):
        """Save processing parameters"""
        np.savez(path,
                 median=self.feature_median,
                 iqr=self.feature_iqr)

    def load(self, path: str):
        """Load processing parameters"""
        params = np.load(path)
        self.feature_median = params['median']
        self.feature_iqr = params['iqr']





class EyeMotionDataset(Dataset):

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 processor: EyeFeatureProcessor = None,
                 noise_scale: float = 0.1):

        # 预处理
        if processor is not None:
            self.features = processor.transform(features)
        else:
            self.features = features

        self.labels = labels
        self.noise_scale = noise_scale
        self.is_training = False

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple:
        feature = self.features[idx].astype(np.float32)
        label = self.labels[idx]

        # Inject Gaussian Noise during training(only for the first 31-dimensional continuous features)
        if self.is_training:
            noise = np.random.normal(0, self.noise_scale, size=31)
            feature[:31] += noise

        return torch.from_numpy(feature), torch.tensor(label, dtype=torch.long)

    def set_training_mode(self, mode: bool):
        """Confiture data augmentation toggle"""
        self.is_training = mode


class MultimodalDataset_DEEP(Dataset):
    def __init__(self, label, image, array,eye,augment=False):
        self.label = label
        self.array = array
        self.Images = image
        self.eye = eye
        self.augment = augment
        self.augment_prob=0.5

    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.Images[idx]
        label = self.label[idx]
        array = self.array[idx]
        eye=self.eye[idx]

        if self.augment:
            image = self._augment_image(image)
            eye = self._augment_eye(eye)

        sample = (image, array, eye,label)

        return sample

    def _augment_image(self, img):
        if np.random.rand() < self.augment_prob:
            # Horizontal flip (mirroring)
            img = cv2.flip(img, 1)

        if np.random.rand() < self.augment_prob:
            # Random rotation（-15°~15°）
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        if np.random.rand() < self.augment_prob:
            # Gaussian noise（for float64）
            noise = np.random.normal(0, 0.01, img.shape).astype(np.float64)
            img = img + noise

        return img

    def _augment_eye(self, eye):
        # Eye Movement Feature Enhancement
        if np.random.rand() < self.augment_prob:
            # 特征值随机缩放（0.9~1.1倍）
            scale = np.random.uniform(0.9, 1.1)
            eye = eye * scale

        return eye