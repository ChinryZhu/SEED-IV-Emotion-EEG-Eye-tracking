'''

convert_mat_to_npy - Preprocess the original SEED-IV dataset into samples for multimodal training
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

import scipy.io
import json
import sys
import warnings
from Utils_Bashivan import *

np.set_printoptions(threshold=sys.maxsize)
from numpy import ndarray

path_eye="Input\\eye_feature_smooth\\eye_feature_smooth"
path_eeg="Input\\eeg_feature_smooth\\eeg_feature_smooth"
with open("Input\\file_list.json", 'r') as f:
    file_list = json.load(f)
with open("Input\\label_list.json", 'r') as f:
    label_list = json.load(f)

label=(label_list[:24],label_list[24:48],label_list[48:])

eeg_array_data=None
label_output_list=[]
img_data=None
eye_array_data=None

for file in file_list:

    file_path_eye=path_eye+file
    file_path_eeg=path_eeg+file
    count = file.split("\\")[1]

    mat_data_eye = scipy.io.loadmat(file_path_eye)
    mat_data_eeg = scipy.io.loadmat(file_path_eeg)
    key_list_eye = [key for key in mat_data_eye if not key.startswith('__')]
    key_list_eeg_total = [key for key in mat_data_eeg if not key.startswith('__')]
    key_list_eeg = [name for name in key_list_eeg_total if ('LDS' in name) and ('psd' in name)]
    # ↑ Change this part if different type of features is needed
    dtype = [('key_eye', 'U10'), ('key_eeg', 'U10'),('label', int)]
    main_list = np.array(list(zip(key_list_eye,key_list_eeg, label[int(count) - 1])), dtype=dtype)
    for key in key_list_eeg:
        raw_data_eeg = mat_data_eeg[key]
        trans_data_eeg = np.transpose(raw_data_eeg, (1, 0, 2))
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            img = gen_images("Input\\SEED_IV_POS_2D.npy", trans_data_eeg, 32, normalize=True, edgeless=False)
            target_warn = any("Numerical issues were encountered" in str(w.message) for w in ws)
            if target_warn:
                print(f"{key} in {file}: Error\n")
                continue
        print(img.shape,trans_data_eeg.shape)
        key_eye=main_list[main_list['key_eeg']==key]['key_eye'].item()

        raw_data_eye=mat_data_eye[key_eye]
        trans_data_eye = raw_data_eye.T
        if np.any(np.isnan(trans_data_eye)):
            print(f"Invalid samples detected  key={key}   file={file}")
            continue

        if eeg_array_data is None:
            eeg_array_data = trans_data_eeg
        else:
            eeg_array_data = np.concatenate([eeg_array_data, trans_data_eeg], axis=0)

        if img_data is None:
            img_data = img
        else:
            img_data = np.concatenate([img_data, img], axis=0)

        if eye_array_data is None:
            eye_array_data = trans_data_eye
        else:
            eye_array_data = np.concatenate([eye_array_data, trans_data_eye], axis=0)

        print(eeg_array_data.shape,img_data.shape,eye_array_data.shape)
        samples, channels, bands = trans_data_eeg.shape
        target_label = int(main_list[main_list['key_eeg'] == key]['label'])
        for i in range(int(samples)):
            label_output_list.append(target_label)


label_output_array: ndarray = np.array(label_output_list, dtype=int)
print(img_data.shape)
print(eye_array_data.shape)
print(eeg_array_data.shape)
indices=np.random.permutation(eeg_array_data.shape[0])
eeg_array_data_shuffled = eeg_array_data[indices]
eye_array_data_shuffled = eye_array_data[indices]
img_data_shuffled = img_data[indices]
label_output_array_shuffled = label_output_array[indices]

for i in range(15):
    original_idx = indices[i]
    print(f"Shuffled Index {i}: Original Index {original_idx}")
    print("X Sample Consistency Check:", np.allclose(eye_array_data_shuffled[i], eye_array_data[original_idx]))
    print("Y Label Consistency Check:", label_output_array_shuffled[i] == label_output_list[original_idx])
    print("Z Sample Consistency Check:", np.allclose(eeg_array_data_shuffled[i], eeg_array_data[original_idx]))


np.save('Input\\multi_array_eeg.npy', eeg_array_data_shuffled)
np.save('Input\\multi_label.npy', label_output_array_shuffled)
np.save('Input\\multi_img.npy', img_data_shuffled)
np.save("Input\\multi_array_eye.npy", eye_array_data_shuffled)








