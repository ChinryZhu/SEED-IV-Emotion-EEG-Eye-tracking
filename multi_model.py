'''

multi_model - Cross-modal attention model fusing EEG and eye movement features for emotion classification
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


from eye_model import *
from eeg_model import *

class CrossModalAttention(nn.Module):
    def __init__(self, eeg_dim=64, eye_dim=128):
        super().__init__()
        self.eeg_query = nn.Linear(eeg_dim, eye_dim)
        self.eye_key = nn.Linear(eye_dim, eye_dim)
        self.eye_value = nn.Linear(eye_dim, eye_dim)

    def forward(self, eeg, eye):
        # eeg: (B,64), eye: (B,128)
        Q = self.eeg_query(eeg)  # (B,128)
        K = self.eye_key(eye)  # (B,128)
        V = self.eye_value(eye)  # (B,128)

        attn = F.softmax(Q @ K.T / np.sqrt(128), dim=-1)  # (B,B)
        context = attn @ V  # (B,128)
        return context

class MultiModel(nn.Module):

    def __init__(self):
        super(MultiModel, self).__init__()

        self.FeatRNN = nn.Sequential(
            RegionRNN(32, 1, 5),
            )

        self.FeatCNN = FreqCNN()


        self.EyeModel = DeepFFNN(input_dim=31)#新增

        self.b_n = nn.BatchNorm1d(256)

        self.ClassifierFC = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 4),
            )

        self.Discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            )

        self.cross_attn = CrossModalAttention()

        self.fusion_fc = nn.Linear(384, 256)


    def forward(self, image, array,eye):
        # Set initial states
        self.batch_size = image.shape[0]

        array = self.FeatRNN(array)# (batch,64)
        image = self.FeatCNN(image)# (batch,64)
        eye = self.EyeModel(eye)  # (batch,128)

        attn_context = self.cross_attn(array, eye)

        #x = self.b_n(torch.cat((array, image), axis=1))
        x = torch.cat([image,array,eye,attn_context], dim=1)
        x = self.fusion_fc(x)
        x = self.b_n(x)

        class_logits = self.ClassifierFC(x)
        if self.training:
            domain_out = self.Discriminator(x.detach())
            return class_logits, domain_out
        else:
            return class_logits