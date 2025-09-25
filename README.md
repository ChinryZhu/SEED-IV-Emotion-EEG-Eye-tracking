# Emotion-EEG & Eye Movement

An emotion estimation model based on the SEED-IV dataset. Composed by the combination of three deep-learning models(CNN for EEG spatial-domain features, RNN for EEG frequency-domain features and FFNN for eye movement features).

## Installation and Dependencies

Installation with pip: `pip install -r requirements.txt`

Import of the environment with conda: `conda env create -f environment.yml`

## Future Improvements

- Theoretically, the optimal approach would be to input the DE features of EEG signals into the RegionRNN network while using PSD features to construct a 2D EEG input for the CNN network.
- The eye-tracking model exhibits mild overfitting, whereas the multimodal model shows more significant overfitting.
- The dataset splitting step did not include a random seed parameter.
- I directly used the feature samples provided by the SEED-IV dataset without implementing feature extraction algorithms on the raw data. BCMI of Shanghai Jiao Tong University might offer such an algorithm, but I have not attempted it.
- Due to practical constraints, no real-world data collection was performed for validation.


## Acknowledgments  

This project references the work of Delvigne et al. from ISIA Lab, UMons.  

&zwnj;**Important notes**&zwnj;:  
- Due to dataset license restrictions (EULA), the example signals provided here are synthetic and do not correspond to any original dataset used in their research.  
- If you use this repository in your research, please cite the original paper:

```bibtex
@article{delvigne2022emotion,
  title={Emotion Estimation from EEG--A Dual Deep Learning Approach Combined with Saliency},
  author={Delvigne, Victor and Facchini, Antoine and Wannous, Hazem and Dutoit, Thierry and Ris, Laurence and Vandeborre, Jean-Philippe},
  journal={arXiv preprint arXiv:2201.03891},
  year={2022}
}