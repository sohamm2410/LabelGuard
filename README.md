# LabelGuard
### Investigating the Impact of Label Noise on Environmental Sound Classification

**LabelGuard** is an applied machine learning project investigating how incorrect training labels affect environmental sound classification models and evaluating approaches for identifying and mitigating label noise.

The project uses the ESC-50 environmental sound dataset to establish a controlled classification pipeline, introduce synthetic label corruption, measure its impact on model performance, and investigate whether label-cleaning strategies can improve classification reliability.

> **Project status:** Research and implementation in progress.

---

## 1. Problem Statement

Supervised machine learning relies on correctly labelled training data. In practice, datasets may contain annotation errors caused by human mistakes, ambiguous samples, inconsistent labelling guidelines, or limitations in the annotation process.

These errors, commonly referred to as **label noise**, can introduce misleading supervision during model training.

For environmental sound classification, a recording containing dog barking might incorrectly be annotated as rain. A model trained on such data may learn incorrect associations, potentially affecting its generalization and classification performance.

LabelGuard investigates this problem through controlled experiments on environmental audio data.

## 2. Research Objective

The primary objective is to investigate the relationship between training-label quality and classification performance.

The project focuses on the following questions:

- How does increasing the proportion of incorrect training labels affect classification performance?
- Are different classification architectures affected differently by label noise?
- Can potentially mislabeled samples be identified using model predictions and confidence-based analysis?
- To what extent can label-cleaning strategies recover performance?

The objective is not simply to maximize classification accuracy, but to understand the reliability of models trained using imperfect annotations.

## 3. Dataset

### ESC-50 — Environmental Sound Classification

The project uses the ESC-50 dataset, a publicly available collection of environmental audio recordings.

| Attribute | Description |
|---|---|
| Dataset | ESC-50 |
| Total recordings | 2,000 |
| Number of classes | 50 |
| Recordings per class | 40 |
| Duration per recording | 5 seconds |
| Sampling frequency | 44.1 kHz |
| Audio channels | Mono |
| Audio format | WAV |

The dataset covers five broad groups of environmental sounds:

- Animals
- Natural soundscapes and water
- Human, non-speech sounds
- Interior/domestic sounds
- Exterior/urban sounds

**Dataset source:** ESC-50, created by Karol J. Piczak.

The dataset is maintained separately from this repository. Refer to the original dataset documentation for licensing, attribution, and download instructions. Its CC BY-NC licence restricts commercial use.

## 4. Technical Approach

The planned implementation is divided into the following stages.

### 4.1 Dataset Exploration

Inspect the dataset structure, metadata, class distribution, audio properties, and sample recordings.

### 4.2 Audio Preprocessing

Prepare audio signals for feature extraction. This stage will address audio loading, signal representation, and consistency of input characteristics.

### 4.3 Feature Extraction

Investigate two commonly used audio representations:

**Mel-Frequency Cepstral Coefficients (MFCC)**

MFCCs represent the short-term spectral characteristics of an audio signal using a perceptually motivated frequency scale.

**Mel Spectrogram**

Mel Spectrograms represent how the frequency content of an audio signal changes over time, mapped onto the Mel frequency scale.

### 4.4 Classification Models

The following model families are planned for investigation:

| Model | Purpose |
|---|---|
| Random Forest | Classical machine learning baseline |
| XGBoost | Gradient-boosted decision tree model |
| CNN | Deep learning model for audio representations |

The final model configurations and input representations will be documented as implementation progresses.

### 4.5 Controlled Label Noise Experiments

Artificial label corruption will be introduced into the training labels at controlled noise levels.

The objective is to compare model behaviour under different training-label conditions while maintaining a consistent evaluation procedure.

The clean evaluation labels will be retained separately to avoid contaminating the evaluation process.

### 4.6 Label Noise Investigation and Mitigation

Potentially incorrect training annotations will be investigated using model behaviour and suitable sample-level analysis techniques.

Candidate cleaning strategies will be evaluated to understand whether correcting suspicious labels improves classification performance.

### 4.7 Evaluation

Model performance will be assessed using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Cohen's Kappa

Results will be compared across clean-label, noisy-label, and label-mitigation experiments.

---

## 5. Technology Stack

| Area | Technology |
|---|---|
| Programming language | Python |
| Data manipulation | Pandas, NumPy |
| Data visualization | Matplotlib, Seaborn |
| Audio processing | SoundFile, Librosa |
| Machine learning | Scikit-learn, XGBoost |
| Deep learning | PyTorch |
| Development environment | Jupyter Notebook, VS Code |
| Version control | Git, GitHub |
| Planned deployment | FastAPI, Streamlit, Docker |

Technologies listed for future stages are planned components and should not be interpreted as completed integrations.

## 6. Repository Structure

```text
LabelGuard/
│
├── data/
│   ├── raw/
│   │   └── ESC-50/
│   │       └── meta/
│   │
│   └── processed/
│
├── notebooks/
│   ├── 02_digital_audio_fundamentals.ipynb
│   └── 03_dataset_exploration.ipynb
│
├── src/
├── experiments/
├── reports/
├── app/
├── api/
│
├── requirements.txt
├── .gitignore
└── README.md
```

The repository structure will be expanded as the implementation develops.

Raw audio files are excluded from version control. Dataset metadata and notebooks are maintained in the repository where appropriate.

## 7. Implementation Progress

Development is being carried out incrementally, with each stage documented through notebooks and experiments.

| Stage | Work | Status |
|---|---|---|
| 01 | Project foundation and environment setup | Completed |
| 02 | Digital audio fundamentals | Completed |
| 03 | ESC-50 dataset exploration and EDA | Completed |
| 04 | Audio preprocessing | Planned |
| 05 | MFCC feature extraction | Planned |
| 06 | Mel Spectrogram feature extraction | Planned |
| 07 | Baseline model development | Planned |
| 08 | Label noise injection | Planned |
| 09 | Model evaluation under noisy labels | Planned |
| 10 | Label noise investigation and mitigation | Planned |
| 11 | Comparative analysis | Planned |
| 12 | Application and deployment | Planned |

### Completed Work

**Digital Audio Fundamentals**

- Generated and examined a synthetic 440 Hz sine wave.
- Explored sampling rate, amplitude, duration, and digital audio representation.
- Visualized and played audio signals.
- Saved and loaded WAV audio files.

**Dataset Exploration**

- Loaded the ESC-50 metadata using Pandas.
- Examined the 50 sound categories.
- Analyzed class distribution.
- Checked missing metadata values and duplicate filenames.
- Inspected WAV file properties.
- Examined recording duration, sampling frequency, and channel configuration.
- Played sample environmental recordings.
- Visualized audio waveforms.
- Connected audio filenames with their assigned metadata labels.
- Manually inspected a small sample of recordings.

No quantitative label-noise experiment or mitigation result has been established at this stage.

## 8. Getting Started

### Prerequisites

- Python 3.x
- Git
- Jupyter Notebook or VS Code
- ESC-50 dataset

### Clone the Repository

```bash
git clone "https://github.com/sohamm2410/LabelGuard.git"
cd LabelGuard
```

### Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dataset Setup

Download the ESC-50 dataset from its original repository and extract it under:

```text
data/raw/ESC-50/
```

The expected dataset structure is:

```text
data/raw/ESC-50/
│
├── audio/
│   └── *.wav
│
└── meta/
    └── esc50.csv
```

The audio directory is intentionally excluded from Git tracking.

### Run the Notebooks

Launch Jupyter:

```bash
jupyter notebook
```

Open the notebooks in sequence to explore the implementation.

---

## 9. Reproducibility and Experimental Integrity

The following principles will guide the experimental implementation:

- Preserve the original dataset and its annotations.
- Maintain separate copies of clean and corrupted labels.
- Apply synthetic label noise only to the designated training labels.
- Keep evaluation labels clean.
- Record noise levels, model configurations, and experimental parameters.
- Use consistent evaluation metrics when comparing experiments.
- Document assumptions, limitations, and observed results.

These practices are important for making comparisons meaningful and reproducible.

## 10. Scope and Limitations

This project focuses on controlled experiments using the ESC-50 environmental sound dataset.

The findings will be interpreted within the context of the selected dataset, models, and noise-generation procedures.

Synthetic label noise does not necessarily represent the full complexity of real-world annotation errors. Therefore, conclusions from controlled experiments should not automatically be generalized to every audio classification application.

## 11. Future Work

- Complete audio preprocessing and feature extraction.
- Establish baseline classification models.
- Implement controlled label corruption experiments.
- Investigate sample-level label error detection.
- Evaluate label correction and cleaning techniques.
- Compare classification performance across experimental conditions.
- Develop an interface for exploring predictions and label quality.

## 12. Author

**Soham**

Bachelor of Engineering — Information Technology

Project focus: Machine Learning, Audio Classification, Data Quality, and Reliable ML Systems.

---

*LabelGuard is an ongoing independent learning and engineering project. Implementation details and experimental findings will be updated as development progresses.*