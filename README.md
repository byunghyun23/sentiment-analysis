# Sentiment Analysis Method and System Combining Domain Sentiment Dictionary and Word Embedding Technique (Patent 10-2022-0066186)
## Introduction
This is a TensorFlow implementation for Sentiment Analysis Method and System Combining Domain Sentiment Dictionary and Word Embedding Technique. This work has been filed.

![image](https://github.com/byunghyun23/sentiment-analysis/blob/main/assets/fig2.png)

![image](https://github.com/byunghyun23/sentiment-analysis/blob/main/assets/fig1.png)

## Installation
1. Clone the project
```
git clone https://github.com/byunghyun23/sentiment-analysis
```
2. Install the prerequisites
```
pip install -r requirements.txt
```

## Dataset
[Naver sentiment movie corpus](https://github.com/e9t/nsmc) was used.
Dataset download is included in the code.

## Training
```
python train.py
```
You can get the prediction results of test data in
```
results.csv
```
after training.
