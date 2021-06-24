# A frame for NLP

provide a simple and commom frame for training and test process in NLP 

## Highlights
This simple wrapper based on Transformers (for managing BERT model) and PyTorch. 
1. support cache.

2. support earlystop. 

3. support tensorboard to record experiment.

4. support other tasks by revising dataloader.py and model.py


## Simple way to run for ccks2021 task3

1. prepare bert_pretrained model and revised  '--model_name_or_path'

2. prepare dataset and revised  '--data_dir'

3. run on colab (main.ipynb). turn debug on False before run main.

4. run on local device(main.py). turn debug on False before run main.

## Stucture
The main module contains the follow files:

- data_process.pynb
split Xeon3NLP_round1_train_20210524.txt to train.txt and dev.txt

- The dataset.py
Text process -> read a file and convert it to a format for model (transform file.txt to dataloader).

- model.py
Build Model  ->  Bert embedding+FC model(SequenceClassification) and other model can be added. 

- pipeline.py contain two classes.  Trainer is for training process. Tester is for testing process which contains model predict and evaluation.

- main.py
main file to run on local device.

- main.ipynb
main file to run on Colab.The main structure is exactly the same as maim.py

- config.py 

- data folder contains files(train.txt,dev.txt,Xeon3NLP_round1_test_20210524.txt). The first two files is processed from Xeon3NLP_round1_train_20210524.txt(use data_process.ipynb) 

- pretrained_model folder contains pretrained model files (.bin,.json,.txt) which can be downloaded [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) for chinese text (also support other language).



## Other 
- 1.one can add some trick to import prediction performance. For example model average,Pseudo label,model stacking. Details can be seen[BDCI top1 scheme](https://github.com/cxy229/BDCI2019-SENTIMENT-CLASSIFICATION).
- 2.other deep network model can be added in model.py not only Bert class models.
