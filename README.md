# Reproducing, Validating, and Enhancing ClinicalBERT 
Madhu Sivaraj (sivaraj4@illinois.edu) and Anish Saha (saha9@illinois.edu)

Group ID: 67, Paper ID: 314

## Abstract
Clinical notes are often underutilized in the medical domain, given its high dimensionality, scarcity, and lack of structure. Unlike its structured, quantitative counterparts such as lab results, procedural codes, and medication history, clinical notes - with the help of deep learning models - have the potential to reveal high-quality, physician-assessed semantic relationships between medical concepts, which would otherwise involve a human perspective. Huang et al. (2020) devised ClinicalBERT, a flexible framework for learning deep representations of clinical notes, which can be useful for domain-specific predictive tasks \cite{cbert}. Pre-trained on unstructured clinical text from MIMIC-III, ClinicalBERT leverages two unsupervised tasks, masked language modeling and next sentence prediction, followed by a problem-specific fine-tuning phase.

The goal of this project is to reproduce, validate, and enhance the results postulated by Huang et al. (2020) for ClinicalBERT, a model fine-tuned on a hospital readmission prediction task. We will attempt to improve performance via the ablations like augmenting the dataset and migrating from the pytorch-pretrained-BERT pipeline to Transformers, while also proving the claim that ClinicalBERT outperforms competitive baseline models (Bag-of-Words, Bi-LSTM, and BERT) using the following performance metrics: area under the receiver operating characteristic curve (AUROC), area under the precision-recall curve (AUPRC), and recall at precision of 80\% (RP80).

## Citation to the original paper (Bibtex format): 
```
@article{DBLP:journals/corr/abs-1904-05342,
  author    = {Kexin Huang and
               Jaan Altosaar and
               Rajesh Ranganath},
  title     = {ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission},
  journal   = {CoRR},
  volume    = {abs/1904.05342},
  year      = {2019},
  url       = {http://arxiv.org/abs/1904.05342},
  eprinttype = {arXiv},
  eprint    = {1904.05342},
  timestamp = {Thu, 25 Apr 2019 13:55:01 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1904-05342.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Link to the original paper’s repo: 
The publicly available code repository provided by the original paper’s authors can be found at https://github.com/kexinhuang12345/clinicalBERT.

## Dependencies:
Run the following command: 
```
pip3 install -r requirements.txt
```

## Data download instructions:
ClinicalBERT relies on the Medical Information Mart for Intensive Care III (MIMIC-III) dataset. To download the dataset, follow the instructions at https://mimic.mit.edu/docs/gettingstarted/.

1. Complete the required [CITI “Data or Specimens Only Research” course](https://www.citiprogram.org/index.cfm?pageID=154&icat=0&ac=0). Supplemental instructions found [here](https://eicu-crd.mit.edu/gettingstarted/access/).
2. Become a credentialed user on [PhysioNet](https://physionet.org/).
3. Navigate to the [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/) on PhysioNet.
4. Sign the data use agreement (DUA). Adherence to the terms of the DUA is paramount.
5. Download the data locally (6.2GB) or follow the tutorials for direct cloud access ([GCP BigQuery](https://mimic.physionet.org/tutorials/intro-to-mimic-iii-bq/); [AWS Athena](https://aws.amazon.com/blogs/big-data/perform-biomedical-informatics-without-a-database-using-mimic-iii-data-and-amazon-athena/)).


Note: We opted to download the data locally. The size of the MIMIC-III dataset zip file was 6.2GB.


## Preprocessing code + command: 
To preprocess the data and generate all necessary datasets (original and augmented) used for this project, run the following notebooks: ```./Preprocess.ipynb``` and ```./ablations/Data_Augmentation.ipynb```. The preprocessing script generates all files in the ```./data/``` directory, which has been omitted from this repo for privacy and file size restrictions.

Ensure that MIMIC-III's directory ```./physionet/``` is in the same relative path as this file.

## Training and Evaluation code + commands:
Training code can be found in ```modeling_readmission.py``` and ```./run_readmission.py```. 
The first file contains the configurations, layers, etc. for the models that make up the architecture of ClinicalBERT, while the latter is a utility script to gather data and configure the training procedure given the input parameters, then train the aforementioned models and evaluate their performance.

The commands for training and evaluation we used are as follows:

### ClinicalBERT
```
python3 ./run_readmission.py --task_name readmission --readmission_mode [READMISSION_MODE] --do_eval --data_dir [INPUT_DIRECTORY_NAME] --bert_model [MODEL_DIRECTORY_NAME] --max_seq_length 512 --output_dir [OUTPUT_DIRECTORY_NAME]
```

#### Train ClinicalBERT Model for Readmission Task based on Early Notes
```
python3 ./run_readmission.py --task_name readmission --readmission_mode early --do_eval --data_dir ./data/3days/ --bert_model ./model/early_readmission --max_seq_length 512 --output_dir ./results/clinicalbert/1/result_early # task: readmission prediction using early (<3 days) clinical notes data
```

#### Train ClinicalBERT Model for Readmission Task based on Discharge Summaries
```
python3 ./run_readmission.py --task_name readmission --readmission_mode discharge --do_eval --data_dir ./data/discharge/ --bert_model ./model/discharge_readmission --max_seq_length 512 --output_dir ./results/clinicalbert/1/result_discharge # task: readmission prediction using discharge summary clinical notes data
```

### BERT
Readmission prediction using early (<3 days) clinical notes data
```
python3 ./run_readmission.py --task_name readmission --do_train --do_eval  --data_dir ./data/3days/ --bert_model ./model/baseline_bert_early --max_seq_length 512 --train_batch_size 128 --learning_rate 2e-5 --num_train_epochs 30 --output_dir [OUTPUT_DIR]
```
Readmission prediction using discharge summary clinical notes data
```
python3 ./run_readmission.py --task_name readmission --do_train --do_eval  --data_dir ./data/discharge/ --bert_model ./model/baseline_bert_discharge --max_seq_length 512 --train_batch_size 128 --learning_rate 2e-5 --num_train_epochs 30 --output_dir [OUTPUT_DIR]
```

### Bag-of-Words
Run all of the cells in the notebook called ```Bag_of_Words_Baseline.iynb``` (found in the ```./baselines/``` directory) to run the Bag-of-Words baseline model and reproduce our reported results.


### BI-LSTM
Run all of the cells in the notebook called ```BiLSTM_Baseline.iynb``` (found in the ```./baselines/``` directory) to run the Bi-LSTM baseline model and reproduce our reported results.

### Ablation 1: Data Augmentation
To reproduce our first set of ablations, simply run the following notebook: ```./ablations/Data_Augmentation.ipynb```, then run our models using the same training as commands as earlier, replacing the ```data_dir``` and ```output_dir``` arguments as necessary. Use the commands below to reproduce our results:

```
python3 ./run_readmission.py --task_name readmission --readmission_mode early --do_eval --data_dir ./data/5days/ --bert_model ./model/early_readmission --max_seq_length 512 --output_dir ./results/ablation_early_5days/1/result_early # task: readmission prediction using early (<5 days) clinical notes data

python3 ./run_readmission.py --task_name readmission --readmission_mode early --do_eval --data_dir ./data/5days/ --bert_model ./model/early_readmission --max_seq_length 512 --output_dir ./results/ablation_early_7days/1/result_early # task: readmission prediction using early (<7 days) clinical notes data

python3 ./run_readmission.py --task_name readmission --readmission_mode early --do_eval --data_dir ./data/aug_early/ --bert_model ./model/early_readmission --max_seq_length 512 --output_dir ./results/clinicalbert/1/result_early # task: readmission prediction using augmented early (<3 days) clinical notes data

python3 ./run_readmission.py --task_name readmission --readmission_mode discharge --do_eval --data_dir ./data/aug_discharge/ --bert_model ./model/discharge_readmission --max_seq_length 512 --output_dir ./results/clinicalbert/1/result_discharge # task: readmission prediction using augmented discharge summary clinical notes data 
```

### Ablation 2: Transformers
To reproduce our second set of ablations, simply run the following notebook: ```./ablations/ClinicalBERT_UpdatedTransformer.ipynb```.

Result metrics (AUPRC, AUROC, and RP-80) for each methodology (and its respective trials) can be seen in its corresponding subdirectory within the ```./results/``` directory.

## Pretrained model:
The pretrained BERT model can be found in ```./model/pretraining/``` -- it was used as a baseline model to compare against.

## Table of results:

### Results obtained from our reproduction of models
30-day readmission using discharge summaries
| Model        | AUROC | AUPRC | RP80 |
|--------------|:-----:|:-----:|:-----:
| ClinicalBERT | 0.748 | 0.723 | 0.276 |
| Bag-of-words | 0.675 | 0.660 | 0.210 |
| BI-LSTM      | 0.702 | 0.690 | 0.115 |
| BERT         | 0.501 | 0.510 | 0.004 |

30-day readmission using early clinical notes
| Model        | AUROC | AUPRC | RP80 |
|--------------|:-----:|:-----:|:-----:
| ClinicalBERT | 0.823 | 0.810 | 0.597 |
| Bag-of-words | 0.675 | 0.660 | 0.058 |
| BI-LSTM      | 0.702 | 0.700 | 0.238 |
| BERT         | 0.501 | 0.510 | 0.004 |

Evaluation accuracy of our reproduced models
| Model        | Discharge | Early Notes |
|--------------|:-----:|:-----:|
| ClinicalBERT | 0.647 | 0.740 |
| Bag-of-words | 0.611 | 0.605 |
| BI-LSTM      | 0.616 | 0.606 | 
| BERT         | 0.466 | 0.466 |


### Results obtained from our ablations
30-day readmission using augmented data
| Clinical Notes | AUROC | AUPRC | RP80 |
|--------------|:-----:|:-----:|:-----:
| Discharge (aug.) | 0.795 | 0.760 | 0.276 |
| Early (aug.)     | 0.823 | 0.810 | 0.597 |
| Early (≤5 days)  | 0.796 | 0.800 | 0.521 |
| Early (≤7 days)  | 0.763 | 0.780 | 0.536 |

30-day readmission using ClinicalBERT model (with Transformers)
| Clinical Notes | AUROC | AUPRC | RP80 |
|--------------|:-----:|:-----:|:-----:
| Discharge | 0.745 | 0.720 | 0.207 |
| Early     | 0.758 | 0.740 | 0.380 |
