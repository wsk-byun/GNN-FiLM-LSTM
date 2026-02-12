# IC50 Prediction with GROVER and Protein Embeddings

This repository contains the implementation of a drug-target interaction model that predicts IC50 values.
It fuses molecular embeddings (from a pretrained GROVER transformer) with protein sequence embeddings using FiLM.


### Predicting IRAK4 IC50 values with GNN
- Molecule Input Expressions ([Part 1](https://wsbyun251.tistory.com/5))
- Data Preprocessing ([Part 2](https://wsbyun251.tistory.com/9))
- Pretraining with GROVER ([Part 3](https://wsbyun251.tistory.com/14))
- Finetuning ([Part 4](https://wsbyun251.tistory.com/15))

<br>

## Workflow Overview
1) Data Preparation: Extract and clean data from ChEMBL.

2) Pretraining: Train the GROVER model on a large corpus of unlabeled SMILES strings.

3) Finetuning: Train the custom fusion model (Molecule + Protein) on the IC50 dataset.

---

### 1. Data Preparation

Download chembl_36_sqlite.tar.gz from the ChEMBL database. Running chembl_data.py extracts:
- ~2.4 million unique, sanitized SMILES strings for pretraining, and
- labeled drug-target pairs (IC50 values), filters for valid protein sequences, and imputes missing pIC50 values for finetuning.

Outputs:
- open/ssl_data.csv: Large unlabelled dataset for GROVER pretraining.
- grover/exampledata/finetune/ssl_molecules_prot.csv: Labeled dataset containing SMILES, Protein Sequences, and pIC50 values.
- grover/exampledata/finetune/ssl_molecules.csv: Helper file containing only SMILES and labels (used for feature generation).

---

### 2. Pretraining (GROVER)

For pretraining, I followed the steps specified in the [original GROVER repository](https://github.com/tencent-ailab/grover).
These steps include extracting semantic motifs, building the vocabulary, and splitting the data.

Before training the final model, I pretrained GROVER with the following specifications:
- hidden_size=128
- depth=1
- num_attn_head=4
- num_mt_block=1
- epochs=500
- earlystop_patience=10

I modified the original GROVER code so that it can halt its training once its validation loss stops improving.
The final pretrained model is saved as grover/model/ssl/model.ep500.

---

### 3. Finetuning (Fusion model)

The final model is a custom architecture defined in fusion_model.py.

GroverFinetuneTask class combines
- GROVER: Processes the molecule graph.
- Bi-LSTM: Processes the protein sequence.
- FiLM & Attention: Fuses the molecular and protein embeddings.
- Training Loop: A custom loop that handles the specific data inputs (graph components + protein indices).

