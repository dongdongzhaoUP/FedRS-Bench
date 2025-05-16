
<div align="center">
  <h2 align="center">FedRS-Bench: Realistic Federated Learning Datasets and Benchmarks in Remote Sensing</h2>
  <a href="https://arxiv.org/abs/2505.08325" style="display: inline-block; text-align: center;">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.08325-b31b1b.svg?style=flat">
  </a>
</div>


## python environment

conda env create -f fedrs.yml

## **1. General Information**

**Number of Labels**: There are a total of 15 labels: Agriculture, Bareland, Forest, Residential River and so on.

**Number of Clients**: The dataset consists of 135 clients.

## **2. Data Sources**

All the images are collected from 8 different datasets, which are as follows:

Eurosat

UC Merced Land Use Dataset

AID

NWPU - RESISC45

WHU-RS19

NaSC-tg2

Optimal-31

RSD46-WHU

The data from each dataset is distributed to different numbers of simulated clients. The specific distribution numbers are as follows:

Data from the Eurosat dataset is distributed to 46 simulated clients.

Data from the UC Merced Land Use Dataset is distributed to 2 simulated clients.

Data from the AID dataset is distributed to 5 simulated clients.

Data from the NWPU - RESISC45 dataset is distributed to 12 simulated clients.

Data from the WHU-RS19 dataset is distributed to 1 simulated client.

Data from the NaSC-tg2 dataset is distributed to 34 simulated clients.

Data from the Optimal-31 dataset is distributed to 1 simulated client.

Data from the RSD46-WHU dataset is distributed to 35 simulated clients.

## **3. Data Folder Structure**

The data folder is named "data", and its internal structure is as follows:

```plaintext
--data
  --train_set  // Training set folder
    --client_0  // Client 0 folder
      --Agriculture  // Folder for images of the Agriculture category
      --Bareland  // Folder for images of the Bareland category
      --Forest  // Folder for images of the Forest category
      --Residential  // Folder for images of the Residential category
      --River  // Folder for images of the River category
      ......
    --client_1  // Client 1 folder
      --Agriculture
      --Bareland
      --Forest
      --Residential
      --River
      ......
    --client_2  // Client 2 folder
      --Agriculture
      --Bareland
      --Forest
      --Residential
      --River
      ......
    ......
    --client_135  // Client 135 folder
      --Agriculture
      --Bareland
      --Forest
      --Residential
      --River
      ......
  --test_set  // Validation set folder
    --Agriculture  // Folder for images of the Agriculture category (test set)
    --Bareland  // Folder for images of the Bareland category (test set)
    --Forest  // Folder for images of the Forest category (test set)
    --Residential  // Folder for images of the Residential category (test set)
    --River  // Folder for images of the River category (test set)
    --Airport
    --Beach
    --Highway
    --Industrial
    --Port
    --Overpass
    --Parkinglot
    --Bridge
    --Mountain
    --Meadow

# Acknowledgement
This work can not be done without the help of the following repos:

- FedDisco: [https://github.com/MediaBrain-SJTU/FedDisco](https://github.com/MediaBrain-SJTU/FedDisco)

# Citation

```ruby
@article{zhao2025fedrs,
  title={FedRS-Bench: Realistic Federated Learning Datasets and Benchmarks in Remote Sensing},
  author={Zhao, Haodong and Peng, Peng and Chen, Chiyu and Huang, Linqing and Liu, Gongshen},
  journal={arXiv preprint arXiv:2505.08325},
  year={2025}
}
```