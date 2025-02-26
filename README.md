# Data Integration with LLM Enhanced ETL Pipeline

This repository contains the mechanisms and algorithm developed to present the Master thesis entitled **Enhancing Data Integration and Automation in Datalakes through Large Language Model(LLM)-Driven ETL Pipelines**. The thesis was completed as a requirement for the DATMAS course at the University of Stavanger in the spring semester of 2024.

The theme of the thesis was to find ways to integrate LLMs to enhance the poor conditions in the prevalent ETL data pipelines by automatic schema and complex metadata extraction.

## Table of Contents
  - [Overview](#overview)
  - [Datasets](#datasets)
  - [Installations](#installations)
  - [Acknowledgements](#acknowledgements)

## Overview
ETL pipelines are manually maintained by data engineers and they also process complex metadata about the underlying schema structure of the data. This, in essence, is a cumbersome and error-prone method. This repository holds an approach to incorporate LLMs to understand these schema structures and complex metadata. The models are not included in this repository, however, we worked with different open-source LLM libraries available in the HuggingFace repository. 

## Datasets
  - **AdventureWorks**: is a widely used comprehensive dataset provided by Microsoft for learning and practicing SQL Server and database management. The dataset contains 759254 rows of semi-structured data distributed over 71 tables in 8 modules of imaginary AdventureWorks Cycles data. This dataset can be obtained by following the instructions in [link](https://github.com/Microsoft/sql-server-samples/tree/master/samples/databases/adventure-works)
  - **NortWind**: is another dataset created by Microsoft that is widely recognized and used as a sample database in the field of software development and database management. There are 3374 rows of data distributed
over 13 tables in the dataset by an imaginary global exporter and importer NorthWind Traders. This dataset can be obtained by following the instructions in [link](https://github.com/microsoft/sql-server-samples/tree/master/samples/databases/northwind-pubs)

please note that schema files used for the training process are not included in the datasets.


## Installations
To use the evolutionary strategies for a continuous learning framework, follow these steps:

  - Dowload the datasets from the link provided.
  - Clone this repository: git clone https://github.com/shoshi-cuet/LLM_Enhanched_ETL_Pipeline
  - Install the required dependencies: pip install -r requirements.txt
  - Preprocess the dataset using the [Preprocess_windowed_input.py](https://github.com/shoshi-cuet/LLM_Enhanched_ETL_Pipeline/edit/main/README.md#:~:text=Preprocess_windowed_input) to run the experiments
  - Afterwards, Do the model training and calculate the accuracy of the extracted generated schema.
  - Explore the code examples in the repository to get started.

## Acknowledgements
We would like to express our gratitude to the University of Stavanger and to Antorweep Chakravorty for his valuable contributions and guidance, and to the authors of the referenced papers, datasets, open-source LLM-providers, and the strong community of LLM-related development that have enabled advancements in the field of continuous learning through evolutionary strategies.
