# BERT_based_framework_for_ACC
* Code and instructions for our paper: SCIE Journal with editor (2025-May)
# Introduction
This study introduces a novel approach to Automated Compliance Checking (ACC) that integrates a BERT-based framework with Graph Neural Networks (GNNs), specifically designed for building compliance in the Architecture, Engineering, and Construction (AEC) industry. The framework leverages the Building Fire Spatial Information Ontology (BFSO) for regulatory knowledge representation, aligning regulatory concepts with BIM data through BERT's semantic consistency capabilities. The GNNs, built upon the same BERT architecture, process the generated graph datasets to predict compliance by analyzing spatial relationships between building elements. This unified approach addresses the complexity of regulatory language and enhances the scalability and adaptability of ACC systems.

# BFSO ontology development and annotation
The development and annotation of the Building Fire Spatial Information Ontology (BFSO) are based on a BERT-based model that automates the generation of the ontology. The model employs an entity-relationship joint extraction approach, where it automatically identifies entities and relationships within regulatory texts. The model was trained for 60 epochs, and the results of the model's performance can be accessed through the following link:
https://github.com/cya6b/Rule-interpretation-for-fire-safety-ontology/tree/main/model%20result%20documents


 
 
# Catalog Structure Description
    ├── ReadMe.md           // help file
    
    ├── dataset    // This section contains SpaceGraph's final JSON format files, along with the original 2D flat image files.

    ├── code              // Core Code Documentation

    │   ├── MethodWLENodeColoring.py (for WLE code computing)

    │   ├── MethodGraphBatching.py (for subgraph batching)

    │   ├── MethodHopDistance.py (for hop distance computing)
    
    ├── result             // Contains subgraph batch results, WLE results, Hop embedding results, and pre-trained models.

    ├── script_1_preprocess.py             // Compute node WLE code, intimacy based subgraph batch, node hop distance.

    ├── script_2_pre_train.py             // For pre-training the NE-Graph-BERT.

    ├── script_3_fine_tuning.py             // As the entry point to run the model on node classification.

    └── script_4_evaluation_plots.py             // Plots drawing and results evaluation purposes.
 
# Contact
If you have any question, feel free to contact me at cya187508866962021@163.com

Fujian Normal University School of Geographical Sciences

# Prerequisites
* python==3.8
* pytorch

  [PyTorch](https://pytorch.org/get-started/locally/)
* numpy
* sklearn
* argparse
* pytorch geometric

  [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
* networkx
* pandas
 
# Execution Guide
###### In order to replicate the results mentioned in paper, please follow the following steps:
Run the command "python script_1_preprocess.py" to compute node WL code, intimacy based subgraph batch, node hop distance.

    python script_1_preprocess.py

Run the command "python script_3_fine_tuning.py" as the entry point to run the model on BIM spatial recognition.

    python script_3_fine_tuning.py

Run the command "python script_4_evaluation_plots.py" for plots drawing and results evaluation purposes.

    python script_4_evaluation_plots.py


# Citation
    @onproceedings{
	    author={Chen, Y. and Jiang, H.},
	    doi={},
	    journal={},
	    year={2025},
    }

 
