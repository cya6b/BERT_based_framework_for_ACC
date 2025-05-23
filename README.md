# BERT_based_framework_for_ACC
* Code and instructions for our paper: SCIE Journal with editor (2025-May)
# Introduction
This study introduces a novel approach to Automated Compliance Checking (ACC) that integrates a BERT-based framework with Graph Neural Networks (GNNs), specifically designed for building compliance in the Architecture, Engineering, and Construction (AEC) industry. The framework leverages the Building Fire Spatial Information Ontology (BFSO) for regulatory knowledge representation, aligning regulatory concepts with BIM data through BERT's semantic consistency capabilities. The GNNs, built upon the same BERT architecture, process the generated graph datasets to predict compliance by analyzing spatial relationships between building elements. This unified approach addresses the complexity of regulatory language and enhances the scalability and adaptability of ACC systems.

# BFSO ontology development and annotation
The development and annotation of the Building Fire Spatial Information Ontology (BFSO) are based on a BERT-based model that automates the generation of the ontology. The model employs an entity-relationship joint extraction approach, where it automatically identifies entities and relationships within regulatory texts. The model was trained for 60 epochs, and the results of the model's performance can be accessed through the following link:

https://github.com/cya6b/Rule-interpretation-for-fire-safety-ontology/tree/main/model%20result%20documents

Additionally, the following video demonstrates the semantic annotation process applied to single sentences of regulatory text. Using class and attribute labels, the entities and relationships within each sentence are annotated, with the annotated results serving as a gold standard dataset for training the BERT-based entity-relationship extraction model in the BFSO construction process. semantic annotation tools: doccano.

https://github.com/user-attachments/assets/d34cd171-4b97-4f97-8f18-87160b40b8f0


 
 
# Catalog Structure Description
    ├── ReadMe.md           // help file
    
    ├── 7_building_node&link    // Contains seven real-world BIM models in IFC format files, as well as a graph dataset (node&link) generated for GNNs compliance checking based on semantic consistency results.

    ├── BFSO_ontology    //In addition to containing the ontology OWL file, it also includes the textual monads of the natural language description, the results of the semantic annotation, and the expressions of the subject-predicate-object (SPO) triples.

    ├── IFC_ontology    //This collection contains seven ontology files of real-world BIM models, which are stored in the .ttl format.

    ├── code              // Core Code Documentation

    │   ├── hrm.py (a high-recall candidate phrase generation method)

    │   ├── MethodWLNodeColoring.py (for WL code computing)

    │   ├── MethodGraphBatching.py (for subgraph batching)

    │   ├── MethodHopDistance.py (for hop distance computing)

    │   ├── MethodBGCC.py (for BGCC model)
    
    ├── result             // Contains subgraph batch results, WL results, Hop embedding results, and pre-trained BGCC models.

    ├── script_1_preprocess.py             // Compute node WL code, intimacy based subgraph batch, node hop distance.

    ├── script_2_pre_train.py             // For pre-training the BGCC.

    ├── script_3_fine_tuning.py             // As the entry point for executing the BGCC model on the automated compliance checking task.

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

Run the command "python script_3_fine_tuning.py" as the entry point to run the BGCC on BIM automated compliance checking.

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

 
