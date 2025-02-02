**Overview:** 
This project aims to extract features from unstructured business website text data using pre-generated BERT embeddings. These features would then be run and evaluated on tree-based models to characterize businesses on hiring capacity/status. 

**Dataset folder is not included as datasets are classified as restricted/sensitive national data.*

General Description of Notebooks (.txt files): 
1. Generate Embeddings:
   - Pull consolidated embeddings from AZURE container database + subset to 10 random samples per Business UEN.
   
3. Merge TTT:
   - Preparation of Train & Validation datasets by merging embeddings with labelled business datasets.
   
5. Modelling:
   - Evaluation of Machine Learning models on various evaluation metrics - Accuracy, Precision, Recall, F1-score and ROC-AUC curve.
   
7. Inference:
   - Utilisation of trained models to run inference on Business Register database. Rule-based inference classifies firm as "Innovative" as long as one sub-page embedding returns a positive classification. Average prediction probability across all    subpages are returned.
