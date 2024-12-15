Overview: This project aims to extract features from unstructured business website text data from BERT embeddings. These features would then be run and evaluated on tree-based models to characterize businesses on innovative capacity/status.

General Description of Notebooks (.txt files): 
1. Generate Embeddings: Pull consolidated embeddings from AZURE container database + subset to 10 random samples per Business UEN.
2. Merge TTT: Preparation of Train & Validation datasets by merging embeddings with labelled business datasets.
3. Modelling: Evaluation of Machine Learning models on various evaluation metrics - Accuracy, Precision, Recall, F1-score and ROC-AUC curve.
4. Inference: Utilisation of trained models to run inference on Business Register database. Rule-based inference classifies firm as "Innovative" as long as one sub-page embedding returns a positive classification. Average prediction probability acorss all subpages are returned.
