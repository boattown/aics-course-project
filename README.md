# Encoded object affordances in uni- and multi-modal language models
aics-course-project

### Folders

Lab log can be found in /notes.

Object labels and affordance annotations from (Zhu et al, 2014) can be found in /data.

Code for data preparation can be found in /code.

### Instructions how to run your system, what other componenets/datasets are required and where they can be obtained

python3 code/train.py

python3 code/train.py --bert_seed=1 --visual_bert_seed=2


### Work plan

- [x] Prepare the data.
- [x] Extract embeddings of objects and affordances from BERT and VisualBERT.
- [x] Train a simple model (linear layer and Sigmoid) to map embeddings to a truth value (1/0) for BERT and VisualBERT.
- [x] Test the model on seen and unseen objects for BERT and VisualBERT.
- [ ] Compare and present the results (in total and per object, in terms of accuracy, precision, recall and F1).
- [x] Train the models 10 times with different data seeds, present the results in a diagram, and take the best checkpoint.

Comments:
