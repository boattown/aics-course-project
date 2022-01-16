# Encoded object affordances in uni- and multi-modal language models
aics-course-project

### Folders

Lab log can be found in /notes.

Object labels and affordance annotations from (Zhu et al, 2014) can be found in /data.

The script for training the models can be found in /code as well as the notebook for testing.


### Instructions how to run your system, what other componenets/datasets are required and where they can be obtained

To train the models, run <code>python3 code/train.py</code>

To train the models with a specified manual seed, run <code>python3 code/train.py --bert_seed=n --visual_bert_seed=n</code>

No additional datasets are required.


### Work plan

- [x] Prepare the data.
- [x] Extract embeddings of objects and affordances from BERT and VisualBERT.
- [x] Train a simple model (linear layer and Sigmoid) to map embeddings to a truth value (1/0) for BERT and VisualBERT.
- [x] Test the model on seen and unseen objects for BERT and VisualBERT.
- [x] Compare and present the results (in total and per object, in terms of accuracy, precision, recall and F1).
- [x] Train the models 10 times with different data seeds, present the results in a diagram, and take the best checkpoint.

Comments:
