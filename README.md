# Encoded object affordances in uni- and multi-modal language models
aics-course-project

### Folders

Object labels and affordance annotations from (Zhu et al, 2014) can be found in /data.

Code for data preparation can be found in /code.

### Instructions how to run your system, what other componenets/datasets are required and where they can be obtained

### Work plan

1. Prepare the data.
2. Extract embeddings of objects and affordances from BERT and LXMERT.
3. Train a simple model (linear layer and Sigmoid) to map embeddings to a truth value (1/0) for BERT and LXMERT.
4. Test the model on seen and unseen objects for BERT and LXMERT.
5. Compare the results.
6. If manageable, replace embeddings of objects with representations of images of the objects.

Comments:
