# Lab Log

### 7 dec

Discussing topics with Nikolai.

### 13 dec  

Reading paper *Zhu Y., Fathi A., Fei-Fei L. (2014) Reasoning about Object Affordances in a Knowledge Base Representation. In: Fleet D., Pajdla T., Schiele B., Tuytelaars T. (eds) Computer Vision – ECCV 2014. ECCV 2014. Lecture Notes in Computer Science, vol 8690. Springer, Cham. https://doi.org/10.1007/978-3-319-10605-2_27*

### 14 dec

Setting up the GitHub repository. Writing project description.

Reading paper *Ilharco, Gabriel & Zellers, Rowan & Farhadi, Ali & Hajishirzi, Hannaneh. (2020). Probing Text Models for Common Ground with Visual Representations.* 

### 15 dec

Writing and submitting project description.

### 20 dec 

Discussing the project idea and implementation with Nikolai. Will start with a mapping of embeddings of object and affordance, without images.
Starting the data preparation.

### 21 dec

Continue preparing the data. 

Notes: Consider splitting the data into train/val/test considering the affordances of the objects instead of randomly, so that e.g. pen, telescope and laptop are in the train set and pencil, microscope and desktop computer in the test set.

Also consider adding multiple images of each object. This way, the model can train on mapping object with its affordances multiple times instead of seeing the same word embedding each time.

### 3 jan

Extracting embeddings of objects and affordances from BERT and VisualBERT and creating dictionaries to map words with their embeddings.

Notes: Extracting embeddings from Bert's vocabulary does not work for objects/affordances consisting of two words such as "vacuum cleaner". Will need to let Bert encode the objects and affordances instead. I decided to use the penultimate layer of the hidden states (see BERT for feature extraction on https://jalammar.github.io/illustrated-bert/) as a representation of the objects and affordances.

Extracting word embeddings from LXMERT does not seem to be possible since the model requires visual input as well. Because of this, I choose VisualBERT instead.

### 4 jan

Create a (draft of a) simple model with linear layer and sigmoid to map the multiplied object and affordance vectors with a truth value (1 and 0). The model needs to train for many epochs to learn because of data sparsity and the simple architecture. Maybe this is not a problem since performance of BERT and VisualBERT is the goal rather than a high performance.

### 5 jan

Test the probes in terms of accuracy (will add preciosion, recall and F1) on the test-data.

Notes: Replace the words with their corresponding embeddings in the train- and test stage. This way, the same dataloaders can be used for the different models to ensure that difference in performance has to do with the representations of the objects and not the split of the data. Despite this change, the accuracy of the probes on the testing data varies a lot from training to training. 




