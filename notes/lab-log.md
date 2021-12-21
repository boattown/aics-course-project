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

Notes to myself: Consider splitting the data into train/val/test considering the affordances of the objects instead of randomly, so that e.g. pen, telescope and laptop are in the train set and pencil, microscope and desktop computer in the test set.

Also consider adding multiple images of each object. This way, the model can train on mapping object with its affordances multiple times instead of seeing the same word embedding each time.
