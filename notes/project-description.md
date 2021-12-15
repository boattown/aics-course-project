# Encoded object affordances in contextual and multi-modal language models
Link to the GitHub repository: https://github.com/boattown/aics-course-project

As we have discussed in the course seminars, generating a suitable verb for actions is a challenging task in automatic description of still images. One approach to this is to use a knowledge base to assign affordance labels to objects given their attributes such as texture, material, and weight (Zhu et al., 2014). In this course project, I will examine what knowledge of object affordances is encoded in a large contextual language model (BERT) trained on text only and compare it with a multi-modal language model (LXMERT) trained on both images and text.

Ilharco et al. (2020) use probing to compare how well multi-modal and contextual language models map nouns from image descriptions to objects in images. To achieve this, they train a simple LSTM to perform the mapping based on the knowledge encoded in the language models. They find that the multi-modal models perform slightly better, but that knowledge about visual features is also encoded in the contextual language models trained on text only.

To answer my question, I will train an LSTM to map objects to the correct affordance(s) and compare how well BERT and LXMERT perform in terms of accuracy on seen and unseen objects. As in Ilharco et al. (2020), I will not do any finetuning for the task. I will use a sample of the affordances, objects and images used in Zhu et al. (2014) which in turn are retrieved from the Stanford 40 actions dataset (Yao et al., 2011) and ImageNet.

I expect the multi-modal language model to perform better on this task, since they benefit from being grounded in visual features. However, I expect the contextual language model to perform similarly since it encodes knowledge that it is not explicitly trained on.

Pre-processing the data might be challenging as well as working with both images and text. However, I expect the training of the LSTM and the evaluation to be more straight forward.
## References
Ilharco, Gabriel & Zellers, Rowan & Farhadi, Ali & Hajishirzi, Hannaneh. (2020). Probing Contextual Language Models for Common Ground with Visual Representations.

B. Yao, X. Jiang, A. Khosla, A.L. Lin, L.J. Guibas, and L. Fei-Fei. Human Action Recognition by Learning Bases of Action Attributes and Parts. Internation Conference on Computer Vision (ICCV), Barcelona, Spain. November 6-13, 2011.

Zhu Y., Fathi A., Fei-Fei L. (2014) Reasoning about Object Affordances in a Knowledge Base Representation. In: Fleet D., Pajdla T., Schiele B., Tuytelaars T. (eds) Computer Vision – ECCV 2014. ECCV 2014. Lecture Notes in Computer Science, vol 8690. Springer, Cham. https://doi.org/10.1007/978-3-319-10605-2_27
