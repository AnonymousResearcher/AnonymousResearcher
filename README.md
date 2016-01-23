## Structured Scene Parsing by Learning CNN-RNN Model with Sentence Description

### Introduction

This work addresses the problem of structured scene parsing, i.e., parsing the input scene into a configuration including hierarchical semantic objects with their interaction relations. We propose a deep architecture consisting of two networks: i) a convolutional neural network (CNN) extracting the image representation for pixelwise object labeling
and ii) a recursive neural network (RNN) discovering the hierarchical object structure and the inter-object relations.
Rather than relying on elaborative annotations (e.g., manually labeled semantic maps and relations), we train our
deep model in a weakly-supervised manner by leveraging the descriptive sentences of the training images. Specifically, we decompose each sentence into a semantic tree consisting of nouns and verb phrases, and facilitate these trees discovering the configurations of the training images. Once these scene configurations are determined, then the
parameters of both the CNN and RNN are updated accordingly by back propagation. The entire model training is ac-
complished through an Expectation-Maximization method. Extensive experiments suggest that our model is capable of
producing meaningful and structured scene configurations and achieving more favorable scene labeling result on PAS-
CAL VOC 2012 compared to other state-of-the-art weakly supervised methods.


### 
Partial of the source code for training our model is released here. The complete code will be released after the review process.# AnonymousResearcher
