#Remote Sensing
The method in this paper involves cooperation projects with national units, and cannot provide a complete simulation model and code.
Some of the source code we provide is sufficient for your verification requirements.

This is a tensorflow implementation of **Bi-directional Deep Neural Network** .
The bidirectional network has been trained. Several sets of test set data and POLARSCAT measured data are provided here.

###Requirement
''''
tensorflow>=2.2.2
pandas>=0.23.0
numpy>=1.19.5
sklearn>=0.24.2
''''
###Data
''''
│iput/
├──ceshi_data/
│  ├── Test set data and POLARSCAT measured data

├──sigma_ceshi/
│  ├── Enter a set of backscattering coefficients
''''

###Note
1. Turn on ''inverse'' mode;
2. Put one set of data you want to test in ''sigma_ceshi'';
3. Each inversion process requires the loss to converge and stabilize;
4. As mentioned in the article, when the loss does not converge well, you can run it again. 
   Either replace the optimizer, or increase the number of iterations, or change the initialization method.
