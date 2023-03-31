# Performance Comparison of Several Location Recommendation Methods

This repository contains code for comparing the performance of several location recommendation methods. The purpose of this project is to help researchers and developers evaluate the effectiveness of different location recommendation methods and choose the best one for their needs.

## Datasets
The datasets used in this project are from Gowalla and Yelp. For Yelp, you can access the preprocessed dataset [here](https://github.com/rahmanidashti/LBSNDatasets/tree/master/Yelp/Yelp_1). For Gowalla, you can access the original dataset [here](http://snap.stanford.edu/data/loc-gowalla.html). You can find the code to preprocess the Gowalla dataset above. The dataset-preprocessing code is to filter out those users with fewer than 15 check-in POIs and those POIs with fewer than 10 visitors. The result is nearly the same as [this dataset](https://github.com/rahmanidashti/LBSNDatasets/tree/master/Gowalla/Gowalla_1).

## Methods
The following location recommendation methods are implemented in this project:
 - **Additive Markov Chain**: [Enabling Probabilistic Differential Privacy Protection for Location Recommendations](https://ieeexplore.ieee.org/document/8306835)
 - **LORE**: [LORE: exploiting sequential influence for location recommendations](https://dl.acm.org/doi/10.1145/2666310.2666400)
 - **LORE with Gravity Model**: [Spatiotemporal Sequential Influence Modeling for Location Recommendations A Gravity-based Approach](https://dl.acm.org/doi/10.1145/2786761)
 
 All of the methods are planned to run in multiprocessing. I also include the simple bash script so it can be run on a server.
 ## Acknowledgements
 The implementation code for LORE and Additive Markov Chain are originally from [this project](https://github.com/dbgroup-uestc/cuiyue/tree/master/RecSys%20-2017/6_LORE) which was created by [the owner](https://github.com/cuiyuebing), Ms. Cuiyuebing. I only made a few minor modifications to enable the code to run in multiprocessing. This will be helpful if you plan to run it on a server. 
