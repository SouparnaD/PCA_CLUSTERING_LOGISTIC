# PCA_CLUSTERING_LOGISTIC
1 Question


1 Problem of anomaly detection: You are given the dataset of network user activity, and the task is to classify each user activity as normal or an attack. Attacks are also categorized as follows
- Denial of Service (dos): Intruder tries to consume server resources as much as possible, so that normal users can’t get resources they need. 
- Remote to Local (r2l): Intruder has no legitimate access to victim machine but tries to gain access. 
- User to Root (u2r): Intruder has limited privilege access to victim machine but tries to get root privilege. 
- Probe: Intruder tries to gain some information about victim machine.

Download dataset from here (http://researchweb.iiit.ac.in/~murtuza.bohra/intrusion_ detection.zip). Dataset contains 29 numerical features and ﬁve classes(one normal and four attacks).

1. Part-1: ) Do dimensionality reduction using PCA on given dataset. Keep the tolerance of 10% (knee method), meaning reconstruction of the original data from the reduced dimensions in PCA space can be done with 10% error. You are only allowed to use eigen decomposition or SVD function from python library(do not use library function to compute PCA directly).
2. Part-2:  Use the reduced dimensions from the ﬁrst part and perform Kmeans clustering with k equal to ﬁve(number of classes in the data). Also calculate the purity of clusters with given class label. Purity is the fraction of actual class samples in that cluster. You are not allowed to use inbuilt function for K-means.
3. Part-3:  Perform GMM (with ﬁve Gaussian) on the reduced dimensions from ﬁrst part and calculate the purity of clusters. You can use python library for GMM.
4. Part-4:  Perform Hierarchical clustering with single-linkage and ﬁve clusters. Also calculate the purity of clusters. Create a pie chart comparing purity of diﬀerent clustering methods you have tried for all classes. You can use python library for hierarchical clustering.
5. Part-5:  Original data of network user activity is available here(https: //www.kaggle.com/what0919/intrusion-detection). Original data also contains categorical feature. If you were to do dimensionality reduction on original data, could you use PCA? Justify. Write a paragraph in report for your explanation/justiﬁcation.
2 Question


2. Question carry forwarded from assignment-2. Use the Admission dataset to perform the following task. Dataset can be downloaded from http://preon. iiit.ac.in/~sanjoy_chowdhury/AdmissionDataset.zip
1. Part-1: Implement logistic regression model to predict if the student will get admit.
2. Part-2: Compare the performances of logistic regression model with KNN model on the Admission dataset.
3. Part-3: Plot a graph explaining the co-relation between threshold value vs precision and recall. Report the criteria one should use while deciding the threshold value. Explain the reason behind your choice of threshold in your model.

3 Question

3. Implement logistic regression using One vs All and One vs One approaches. Use the following dataset http://preon.iiit.ac.in/~sanjoy_chowdhury/wine-quality. zip for completing the task. Report your observations and accuracy of the model.
