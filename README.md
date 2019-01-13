# Machine-Learning-in-Action
### Project Description
In this project I will try to implement some machine learning algorithms in python.
### Branch Management
* There are two branches here: master and dev, I will do daily development in the dev branch, and merge the fully implemented algorithms into the branch master.
* If you want to see the contents of different branches, you can use the following command to switch:
```
$ git checkout dev
$ git checkout master
```
### Data Set
* All classification algorithms (**except decision tree**) for this project will be demonstrated in dealing with handwritten digit recognition. We will use the mnist dataset, which you can get [here](http://yann.lecun.com/exdb/mnist/).

### Contents
* *KNN ( K-NearestNeighbor )*
  **data set : Mnist
  accuracy : 95.8%**
  When using knn for handwritten digit recognition, 95.8% accuracy was obtained on the mnist data set. However, the testing process can take half an hour or more.

* *Decision Tree*
  **data set : None
  accuracy : Unknown**
  This algorithm can perform very well on expert systems, but it is very unsuitable for mnist data sets, so I did not test it.
* *Naive-Bayes*
  **data set : Mnist
  accuracy : 84.1%**
  
  Naive Bayes is able to achieve good results in text classification, but this algorithm is not suitable for processing images. Compared to KNN, it is faster in testing, but the accuracy on the mnist dataset is only about 84%.

### Convention
* **Naming :** ~~The naming methods of files and functions are UnderScore-Case, and the variable names are named with Camel-Case.~~ I will use these two different naming methods:**Camel-Case** and **UnderScore-Case**.
  