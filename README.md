# Hyper-RKHS-Classifier
A classifier which includes Hyper-RKHS regularization to learn both kernel and hypothesis, which is found by minimizing the lagrangian of a Least Square SVM  with a Hyper-RKHS regularizer.

The coefficients corresponding to a linear combination of hyperkernels in Hyper-RKHS can be solved for in closed form taking quadratic, taking quadratic as opposed to cubic time.  Since the kernel matrix is of low rank, the solution to the matrix inverse problem in order to solve for the linear kernel parameters in the least square SVM formulation should lie in a low dimensional subspace (r << n), leading to a quick convergence.

Usage is inhereted from SKLEARN BaseEstimator and ClassifierMixin classes and can be seen in the test_HRKHS.py file.
