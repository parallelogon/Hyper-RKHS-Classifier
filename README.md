# Hyper-RKHS-Classifier
A classifier which includes Hyper-RKHS regularization to learn both kernel and hypothesis, which is found by minimizing the lagrangian:
<img src="https://bit.ly/2YErByY" align="center" border="0" alt="\mathcal{L} = \frac{1}{2}\epsilon^\top \epsilon + \frac{\lambda}{2}||f||_H + \frac{\lambda_Q}{2}||k||_{\mathcal{\underline{H}}}" width="244" height="43" />

such that <img src="https://bit.ly/3uV55Oa" align="center" border="0" alt="yf(x) = 1 - \epsilon" width="112" height="19" />

with respect to 
<img src="https://bit.ly/3mDaRjV" align="center" border="0" alt="f \in \mathcal{H}" width="47" height="19" />
and
<img src="https://bit.ly/3oLtyEu" align="center" border="0" alt="k \in \mathcal{\underline{H}}" width="50" height="19" />
with <img src="https://bit.ly/3ADSyjD" align="center" border="0" alt="\mathcal{H}" width="19" height="14" /> the RKHS and corresponding Hyper-RKHS <img src="https://bit.ly/3iP535M" align="center" border="0" alt="\mathcal{\underline{H}}" width="19" height="18" /> 
This is the Least Square SVM formulation with a Hyper-RKHS regularizer.

The coefficients corresponding to a linear combination of hyperkernels in Hyper-RKHS can be solved for in closed form using the above solution, taking quadratic as opposed to cubic time.  Since the kernel matrix is of low rank, the solution to the matrix inverse problem in order to solve for the linear kernel parameters in the least square SVM formulation should lie in a low dimensional subspace (r << n), leading to a quick convergence.

Usage is inhereted from SKLEARN BaseEstimator and ClassifierMixin classes and can be seen in the test_HRKHS.py file.
