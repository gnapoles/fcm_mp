# Fuzzy Cognitive Map with Moore-Penrose inverse learning

This package implements a learning method based on the Moore-Penrose inverse for hybrid Fuzzy Cognitive Maps (FCP_MP). In this hybrid model, the user can specify how the problem features interact or let the algorithm compute that matrix from the data using unsupervised learning. The supervised learning step focuses on computing the relationships between the last hidden state of the Fuzzy Cognitive Maps and the outputs. Therefore, the model is devoted to solving multi-output regression problems where problem features are connected in non-trivial ways.

## Install

FCP_MP can be installed from [PyPI](https://pypi.org/project/fcm_mp)

<pre>
pip install fcm_mp
</pre>

## Background

The Fuzzy Cognitive Map model implemented in this package is designed for multi-output regression problems. The model is composed of two blocks. The inner block concerns the input concepts and the relationships between them. In this part, the expert is expected to define weights in the [−1, 1] interval characterizing the relationships between input concepts. The outer block concerns the relationships between input and output concepts. These relationships are not defined by the expert, but computed from the historical data using the Moore-Penrose inverse learning algorithm. Fig. 1 shows an example involving five variables where three are inputs while the others are outputs.

<p align="center">
  <img src="https://github.com/gnapoles/fcm_mp/blob/main/architecture.png?raw=true" width="400" />
</p>

The weight matrix of the FCM_MP model is denoted as $W$, and it is composed of two sub-matrices $W^I$ and $W^O$. $W^I$ contains information concerning relationships ideally defined by the experts. $W^O$ collects relationships that will be learned from historical data. $W^I$ remains fixed in the learning procedure.

## Example Usage

The syntax for the usage of FCM_MP is compatible with scikit-learn library.

### Training

Let's assume that we want to solve a decision-making problem involving three input variables ($x_1$, $x_2$ and $x_3$), two output variables ($y_1$ and $y_2$) and five problem instances.

```python
# This matrix contains the data concerning the input variables
X = np.array([[0.37, 0.95, 0.73],
              [0.60, 0.16, 0.16],
              [0.06, 0.87, 0.60],
              [0.71, 0.02, 0.97],
              [0.83, 0.21, 0.18]])

# This matrix contains the data concerning the output variables
Y = np.array([[0.35, 0.47],
              [0.37, 0.43],
              [0.42, 0.50],
              [0.26, 0.48],
              [0.33, 0.4]])                
```

The next step consists of defining a weight matrix $W^I$ characterizing the interaction between the input variables. Ideally, this matrix should be provided by human experts during a knowledge engineering process. To develop our example, we will use an arbitrary weight matrix defined below.

```python
# This matrix characterizes the relationships between input variables
Wi = np.array([[0.00, -1.00, -0.27],
               [-0.50, 0.00, 0.15],
               [-0.20, 0.23, 0.00]])              
```

Now, we are ready to build the FCM model. Besides the weight matrix defining the interaction between the input variables, we can specify the number of iterations $T$ to be performed during reasoning, the nonlinearity coefficient $\phi$, and the initial slope $\lambda$ and offset $h$ of the sigmoid function.

```python
from fcm.FCM_MP import FCM_MP
# We first define parameters and then build the model
model = FCM_MP(T=10, phi=0.5, slope=1.0, offset=0.0)
model.fit(X,Y)
```

### Prediction

We can contrast the predictions made by the model with the ground-truth. To obtain the predictions for the training data $X$, we can call the `model.predict(X)` function, which results in the following matrix:

$$\[
\hat{\textbf{Y}} = \begin{pmatrix}
0.37 & 0.47 \\
0.36 & 0.43 \\
0.41 & 0.49 \\
0.26 & 0.48 \\
0.34 & 0.41 \\
\end{pmatrix}.
\]$$

As we can see, the predictions computed by the FCM_MP model are reasonably close to the ground truth $\textbf{Y}$. If we want to quantify how the predictions differ from the ground truth, we can compute a performance metric for regression problems such as the Root Mean Square Error (RMSE), as shown below.

```python
rmse = np.sqrt(np.mean((Y-Y_hat)**2))
print(np.round(rmse, 4))
# RMSE=0.0088
```

## References

If you use the LSTCN model in your research please cite the following papers:

1.  Nápoles, G., Grau, I., Jastrzębska, A., & Salgueiro, Y. (2022). Long short-term cognitive networks. Neural Computing and Applications, 1-13. [paper](https://link.springer.com/article/10.1007/s00521-022-07348-5) [bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:tsqxxO4Ul0kJ:scholar.google.com/&output=citation&scisdr=CgXfrbsrEOqYxeaCl0s:AAGBfm0AAAAAY32Ej0sEhR2wzKa7dk6C4kVxUT3em6HS&scisig=AAGBfm0AAAAAY32Ej-1zPkScA5cUw8kSxfjYNDERIFe1&scisf=4&ct=citation&cd=-1&hl=en)

2.  Nápoles, G., Vanhoenshoven, F., & Vanhoof, K. (2019). Short-term cognitive networks, flexible reasoning and nonsynaptic learning. Neural Networks, 115, 72-81. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608019300930) [bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:WE6oovxx-9gJ:scholar.google.com/&output=citation&scisdr=CgXfrbsrEOqYxeaDEbk:AAGBfm0AAAAAY32FCbnEY_3UOTzV4qh2Jkjw8uWRKmkg&scisig=AAGBfm0AAAAAY32FCXb7V_h61rxVMwqW-tIpnRav5ps2&scisf=4&ct=citation&cd=-1&hl=en)

Some application papers with nice examples and further explanations:

3.  Morales-Hernández, A., Nápoles, G., Jastrzebska, A., Salgueiro, Y., & Vanhoof, K. (2022). Online learning of windmill time series using Long Short-term Cognitive Networks. Expert Systems with Applications, 117721. [paper](https://www.sciencedirect.com/science/article/pii/S0957417422010065) [bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:zw7eSIZeni8J:scholar.google.com/&output=citation&scisdr=CgXfrbsrEOqYxeaDLaY:AAGBfm0AAAAAY32FNaZ4Y4UCT9Pi0MyrcnXkbVr9ZKQK&scisig=AAGBfm0AAAAAY32FNS8iIT36tfp463gOvpckF52eUHpt&scisf=4&ct=citation&cd=-1&hl=en)

4.  Grau, I., de Hoop, M., Glaser, A., Nápoles, G., & Dijkman, R. (2022). Semiconductor Demand Forecasting using Long Short-term Cognitive Networks. In Proceedings of the 34th Benelux Conference on Artificial Intelligence and 31st Belgian-Dutch Conference on Machine Learning, BNAIC/BeNeLearn 2022. [paper](https://bnaic2022.uantwerpen.be/wp-content/uploads/BNAICBeNeLearn_2022_submission_4148.pdf) [bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:d8vQmLWkfxoJ:scholar.google.com/&output=citation&scisdr=CgXfrbsrEOqYxeaDRPY:AAGBfm0AAAAAY32FXPaTi5GsMnukoQWrf0Om83a-J6W6&scisig=AAGBfm0AAAAAY32FXC9uZn6HZlt2vf6hQPhocM_e53y2&scisf=4&ct=citation&cd=-1&hl=en)

This following paper introduces the dataset used in the [example](https://github.com/gnapoles/lstcn/blob/main/examples):

5.  Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 12, pp. 11106-11115). [paper](https://ojs.aaai.org/index.php/AAAI/article/view/17325) [bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:VigqltkXN1QJ:scholar.google.com/&output=citation&scisdr=CgXfrbsrEPaI6B_rEX0:AAGBfm0AAAAAY4PtCX1Nw5q5QeLUSLGippUjAq9GB4dc&scisig=AAGBfm0AAAAAY4PtCStnIoFzxpsoHCwvNsGMCy_qkZUZ&scisf=4&ct=citation&cd=-1&hl=en)
