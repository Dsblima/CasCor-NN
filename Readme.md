# Cascade Correlation Neural Network (under construction)

**This project uses Cascade Correlation for regression. Besides that, it is a adaption of https://github.com/AndreeaMusat/Cascade-Correlation-Neural-Network/. In the original project Cascade Correlation was used for classification.**

## Architecture

At first, the architecture consists only of input-output connections. Then, the following steps are performed repeatedly:
1. The network (more precisely, the weights of the output units) is trained until a criterion of convergence is satisfied.
2. A new hidden unit is added to the network. Its inputs are all the original inputs  and all the outputs of all the previously-introduced hidden units and it is trained to maximize the magnitude of the covariance between its value and the residual error observed at the output units (in this step we freeze all the connections trained during step 1). The hidden unit is chosen from a pool of candidates trained simultaneously. When the new hidden unit is added to the network, its input weight is frozen and only the output weight will be trained in step 1.
3. Go to step 1.


