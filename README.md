# A customisable fully connected neural network

```
from network import Model

mlp = Model()
mlp.add_layer(784, 200, act_fn='sigmoid')
mlp.add_layer(200, 128, act_fn='sigmoid')
mlp.add_layer(128, 64, act_fn='sigmoid')
mlp.add_layer(64, 10, act_fn='softmax')

mlp.fit(x_train, y_train,  epochs=10)
```

```
epoch  0 loss =  0.7671066487120393 acc =  0.8809259259259259
epoch  1 loss =  0.4219997210768396 acc =  0.9203703703703704
epoch  2 loss =  0.31908042027747724 acc =  0.9366666666666666
epoch  3 loss =  0.26321430571995635 acc =  0.9472222222222222
epoch  4 loss =  0.2271466179570256 acc =  0.9533333333333334
epoch  5 loss =  0.20133359756502975 acc =  0.9577777777777777
epoch  6 loss =  0.18176775551859337 acc =  0.9618518518518518
epoch  7 loss =  0.1666178338128933 acc =  0.9662962962962963
epoch  8 loss =  0.15475931934489515 acc =  0.9688888888888889
epoch  9 loss =  0.14591911157507992 acc =  0.9718518518518519
epoch  10 loss =  0.13975035486280465 acc =  0.9724074074074074
```

This uses [cupy](https://cupy.dev/) instead of `numpy` if available to have GPU accelerated computations.

An example file demonstrating classification of Iris and MNIST data is included.