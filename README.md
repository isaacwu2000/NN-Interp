# NN-Interp

Using `pytorch`, I train a model to do multiplication of two floats on [-2, 2].

The models layers have lengths 2 (inputs), 4 (hidden), 4 (hidden), 1 (output)

The four cases are when both input floats are on [-1, 1], are on the union of [-2, -1] and [1, 2], or when one of the input floats is 0.

Then, I try to understand the hidden activations of the model by just printing them out and seeing if I can find human patterns. (for the 4 cases, maybe 5 examples each)

For a more reliable method, I train a separate interpreting model to predict the outputs based on the first hidden layer: 4 (inputs -- the hidden layer), 1 (output). I see how the separate interpreting model's predictions change for the four cases.

I do the same for the second hidden layer.

I plot the manifold that is this neural network (the NN should resemble z (output) = x (input) * y (the other input))