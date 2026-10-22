# NN-Interp

Using `pytorch`, I train a model to do multiplication of two floats on [-2, 2].

The models layers have lengths 2 (inputs), 4 (hidden), 2 (hidden), 1 (output)

The four cases are when both input floats are on [-1, 1], are on the union of [-2, -1] and [1, 2], or when one of the input floats is 0.

Then, I try to understand the hidden activations of the model by just printing them out and seeing if I can find human patterns. (for the 4 cases, maybe 5 examples each)

For a more reliable method, I train a separate interpreting model to predict the outputs based on the first hidden layer: 4 (inputs -- the hidden layer), 1 (output). I see how the separate interpreting model's predictions change for the four cases.

I do the same for the second hidden layer, with a 2-input, 1-output interpreting model.

I plot the manifold that is this neural network (the NN should resemble z (output) = x (input) * y (the other input))

Run `main.ipynb` from top to bottom with PyTorch and Matplotlib installed. The model has 25 trainable parameters and uses tanh hidden activations. Adam training is followed by L-BFGS refinement, targeting held-out MSE below 0.001.

The notebook plots the learned surface at several training stages, with held-out validation MSE beside each plot. An independent final test set is used to report final accuracy. Training can take a few minutes on a CPU.

Linear probes include training/validation MSE curves, surfaces across probe training, PCA views of the first layer's four activations, and the original per-example case tables. All aggregate errors are explicitly labeled as mean squared error.
