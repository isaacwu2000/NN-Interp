# NN-Interp

Train a PyTorch network to multiply two independent floats sampled uniformly from [-2, 2], then investigate what each hidden neuron retains about the inputs.

The model is **2 → 4 → 1**, with one tanh hidden layer, a linear output, and **17 trainable parameters**. Adam runs for 12,000 steps, followed by L-BFGS refinement. Both optimizers use a **full batch of 4,096 training examples** per loss evaluation. Validation uses 4,096 separate examples; the independent final test set contains 16,384 examples.

With seed 42, the single-hidden-layer model reaches validation MSE **0.0002811 after Adam** and **6.20 × 10⁻⁸ after L-BFGS**. Final test MSE is **6.40 × 10⁻⁸**, below the **0.001** target (test MAE: 0.000189). These are average errors on sampled inputs, not worst-case guarantees.

Run [main.ipynb](main.ipynb) from top to bottom with PyTorch and Matplotlib installed. The notebook includes training curves, multiplication surfaces at several training stages, and five activation examples for each diagnostic case: both inputs in [-1, 1], both in [-2, -1] ∪ [1, 2], x = 0, and y = 0. Mixed inner/outer inputs are included in overall training and testing.

The former linear output probes are replaced by **four inverse probes**, one per hidden neuron. Each is a nonlinear **1 → 32 → 32 → 2** tanh decoder that takes only that neuron's scalar activation z and predicts the original input pair. The multiplication network stays frozen. Each probe trains on 4,096 fresh examples for 3,000 full-batch Adam updates, with a separate 4,096-example validation set and training-only activation standardization.

Input order does not matter. For each example, the probe loss chooses the smaller MSE between matching its two outputs to (x, y) or (y, x), then averages those minima across the batch. This same unordered MSE is used for training, validation, and testing. Probe reconstruction errors measure input recovery, separately from the network's multiplication error.

| Hidden neuron | Test unordered input MSE |
| --- | ---: |
| 1 | 0.655792 |
| 2 | 0.174974 |
| 3 | 0.175070 |
| 4 | 0.655611 |

Each probe has learning curves, held-out case evaluations, and a **3D plot (z, predicted x, predicted y)**. Each held-out example shows only the direct or swapped predicted pair that wins the unordered loss against its original inputs. Orange points show those predictions; opaque dark-blue points show the actual pairs. Choosing the winning coordinate assignment uses the known target, so the plot shows evaluation points rather than a continuous decoder curve. Both output coordinates contribute to the loss. Five-example tables still sort pairs for readability. A single scalar generally cannot recover two independent inputs exactly, so low multiplication MSE does not imply low inverse-probe MSE.
