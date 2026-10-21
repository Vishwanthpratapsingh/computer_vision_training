# Anatomy of the Training Loop

> DRAFT: read every paragraph, check it against your own `train.py`, and rewrite it in your
> own words before you push. Replace the numbers in section 7 with your real results.

Training a network means repeating four steps on batches of data: forward, loss, backward,
step. In `train.py` they are the four marked lines inside `run_epoch`.

## 1. Data and normalization
A batch of CIFAR-10 is a tensor of shape (128, 3, 32, 32): 128 images, 3 color channels,
32x32 pixels. I subtract each channel's mean and divide by its standard deviation, so inputs
are centered near 0 with a spread near 1. Without this, raw pixel values (0-255 or 0-1) are
all positive and on a different scale per channel, which makes gradients lopsided and
training slower and less stable. Normalized inputs let one learning rate work well for all
weights, so the loss falls faster.

## 2. Forward pass
The batch goes through the layers in order. In the CNN, convolution filters slide over the
image and produce feature maps (edges first, then shapes), ReLU adds non-linearity, pooling
shrinks the maps, and a final linear layer outputs 10 numbers per image. These numbers are
logits: raw scores, not probabilities, because they can be negative and do not sum to 1.

## 3. Loss
Cross-entropy turns the logits into probabilities (softmax) and measures how much probability
the model put on the correct class. If the model is confident and right, the loss is near 0.
If it is confident and wrong, the loss is large. This single number says how bad the current
weights are, and it is what we try to reduce.

## 4. Backward pass (backprop)
`loss.backward()` uses the chain rule to compute the gradient of the loss with respect to every
weight: how much the loss would change if that weight changed slightly. Autograd records the
operations of the forward pass, then walks backwards from the loss, multiplying local
gradients layer by layer until it reaches the first layer. The gradients are stored in each
parameter's `.grad`. Because PyTorch adds new gradients to the old ones, I call
`opt.zero_grad()` before each backward pass.

## 5. Optimizer step
`opt.step()` moves each weight a small amount in the direction that reduces the loss. For
plain SGD the rule is `w = w - lr * w.grad`. I use Adam, which also keeps running averages of
the gradients to adapt the step size per weight. The learning rate controls the step size: too
high and the loss jumps around or diverges, too low and training is very slow.

## 6. Train, validation and test
I train on 45,000 images and hold out 5,000 for validation. Validation shows how the model does
on images it did not train on, which I use to tune the learning rate and batch size. The test
set is touched only once at the end for the final accuracy. If training accuracy keeps rising
while validation accuracy stalls, the model is overfitting. `model.train()` turns on dropout
and lets BatchNorm use batch statistics. `model.eval()` turns dropout off and uses the learned
statistics, so predictions are stable.

## 7. What I observed
- MLP: test accuracy ___ (a plain MLP ignores image structure, so it plateaus low)
- CNN: test accuracy ___ (convolutions share weights across the image, so it does much better)
- What I tried (learning rate, batch size, augmentation) and what changed: ___
- Overfitting: ___ (describe the gap between train and validation curves)
- W&B run: ___
