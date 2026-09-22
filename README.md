# Neural Network in C++

A neural network built from scratch in C++ to recognize handwritten digits from the **MNIST dataset** (28x28 grayscale images, 60,000 training samples, 10,000 test samples) — no machine learning libraries, just raw matrix math and calculus.

The repo contains two versions, written a year apart in difficulty:

- **`nn.cpp`** — a simple prototype
- **`nnAdvanced.cpp`** — a more advanced, class-based rewrite

The advanced version reaches **92% accuracy** on the test set.

## `nn.cpp` — the prototype

A straightforward first implementation, with the network's shape and logic written directly into the code rather than made configurable:

- Number of layers and number of neurons per layer are hardcoded
- Forward propagation is written out explicitly for each of the 3 layers
- Backpropagation is also written out explicitly, layer by layer, rather than generalized into a loop or recursive routine

This version was mainly about getting a working network end to end before generalizing the design.

## `nnAdvanced.cpp` — the advanced version

A from-scratch rewrite with a proper class-based structure, focused on efficiency and accuracy:

- **Recursive backpropagation** using bottom-up dynamic programming, so intermediate gradient computations aren't repeated across layers — this was the most involved part of the project and took several days to get right
- **Multiple loss functions** implemented, selectable by uncommenting the one you want to use in the code
- **Adam optimizer**, added later, which noticeably improved accuracy over plain gradient descent
- **Xavier initialization** for the network's weights, rather than naive random initialization
- Class-based structure, making the network easier to extend than `nn.cpp`

## Results

The advanced model (`nnAdvanced.cpp`) achieves **92% accuracy** on the MNIST test set. Training progress (epoch vs. loss) is plotted in the `Training Epoch vs Loss Graph/` folder, and raw training logs are recorded in `log.txt` / `log1.txt`.

## Project structure

```
.
├── nn.cpp                          # Simple prototype: hardcoded layers, forward prop & backprop
├── nnAdvanced.cpp                  # Advanced, class-based version: recursive DP backprop, Adam, Xavier init
├── log.txt                         # Training log
├── log1.txt                        # Training log
└── Training Epoch vs Loss Graph/   # Plots of training loss over epochs
```

## Getting started

### Prerequisites

- A C++ compiler supporting C++11 or later (e.g. `g++`)
- The MNIST dataset (images + labels), in the format the code expects

### Build and run

```bash
git clone https://github.com/ruchirjoshi0405/Neural-Network-in-cpp.git
cd Neural-Network-in-cpp

# Prototype version
g++ -O2 -o nn nn.cpp
./nn

# Advanced version
g++ -O2 -o nnAdvanced nnAdvanced.cpp
./nnAdvanced
```

> To try a different loss function in `nnAdvanced.cpp`, comment out the current one and uncomment the loss function you want to use, then recompile.

## What this project was about

This was a hobby project focused on understanding neural networks at a low level — implementing forward propagation, backpropagation, weight initialization, and optimization by hand in C++, without relying on any deep learning framework.

## Tech stack

- C++

## Author

[ruchirjoshi0405](https://github.com/ruchirjoshi0405)
