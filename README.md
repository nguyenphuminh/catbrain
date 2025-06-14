# CatBrain

GPU accelerated neural networks made simple for Javascript, influenced by [Brain.js](https://github.com/BrainJS/brain.js).

## Setup

Install through npm:
```
npm install catbrain
```

Note: Be sure to have **Node18 and Python 3.9 specifically** already installed or else gpu.js and its deps can not be installed. If it still breaks, try:
```sh
export npm_config_python=/path/to/executable/python
```

or if you are on Windows:
```bat
set NODE_GYP_FORCE_PYTHON=/path/to/executable/python.exe
```

## Tutorial

Here is how to create, train, and run a neural net using CatBrain. All the options and config are shown as comments.
```js
const { CatBrain } = require("catbrain");

// Create a neural network
const neuralNetwork = new CatBrain({
    // Init layers with their size, the first and last are input and output layers
    layers: [2, 3, 1],

    // Optional

    // Training config
    learningRate: 0.02, // Learning rate, default is 0.01
    decayRate: 0.9999, // Learning decay rate for each iteration, default is 1
    shuffle: true, // Choose whether to shuffle the dataset, default is true

    // Momentum optimizer
    momentum: 0.2, // Momentum constant, default is 0.1
    dampening: 0.2, // Momentum dampening, default is 0.1
    nesterov: true, // Enable Nesterov Accelerated Gradient, default is false
    
    // Activation config
    activation: "relu", // sigmoid/tanh/relu/leakyRelu/swish/mish/softplus/linear, default is relu
    outputActivation: "sigmoid", // Activation at output layer, default is sigmoid
    leakyReluAlpha: 0.01, // Alpha of leaky relu if you use it, default is 0.01
    reluClip: 5, // Relu clipping, applied in activation functions reaching infinity, default is 5
    // Weight init function, default depends on what activation is used (check ./src/rand.ts)
    // Options: xavierUniform, xavierNormal, heUniform, heNormal, lecunUniform, lecunNormal
    // There is also "basicUniform" which initializes with random numbers from 0 to 1
    weightInit: "heNormal",

    // Options to load existing models, randomly initialized depends on activation if not provided
    // Though, do note that biases are initialized as 0
    // weights: ArrayLike<number>[][],
    // biases: ArrayLike<number>[],
    // deltas: ArrayLike<number>[][], // Load this if you want to resume training

    // gpu.js options, this will be passed to the GPU constructor
    // gpuOptions: {}
    // Do note that this is heavily in-dev and not recommended for use at all currently
});

// Train
neuralNetwork.train(
    // Number of iterations
    100000,
    // Dataset as an array
    [
        // A data object with expected outputs of inputs 
        { inputs: [0, 0], outputs: [0] },
        { inputs: [0, 1], outputs: [1] },
        { inputs: [1, 0], outputs: [1] },
        { inputs: [1, 1], outputs: [0] }
    ]
    // You can also pass in optional training config as well:
    // , {
    //     learningRate: 0.02, // Will use original learning rate if not provided
    //     decayRate: 0.9999, // Will use original decay rate if not provided
    //     momentum: 0.1, // Will use original momentum if not provided
    //     dampening: 0.1, // Will use original dampening if not provided
    //     nesterov: 0.1, // Will use original nesterov if not provided
    //     shuffle: true, // Will use original shuffle option if not provided
    //     // A function called before every iteration
    //     callback: (status) => {
    //         console.log(status.iteration)
    //     },
    //     enableGPU: true // Default is false if not specified
    // }
);

// Run the neural net with our own input
console.log(neuralNetwork.feedForward([1, 0]));
// You can run it with GPU enabled:
// console.log(neuralNetwork.feedForward([1, 0], { enableGPU: true }));

// Export model to JSON
// console.log(neuralNetwork.toJSON());
```

## Examples

There are several demos available in `./examples`:
* [MNIST digit recognition](https://github.com/nguyenphuminh/catbrain/tree/main/examples/mnist)
* [Tictactoe winner guesser](https://github.com/nguyenphuminh/catbrain/blob/main/examples/tictactoe.js)

## Todos

Currently what I have in mind are:

* Option to configure each layer independently to do more than just MLPs.
* Proper GPU acceleration, possibly with CUDA/ROCm and Node C++ bindings.
* More GD opimizers or different optimization algos.
* More pre-built neural network architectures.
* Code refactoring, test cases, and optimization.
* Minor utilities for convenience.
* More activation functions.

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the GPL 3.0 License.
