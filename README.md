# CatBrain

Neural networks made simple for Javascript

## Setup

Install through npm:
```
npm install catbrain
```

## Tutorial

Here is how to create, train, and run a neural net using Catbrain. All the options and config are shown as comments.
```js
const { CatBrain } = require("catbrain");

// Create a neural network
const neuralNetwork = new CatBrain({
    // Init layers with their size, the first and last are input and output layers
    layers: [2, 3, 1],

    // Optional config
    learningRate: 0.02, // Learning rate, default is 0.01
    decayRate: 0.9999, // Learning decay rate for each iteration, default is 1
    shuffle: true, // Choose whether to shuffle the dataset, default is true
    activation: "sigmoid", // sigmoid/tanh/relu/leakyRelu/swish/softplus/linear, default is sigmoid
    outputActivation: "sigmoid", // Activation at output layer, default is sigmoid
    leakyReluAlpha: 0.01, // Alpha of leaky relu if you use it, default is 0.01
    reluClip: 5, // Relu clipping, applied in activation functions reaching infinity, default is 5

    // Options to load existing models, randomly initialized if not provided
    // weights: number[][][],
    // biases: number[][]
});

// Train
neuralNetwork.train(
    // Amount of iterations
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
    //     // A function called before every iteration
    //     callback: (status) => {
    //         console.log(status.iteration)
    //     }
    // }
);

// Run the neural net with our own input
console.log(neuralNetwork.feedForward([1, 0]));
```

## Examples

There are several demos available in `./examples`:
* [MNIST digit recognition](https://github.com/nguyenphuminh/catbrain/tree/main/examples/mnist)
* [Tictactoe winner guesser](https://github.com/nguyenphuminh/catbrain/blob/main/examples/tictactoe.js)

## Todos

Currently what I have in mind are:

* Option to configure each layer independently.
* Code refactoring and optimization.
* More activation functions.
* GPU acceleration.
* More neural network architectures.
* Minor utilities for convenience.

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the GPL 3.0 License.
