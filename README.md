# CatBrain

An experimental neural network framework

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
    // Required
    inputAmount: 2, // Amount of input nodes
    hiddenAmounts: [3], // Amount of nodes for each hidden layer
    outputAmount: 1, // Amount of output nodes

    // Optional config
    learningRate: 0.02, // Learning rate, default is 0.01
    decayRate: 0.9999, // Learning decay rate for each iteration, default is 1
    shuffle: true, // Choose whether to shuffle the dataset, default is true
    activation: "sigmoid", // sigmoid/tanh/relu/leakyRelu, default is sigmoid
    leakyReluAlpha: 0.01, // Alpha of leaky relu if you use it, default is 0.01
    reluClip: 5, // Relu clipping, default is 5

    // Options to load existing models, randomly initialized if not provided
    // hiddenWeights: number[][][],
    // hiddenBiases: number[][],
    // outputWeights: number[][],
    // outputBias: number[]
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
    // You can also pass in optional training options as well:
    // , {
    //     learningRate: 0.02, // Will use original learning rate if not provided
    //     decayRate: 0.9999, // Will use original decay rate if not provided
    // }
);

// Run the neural net with our own input
console.log(neuralNetwork.feedForward([1, 0]));
```

## Examples

There are several examples available in `./examples`:
* [MNIST digit recognition](./examples/mnist)
* [Tictactoe winner guesser](./examples/tictactoe.js)

## Todos

Currently what I have in mind are:

* Code refactoring.
* More activation functions.
* Training with GPU.
* More neural network architectures.

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the GPL 3.0 License.
