# CatBrain

An attempt to create a neural network framework

## Setup

Install through npm:
```
npm install catbrain
```

## Use

Create a Javascript file like this:
```js
const { CatBrain } = require("catbrain");

// Create a neural network
const neuralNetwork = new CatBrain({
    inputAmount: 2, // Amount of input nodes
    hiddenAmounts: [3], // Amount of nodes for each hidden layer
    outputAmount: 1, // Amount of output nodes
    learningRate: 0.02 // Learning rate
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
