# CatBrain

An attempt to create a neural network framework

## Setup

Clone the repository:
```sh
git clone https://github.com/nguyenphuminh/catbrain.git
```

Install the dependencies:
```
npm i
```

## Use

Create a Javascript file like this:
```js
const { CatBrain } = require("./index");

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
        // A data object with expected output of an input 
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

Current what I have in my mind are:

* Refactor code.
* More activation functions.
* Train with GPU.
* More neural network models?

## Copyrights and License

Copyrights © 2024 Nguyen Phu Minh.

This project is licensed under the GPL 3.0 License.
