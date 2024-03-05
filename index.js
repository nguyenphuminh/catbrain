const { CatBrain } = require("./dist/core");

// Test the neural network
const neuralNetwork = new CatBrain(2, 6, 4);
neuralNetwork.train(10000, [
    { inputs: [ -0.5, -0.5 ], outputs: [ 1, 0, 0, 0 ] },
    { inputs: [ 0.5, -0.5 ], outputs: [ 0, 1, 0, 0 ] },
    { inputs: [ -0.5, 0.5 ], outputs: [ 0, 0, 1, 0 ] },
    { inputs: [ 0.5, 0.5 ], outputs: [ 0, 0, 0, 1 ] }
]);

console.log(neuralNetwork.feedForward([ 0.2, 0.3 ]));
