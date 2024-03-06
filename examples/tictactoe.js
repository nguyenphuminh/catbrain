const { CatBrain } = require("../dist/core");

// Try to see who wins in a won position of a tictactoe game from a set of 32 games
const neuralNetwork = new CatBrain({
    inputAmount: 9, 
    hiddenAmounts: [6], 
    outputAmount: 1,
    learningRate: 0.01
});

// 1 for X, -1 for O, 0 for empty, the outputs is near 0 if O wins, near 1 if X wins
neuralNetwork.train(50000, [
    // 2 sides, 3 cases for horizontal, 2 examples for each case
    { inputs: [ 1, 1, 1, -1, -1, 0, 0, 0, 0 ], outputs: [1] },
    { inputs: [ 1, 1, 1, 0, -1, 0, 0, -1, 0 ], outputs: [1] },
    { inputs: [ 0, -1, 0, 1, 1, 1, 0, -1, 0 ], outputs: [1] },
    { inputs: [ 0, -1, 0, 1, 1, 1, 0, 0, -1 ], outputs: [1] },
    { inputs: [ 0, -1, 0, 0, -1, 0, 1, 1, 1 ], outputs: [1] },
    { inputs: [ 0, -1, 0, 0, 0, -1, 1, 1, 1 ], outputs: [1] },
    
    { inputs: [ -1, -1, -1, 1, 1, 0, 0, 0, 0 ], outputs: [0] },
    { inputs: [ -1, -1, -1, 0, 1, 0, 0, 1, 0 ], outputs: [0] },
    { inputs: [ 0, 1, 0, -1, -1, -1, 0, 1, 0 ], outputs: [0] },
    { inputs: [ 0, 1, 0, -1, -1, -1, 0, 0, 1 ], outputs: [0] },
    { inputs: [ 0, 1, 0, 0, 1, 0, -1, -1, -1 ], outputs: [0] },
    { inputs: [ 0, 1, 0, 0, 0, 1, -1, -1, -1 ], outputs: [0] },

    // 2 sides, 3 cases for vertical, 2 examples for each case
    { inputs: [ 1, -1, 0, 1, -1, 0, 1, 0, 0 ], outputs: [1] },
    { inputs: [ 1, 0, 0, 1, -1, 0, 1, 0, -1 ], outputs: [1] },
    { inputs: [ 0, 1, 0, -1, 1, 0, -1, 1, 0 ], outputs: [1] },
    { inputs: [ -1, 1, 0, 0, 1, 0, 0, 1, -1 ], outputs: [1] },
    { inputs: [ -1, 0, 1, 0, -1, 1, 0, 0, 1 ], outputs: [1] },
    { inputs: [ 0, 0, 1, -1, 0, 1, -1, 0, 1 ], outputs: [1] },

    { inputs: [ -1, 1, 0, -1, 1, 0, -1, 0, 0 ], outputs: [0] },
    { inputs: [ -1, 0, 0, -1, 1, 0, -1, 0, 1 ], outputs: [0] },
    { inputs: [ 0, -1, 0, 1, -1, 0, 1, -1, 0 ], outputs: [0] },
    { inputs: [ 1, -1, 0, 0, -1, 0, 0, -1, 1 ], outputs: [0] },
    { inputs: [ 1, 0, -1, 0, 1, -1, 0, 0, -1 ], outputs: [0] },
    { inputs: [ 0, 0, -1, 1, 0, -1, 1, 0, -1 ], outputs: [0] },

    // 2 sides, 2 cases for diagonal, 2 examples for each case
    { inputs: [ 1, 0, -1, 0, 1, 0, -1, 0, 1 ], outputs: [1] },
    { inputs: [ 1, -1, 0, 0, 1, 0, 0, -1, 1 ], outputs: [1] },
    { inputs: [ -1, 0, 1, 0, 1, 0, 1, 0, -1 ], outputs: [1] },
    { inputs: [ -1, -1, 1, 0, 1, 0, 1, 0, 0 ], outputs: [1] },

    { inputs: [ -1, 0, 1, 0, -1, 0, 1, 0, -1 ], outputs: [0] },
    { inputs: [ -1, 1, 0, 0, -1, 0, 0, 1, -1 ], outputs: [0] },
    { inputs: [ 1, 0, -1, 0, -1, 0, -1, 0, 1 ], outputs: [0] },
    { inputs: [ 1, 1, -1, 0, -1, 0, -1, 0, 0 ], outputs: [0] },
]);

// O won
console.log(neuralNetwork.feedForward([ 1, 1, -1, -1, -1, -1, 1, 0, 1 ]));
// X won
console.log(neuralNetwork.feedForward([ 1, 0, -1, 0, 1, 0, -1, 0, 1 ]));
