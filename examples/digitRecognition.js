const { CatBrain } = require("../dist/core");
const dataset = require("./datasets/digitalRecognition");

// Detect a digit from a 7x7 image, dataset is extremely small and hand-written,
// so this will probably be inaccurate in a lot of cases, but eh it kind of works
const neuralNetwork = new CatBrain({
    inputAmount: 49, 
    hiddenAmounts: [ 18, 18 ], 
    outputAmount: 10,
    learningRate: 0.001
});

neuralNetwork.train(200000, dataset);

const result = neuralNetwork.feedForward([
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 0,
    0, 0, 1, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 1, 0,
    0, 0, 1, 0, 1, 0, 0,
    0, 0, 0, 1, 0, 0, 0
]);

/*
const result = neuralNetwork.feedForward([
    0, 0, 0, 1, 1, 0, 0,
    0, 0, 1, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 1, 1, 0, 0,
    0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 1, 0, 0, 0
]);*/

console.log(result);

console.log(indexOfMaxValue(result));

function indexOfMaxValue(array) {
    let maxValue = array[0];
    let maxIndex = 0;

    for (let index = 1; index < array.length; index++) {
        if (maxValue < array[index]) {
            maxValue = array[index];
            maxIndex = index;
        }
    }

    return maxIndex;
}
