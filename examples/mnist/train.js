const mnist = require("mnist");
const { CatBrain } = require("../../dist/core");
const fs = require("fs");

// Generate the dataset
const set = mnist.set(60000, 1000);
// Training set
const trainingSet = set.training;
// Test net
const testSet = set.test;

// Create a CatBrain instance
const neuralNetwork = new CatBrain({
    inputAmount: 784, 
    hiddenAmounts: [ 256, 256, 256 ], 
    outputAmount: 10,
    learningRate: 0.001
});

// Train
const start = Date.now();
console.log("Training...");
neuralNetwork.train(300000, normalizeSet(trainingSet));
console.log(`Training ended in ${Date.now() - start}ms`);

// Calculate accuracy
console.log(`Accuracy: ${calculateAccuracy(testSet) * 100}%`);

// Export the model for use in the future
delete neuralNetwork["hiddenLayers"];
fs.writeFileSync("./newModel.json", JSON.stringify(neuralNetwork));
console.log("Model exported to \"newModel.json\"");

function normalize(dataObject) {
    const input = dataObject.input || dataObject.inputs;
    const output = dataObject.output || dataObject.outputs;

    return {
        inputs: input.map(el => Math.round(el)),
        outputs: output
    }
}

function normalizeSet(dataSet) {
    return dataSet.map(dataObject => {
        return normalize(dataObject);
    });
}

function calculateAccuracy(testSet) {
    return testSet.filter(testObject => {
        const expectedOutput = JSON.stringify(testObject.output);
        const actualOutput = JSON.stringify(neuralNetwork.feedForward(testObject.input).map(el => Math.round(el)));

        return expectedOutput === actualOutput;
    }).length / testSet.length;
}
