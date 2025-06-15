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
    layers: [ 784, 256, 256, 256, 10 ], 
    learningRate: 0.001,
    nesterov: true
});

// Train
console.log("Training...");
const enableGPU = false;
const normalizedSet = normalizeSet(trainingSet);
const start = performance.now();
neuralNetwork.train(1000, normalizedSet, {
    enableGPU
});
console.log(`Training ended in ${performance.now() - start}ms`);

// Calculate accuracy
console.log("Testing...");
const startTest = performance.now();
console.log(`Accuracy: ${calculateAccuracy(testSet) * 100}%`);
console.log(`Testing ended in ${performance.now() - startTest}ms`);

// Export the model for use in the future
fs.writeFileSync("./newModel.json", neuralNetwork.toJSON());
console.log("Model exported to \"newModel.json\"");

function normalize(dataObject) {
    const input = dataObject.input || dataObject.inputs;
    const output = dataObject.output || dataObject.outputs;

    return {
        inputs: Float32Array.from(input.map(el => Math.round(el))),
        outputs: Float32Array.from(output)
    }
}

function normalizeSet(dataSet) {
    return dataSet.map(dataObject => {
        return normalize(dataObject);
    });
}

function calculateAccuracy(testSet) {
    return testSet.filter(testObject => {
        if (typeof testObject.input[0] === "undefined" || typeof testObject.output[0] === "undefined") return true;

        const input = Float32Array.from(testObject.input);
        const expectedOutput = Float32Array.from(testObject.output);
        const actualOutput = Float32Array.from(neuralNetwork.feedForward(input, { enableGPU }));

        let expectedLabel = expectedOutput.indexOf(1);

        let actualLabel = 0;
        let actualValue = actualOutput[0];

        for (let index = 1; index < actualOutput.length; index++) {
            if (actualValue < actualOutput[index]) {
                actualValue = actualOutput[index];
                actualLabel = index;
            }
        }

        return actualLabel === expectedLabel;
    }).length / testSet.length;
}
