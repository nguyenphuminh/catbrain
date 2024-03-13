const mnist = require("mnist");
const { CatBrain } = require("../../dist/core");
const fs = require("fs");

// Generate the dataset
const set = mnist.set(0, 1);
// Normalized sample
const sample = set.test[0].input.map(el => Math.round(el));
// Load the model
const neuralNetwork = new CatBrain(JSON.parse(fs.readFileSync("./model.json")));
// Log out the image
logImage(sample);
// Test it out :)
console.log(`\nNumber: ${indexOfMaxValue(neuralNetwork.feedForward(sample))}`);

// Function to log out the image
function logImage(dataObject) {
    for (let i = 0; i < dataObject.length; i++) {
        process.stdout.write((dataObject[i] ? "0" : "1") + " ");

        if ((i + 1) % 28 === 0) {
            console.log("");
        }
    }
}

// Function to log out the result
function indexOfMaxValue(output) {
    let maxIndex = 0;

    for (let i = 0; i < output.length; i++) {
        if (output[maxIndex] < output[i]) {
            maxIndex = i;
        }
    }

    return maxIndex;
}
