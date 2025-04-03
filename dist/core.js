"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CatBrain = void 0;
const activation_1 = require("./activation");
const utils_1 = require("./utils");
class CatBrain {
    inputAmount; // Amount of input nodes
    hiddenAmounts; // Amount of hidden nodes in each hidden layer
    outputAmount; // Amount of output nodes
    learningRate;
    shuffle; // Choose whether to shuffle dataset when training
    activation;
    hiddenLayers;
    hiddenWeights;
    hiddenBiases;
    outputWeights;
    outputBias;
    constructor(options) {
        // Basic configuration
        this.inputAmount = options.inputAmount;
        this.hiddenAmounts = options.hiddenAmounts;
        this.outputAmount = options.outputAmount;
        this.learningRate = options.learningRate || 0.01;
        this.shuffle = typeof options.shuffle === "boolean" ? options.shuffle : true;
        // Activation function configuration
        this.activation = new activation_1.Activation({
            leakyReluAlpha: options.leakyReluAlpha || 0.01,
            reluClip: options.reluClip || 5,
            activation: options.activation || "sigmoid"
        });
        // Model configuration
        // Init hidden layers with the configured amount of neurons and set them to 0 at first
        this.hiddenLayers = this.hiddenAmounts.map(hiddenAmount => new Array(hiddenAmount).fill(0));
        // Init a list of randomized weights for each node of each layer
        this.hiddenWeights = options.hiddenWeights || Array.from({ length: this.hiddenLayers.length }, (hiddenLayer, layerIndex) => {
            return Array.from({ length: this.hiddenAmounts[layerIndex] }, (hiddenNode, nodeIndex) => {
                // Amount of weights of a node is the number of nodes of the previous layer
                if (layerIndex === 0) {
                    return Array.from({ length: this.inputAmount }, () => Math.random() * 2 - 1);
                }
                else {
                    return Array.from({ length: this.hiddenAmounts[layerIndex - 1] }, () => Math.random() * 2 - 1);
                }
            });
        });
        // Init a list of biases for each node of each layer
        this.hiddenBiases = options.hiddenBiases || Array.from({ length: this.hiddenLayers.length }, (hiddenLayer, layerIndex) => {
            return new Array(this.hiddenAmounts[layerIndex]).fill(0);
        });
        // Output configuration
        // Init a list of weights targetting the node of that position in the output layer
        this.outputWeights = options.outputWeights || Array.from({ length: this.outputAmount }, () => Array.from({ length: this.hiddenAmounts[this.hiddenAmounts.length - 1] }, () => Math.random() * 2 - 1));
        // Init a list of bias for each output node
        this.outputBias = options.outputBias || new Array(this.outputAmount).fill(0);
    }
    feedForward(inputs) {
        // Propagate hidden layers with layers behind them
        for (let index = 0; index < this.hiddenLayers.length; index++) {
            this.weighedSum(this.hiddenLayers[index], this.hiddenWeights[index], this.hiddenBiases[index], index === 0 ? inputs : this.hiddenLayers[index - 1]);
        }
        // Propagate output from the last hidden layer
        const output = new Array(this.outputAmount);
        this.weighedSum(output, this.outputWeights, this.outputBias, this.hiddenLayers[this.hiddenLayers.length - 1], true);
        return output;
    }
    backPropagate(inputs, target) {
        const output = this.feedForward(inputs);
        const errorsOutput = new Array(this.outputAmount).fill(0);
        const errorsHidden = Array.from({ length: this.hiddenAmounts.length }, () => new Array(this.hiddenAmounts[0]).fill(0));
        // Backpropagate from output layer
        for (let nodeIndex = 0; nodeIndex < this.outputAmount; nodeIndex++) {
            // Calculate error from expected value
            errorsOutput[nodeIndex] = target[nodeIndex] - output[nodeIndex];
            // Update weights for each output node
            for (let prevNodeIndex = 0; prevNodeIndex < this.hiddenAmounts[this.hiddenAmounts.length - 1]; prevNodeIndex++) {
                this.outputWeights[nodeIndex][prevNodeIndex] +=
                    this.learningRate *
                        errorsOutput[nodeIndex] *
                        this.activation.sigmoidDerivative(output[nodeIndex]) * // Always apply sigmoid in the output layer
                        this.hiddenLayers[this.hiddenLayers.length - 1][prevNodeIndex];
            }
            // Update bias for each output node
            this.outputBias[nodeIndex] += this.learningRate * errorsOutput[nodeIndex];
        }
        // Backpropagate from hidden layer to hidden layer
        for (let hiddenLayer = this.hiddenLayers.length - 1; hiddenLayer >= 0; hiddenLayer--) {
            for (let nodeIndex = 0; nodeIndex < this.hiddenAmounts[hiddenLayer]; nodeIndex++) {
                // We will calculate the hidden errors first
                errorsHidden[hiddenLayer][nodeIndex] = 0;
                // If hidden layer is close to output, calculate hidden errors from output errors
                if (hiddenLayer === this.hiddenLayers.length - 1) {
                    for (let outputIndex = 0; outputIndex < this.outputAmount; outputIndex++) {
                        errorsHidden[hiddenLayer][nodeIndex] +=
                            this.outputWeights[outputIndex][nodeIndex] *
                                errorsOutput[outputIndex];
                    }
                    // Or else calculate from the next hidden layer's errors
                }
                else {
                    for (let nextNodeIndex = 0; nextNodeIndex < this.hiddenAmounts[hiddenLayer + 1]; nextNodeIndex++) {
                        errorsHidden[hiddenLayer][nodeIndex] +=
                            this.hiddenWeights[hiddenLayer + 1][nextNodeIndex][nodeIndex] *
                                errorsHidden[hiddenLayer + 1][nextNodeIndex];
                    }
                }
                // Update weights for each hidden node
                for (let prevNodeIndex = 0; prevNodeIndex < this.hiddenAmounts[hiddenLayer - 1]; prevNodeIndex++) {
                    this.hiddenWeights[hiddenLayer][nodeIndex][prevNodeIndex] +=
                        this.learningRate *
                            errorsHidden[hiddenLayer][nodeIndex] *
                            this.activation.derivative(this.hiddenLayers[hiddenLayer][nodeIndex]) *
                            this.hiddenLayers[hiddenLayer - 1][prevNodeIndex];
                }
                // Update bias for each hidden node
                this.hiddenBiases[hiddenLayer][nodeIndex] += this.learningRate * errorsHidden[hiddenLayer][nodeIndex];
            }
        }
        // Backpropagate from hidden layer to input layer
        for (let nodeIndex = 0; nodeIndex < this.hiddenAmounts[0]; nodeIndex++) {
            for (let prevNodeIndex = 0; prevNodeIndex < this.inputAmount; prevNodeIndex++) {
                this.hiddenWeights[0][nodeIndex][prevNodeIndex] +=
                    this.learningRate *
                        errorsHidden[0][nodeIndex] *
                        this.activation.derivative(this.hiddenLayers[0][nodeIndex]) *
                        inputs[prevNodeIndex];
            }
        }
    }
    train(iterations, trainingData) {
        let dataObjectIndex = 0;
        // Shuffle the dataset first
        if (this.shuffle)
            (0, utils_1.shuffle)(trainingData);
        for (let iteration = 0; iteration < iterations; iteration++) {
            const data = trainingData[dataObjectIndex];
            this.backPropagate(data.inputs, data.outputs);
            // If we have gone through all of the dataset, reshuffle it and continue training
            if (dataObjectIndex === trainingData.length - 1) {
                if (this.shuffle)
                    (0, utils_1.shuffle)(trainingData);
            }
            // Move to the next data object, reset to the first if reached limit
            dataObjectIndex = (dataObjectIndex + 1) % trainingData.length;
        }
    }
    weighedSum(currentLayer, currentWeights, currentBiases, prevLayer, isOutput) {
        for (let index = 0; index < currentLayer.length; index++) {
            currentLayer[index] = 0;
            // Get weighed sum
            for (let prevIndex = 0; prevIndex < prevLayer.length; prevIndex++) {
                const weight = currentWeights[index][prevIndex];
                const prevNode = prevLayer[prevIndex];
                currentLayer[index] += weight * prevNode;
            }
            // Add bias
            currentLayer[index] += currentBiases[index];
            // Activate
            if (isOutput) {
                // Always apply sigmoid in the output layer
                currentLayer[index] = this.activation.sigmoid(currentLayer[index]);
            }
            else {
                currentLayer[index] = this.activation.activate(currentLayer[index]);
            }
        }
    }
}
exports.CatBrain = CatBrain;
