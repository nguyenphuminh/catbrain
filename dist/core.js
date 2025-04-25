"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CatBrain = void 0;
const activation_1 = require("./activation");
const utils_1 = require("./utils");
class CatBrain {
    layers;
    layerValues;
    weights;
    biases;
    errors;
    learningRate;
    decayRate;
    shuffle;
    activationOptions;
    activation;
    derivative;
    outputActivation;
    outputDerivative;
    constructor(options) {
        // Training configuration
        this.learningRate = options.learningRate || 0.01;
        this.decayRate = options.decayRate || 1;
        this.shuffle = options.shuffle ?? true;
        // Activation function configuration
        this.activationOptions = {
            leakyReluAlpha: options.leakyReluAlpha || 0.01,
            reluClip: options.reluClip || 5
        };
        const activation = options.activation || "sigmoid";
        this.activation = activation_1.Activation[activation] || activation_1.Activation.sigmoid;
        const derivativeMethod = activation_1.Activation[activation + "Derivative"] || activation_1.Activation.sigmoidDerivative;
        this.derivative = (preActValue, actValue) => {
            if (activation === "sigmoid" || activation === "tanh") {
                return derivativeMethod(actValue);
            }
            return derivativeMethod(preActValue, this.activationOptions);
        };
        const outputActivation = options.outputActivation || "sigmoid";
        this.outputActivation = activation_1.Activation[outputActivation] || activation_1.Activation.sigmoid;
        const outputDerivativeMethod = activation_1.Activation[outputActivation + "Derivative"] || activation_1.Activation.sigmoidDerivative;
        this.outputDerivative = (preActValue, actValue) => {
            if (outputActivation === "sigmoid" || outputActivation === "tanh") {
                return outputDerivativeMethod(actValue);
            }
            return outputDerivativeMethod(preActValue, this.activationOptions);
        };
        // Model configuration
        this.layers = options.layers;
        // Init layers with the configured size and set them to 0s at first
        this.layerValues = this.layers.map(layerSize => new Array(layerSize).fill(0));
        // Init a list of randomized weights for each node of each layer
        this.weights = options.weights || Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            return Array.from({ length: this.layers[layerIndex] }, () => {
                // Amount of weights of a node is the number of nodes of the previous layer
                return Array.from({ length: this.layers[layerIndex - 1] }, () => Math.random() * 2 - 1);
            });
        });
        // Init a list of biases for each node of each layer
        this.biases = options.biases || Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            return new Array(this.layers[layerIndex]).fill(0);
        });
        // Errors cache
        this.errors = Array.from({ length: this.layers.length }, (layer, layerIndex) => new Array(this.layers[layerIndex]).fill(0));
    }
    feedForward(inputs, getPreActivation) {
        const preActLayers = [];
        // Feed new inputs to our first (input) layer
        this.layerValues[0] = inputs;
        // Propagate hidden layers with layers behind them
        for (let index = 1; index < this.layers.length; index++) {
            // Get sum
            this.weighedSum(this.layerValues[index], this.weights[index], this.biases[index], this.layerValues[index - 1]);
            // Push a copy
            if (getPreActivation)
                preActLayers.push([...this.layerValues[index]]);
            // Activate
            this.activateLayer(this.layerValues[index], index === this.layers.length - 1);
        }
        const output = this.layerValues[this.layerValues.length - 1];
        if (getPreActivation)
            return [output, preActLayers];
        return output;
    }
    backPropagate(inputs, target, options) {
        const trainingOptions = {
            learningRate: options?.learningRate || this.learningRate
        };
        const [output, preActLayers] = this.feedForward(inputs, true);
        for (let layer = this.layers.length - 1; layer >= 1; layer--) {
            const derivative = layer === this.layers.length - 1 ? this.outputDerivative : this.derivative;
            for (let nodeIndex = 0; nodeIndex < this.layers[layer]; nodeIndex++) {
                // Wipe error
                this.errors[layer][nodeIndex] = 0;
                // Output layer error
                if (layer === this.layers.length - 1) {
                    this.errors[layer][nodeIndex] = target[nodeIndex] - output[nodeIndex];
                }
                // Hidden layer error
                else {
                    for (let nextNodeIndex = 0; nextNodeIndex < this.layers[layer + 1]; nextNodeIndex++) {
                        this.errors[layer][nodeIndex] +=
                            this.weights[layer + 1][nextNodeIndex][nodeIndex] *
                                this.errors[layer + 1][nextNodeIndex];
                    }
                }
                // Update weights for each node
                for (let prevNodeIndex = 0; prevNodeIndex < this.layers[layer - 1]; prevNodeIndex++) {
                    this.weights[layer][nodeIndex][prevNodeIndex] +=
                        trainingOptions.learningRate *
                            this.errors[layer][nodeIndex] *
                            derivative(preActLayers[layer - 1][nodeIndex], this.layerValues[layer][nodeIndex]) *
                            this.layerValues[layer - 1][prevNodeIndex];
                }
                // Update bias for each node
                this.biases[layer][nodeIndex] += trainingOptions.learningRate * this.errors[layer][nodeIndex];
            }
        }
    }
    train(iterations, trainingData, options) {
        const trainingOptions = {
            learningRate: options?.learningRate || this.learningRate,
            decayRate: options?.decayRate || this.decayRate
        };
        // Shuffle the dataset first
        if (this.shuffle)
            (0, utils_1.shuffle)(trainingData);
        let dataObjectIndex = 0;
        for (let iteration = 0; iteration < iterations; iteration++) {
            if (typeof options?.callback === "function")
                options.callback({ iteration });
            const data = trainingData[dataObjectIndex];
            this.backPropagate(data.inputs, data.outputs, trainingOptions);
            // If we have gone through all of the dataset, reshuffle it and continue training
            if (dataObjectIndex === trainingData.length - 1) {
                if (this.shuffle)
                    (0, utils_1.shuffle)(trainingData);
            }
            // Move to the next data object, reset to the first if reached limit
            dataObjectIndex = (dataObjectIndex + 1) % trainingData.length;
            // Update the learning rate
            trainingOptions.learningRate *= trainingOptions.decayRate;
        }
    }
    /*//////////////////////////////////////////////////////////////
                                Utilities
    //////////////////////////////////////////////////////////////*/
    weighedSum(currentLayer, currentWeights, currentBiases, prevLayer) {
        for (let index = 0; index < currentLayer.length; index++) {
            // Add bias
            currentLayer[index] = currentBiases[index];
            // Get weighed sum
            for (let prevIndex = 0; prevIndex < prevLayer.length; prevIndex++) {
                const weight = currentWeights[index][prevIndex];
                const prevNode = prevLayer[prevIndex];
                currentLayer[index] += weight * prevNode;
            }
        }
    }
    activateLayer(currentLayer, isOutput) {
        for (let index = 0; index < currentLayer.length; index++) {
            // Activate
            if (isOutput) {
                // Always apply sigmoid in the output layer
                currentLayer[index] = this.outputActivation(currentLayer[index], this.activationOptions);
            }
            else {
                currentLayer[index] = this.activation(currentLayer[index], this.activationOptions);
            }
        }
    }
}
exports.CatBrain = CatBrain;
