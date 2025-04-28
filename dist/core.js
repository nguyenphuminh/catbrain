"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CatBrain = void 0;
const activation_1 = require("./activation");
const rand_1 = require("./rand");
const utils_1 = require("./utils");
class CatBrain {
    // Mostly for external use
    layers;
    weights;
    biases;
    weightInit;
    activation;
    outputActivation;
    leakyReluAlpha;
    reluClip;
    learningRate;
    decayRate;
    shuffle;
    // Mostly for internal use
    activationFunc;
    derivativeFunc;
    outputActivationFunc;
    outputDerivativeFunc;
    layerValues;
    preActLayerValues;
    errors;
    activationOptions;
    constructor(options) {
        // Training configuration
        this.learningRate = options.learningRate || 0.01;
        this.decayRate = options.decayRate || 1;
        this.shuffle = options.shuffle ?? true;
        // Activation function configuration
        this.leakyReluAlpha = options.leakyReluAlpha || 0.01;
        this.reluClip = options.reluClip || 5;
        this.activationOptions = {
            leakyReluAlpha: this.leakyReluAlpha,
            reluClip: this.reluClip
        };
        this.activation = options.activation || "relu";
        this.activationFunc = activation_1.Activation[this.activation] || activation_1.Activation.sigmoid;
        const derivativeMethod = activation_1.Activation[this.activation + "Derivative"] || activation_1.Activation.sigmoidDerivative;
        this.derivativeFunc = (preActValue, actValue) => {
            if (this.activation === "sigmoid" || this.activation === "tanh") {
                return derivativeMethod(actValue);
            }
            return derivativeMethod(preActValue, this.activationOptions);
        };
        this.outputActivation = options.outputActivation || "sigmoid";
        this.outputActivationFunc = activation_1.Activation[this.outputActivation] || activation_1.Activation.sigmoid;
        const outputDerivativeMethod = activation_1.Activation[this.outputActivation + "Derivative"] || activation_1.Activation.sigmoidDerivative;
        this.outputDerivativeFunc = (preActValue, actValue) => {
            if (this.outputActivation === "sigmoid" || this.outputActivation === "tanh") {
                return outputDerivativeMethod(actValue);
            }
            return outputDerivativeMethod(preActValue, this.activationOptions);
        };
        // Model configuration
        this.layers = options.layers;
        // Choose weight init function
        this.weightInit = options.weightInit || rand_1.weightInitWithAct[this.activation] || "basicUniform";
        const weightInit = rand_1.Rand[this.weightInit];
        // Init layers with the configured size and set them to 0s at first
        this.layerValues = this.layers.map(layerSize => new Array(layerSize).fill(0));
        // Init preactivation layers
        this.preActLayerValues = this.layers.map(layerSize => new Array(layerSize).fill(0));
        // Init a list of randomized weights for each node of each layer
        this.weights = options.weights || Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            const outSize = this.layers[layerIndex];
            return Array.from({ length: outSize }, () => {
                const inSize = this.layers[layerIndex - 1];
                // Amount of weights of a node is the number of nodes of the previous layer
                return Array.from({ length: inSize }, () => weightInit(inSize, outSize));
            });
        });
        // Init a list of biases for each node of each layer
        this.biases = options.biases || Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            return new Array(this.layers[layerIndex]).fill(0);
        });
        // Errors cache
        this.errors = Array.from({ length: this.layers.length }, (layer, layerIndex) => new Array(this.layers[layerIndex]).fill(0));
        // Input weights, biases and pre-act values are non-existent, this is me being lazy
        this.preActLayerValues[0] = null;
        this.weights[0] = null;
        this.biases[0] = null;
        this.errors[0] = null;
    }
    /*//////////////////////////////////////////////////////////////
                                User APIs
    //////////////////////////////////////////////////////////////*/
    feedForward(inputs) {
        // Feed new inputs to our first (input) layer
        this.layerValues[0] = inputs;
        // Propagate layers with layers behind them
        for (let index = 1; index < this.layerValues.length; index++) {
            const currentLayer = this.layerValues[index];
            const weights = this.weights[index];
            const biases = this.biases[index];
            const prevLayer = this.layerValues[index - 1];
            const isOutput = index === this.layers.length - 1;
            const preActCurrentLayer = this.preActLayerValues[index];
            for (let index = 0; index < currentLayer.length; index++) {
                // Add bias
                preActCurrentLayer[index] = biases[index];
                // Get weighed sum
                for (let prevIndex = 0; prevIndex < prevLayer.length; prevIndex++) {
                    const weight = weights[index][prevIndex];
                    const prevNode = prevLayer[prevIndex];
                    preActCurrentLayer[index] += weight * prevNode;
                }
                // Activate
                if (isOutput) {
                    currentLayer[index] = this.outputActivationFunc(preActCurrentLayer[index], this.activationOptions);
                }
                else {
                    currentLayer[index] = this.activationFunc(preActCurrentLayer[index], this.activationOptions);
                }
            }
        }
        return this.layerValues[this.layerValues.length - 1];
    }
    backPropagate(inputs, target, options) {
        const trainingOptions = {
            learningRate: options?.learningRate || this.learningRate
        };
        const output = this.feedForward(inputs);
        const lastLayer = this.layerValues.length - 1;
        for (let layer = lastLayer; layer >= 1; layer--) {
            for (let nodeIndex = 0; nodeIndex < this.layers[layer]; nodeIndex++) {
                // Calculate derivative ahead of time
                const preActNeuron = this.preActLayerValues[layer][nodeIndex]; // layer - 1 because this does not have pre-act input layer
                const actNeuron = this.layerValues[layer][nodeIndex];
                const derivative = layer === lastLayer ?
                    this.outputDerivativeFunc(preActNeuron, actNeuron) :
                    this.derivativeFunc(preActNeuron, actNeuron);
                // Calculate error
                this.errors[layer][nodeIndex] = 0;
                // Output layer error
                if (layer === lastLayer) {
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
                            derivative *
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
}
exports.CatBrain = CatBrain;
