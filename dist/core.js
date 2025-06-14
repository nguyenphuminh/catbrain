"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CatBrain = void 0;
const gpu_js_1 = require("gpu.js");
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
    momentum;
    dampening;
    nesterov;
    learningRate;
    decayRate;
    shuffle;
    gpuOptions;
    // Mostly for internal use
    activationFunc;
    derivativeFunc;
    outputActivationFunc;
    outputDerivativeFunc;
    layerValues;
    preActLayerValues;
    errors;
    deltas;
    kernels;
    gpu;
    constructor(options) {
        // Training configuration
        this.momentum = options.momentum || 0.1;
        this.dampening = options.dampening || 0.1;
        this.nesterov = options.nesterov ?? false;
        this.learningRate = options.learningRate || 0.01;
        this.decayRate = options.decayRate || 1;
        this.shuffle = options.shuffle ?? true;
        // Activation function configuration
        this.leakyReluAlpha = options.leakyReluAlpha || 0.01;
        this.reluClip = options.reluClip || 5;
        this.activation = options.activation || "relu";
        this.activationFunc = activation_1.Activation[this.activation] || activation_1.Activation.relu;
        this.derivativeFunc = activation_1.Activation[this.activation + "Derivative"] || activation_1.Activation.reluDerivative;
        this.outputActivation = options.outputActivation || "sigmoid";
        this.outputActivationFunc = activation_1.Activation[this.outputActivation] || activation_1.Activation.sigmoid;
        this.outputDerivativeFunc = activation_1.Activation[this.outputActivation + "Derivative"] || activation_1.Activation.sigmoidDerivative;
        // Model configuration
        this.layers = options.layers;
        // Choose weight init function
        this.weightInit = options.weightInit || rand_1.weightInitWithAct[this.activation] || "basicUniform";
        const weightInit = rand_1.Rand[this.weightInit];
        // Init layers with the configured size and set them to 0s at first
        this.layerValues = this.layers.map(layerSize => new Array(layerSize).fill(0));
        // Init preactivation layers
        this.preActLayerValues = Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            if (layerIndex === 0)
                return null;
            return new Array(this.layers[layerIndex]).fill(0);
        });
        // Init a list of randomized weights for each node of each layer
        this.weights = options.weights || Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            if (layerIndex === 0)
                return null;
            const outSize = this.layers[layerIndex];
            return Array.from({ length: outSize }, () => {
                const inSize = this.layers[layerIndex - 1];
                return Array.from({ length: inSize }, () => weightInit(inSize, outSize));
            });
        });
        // Init a list of biases for each node of each layer
        this.biases = options.biases || Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            if (layerIndex === 0)
                return null;
            return new Array(this.layers[layerIndex]).fill(0);
        });
        // Errors cache
        this.errors = Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            if (layerIndex === 0)
                return null;
            return new Array(this.layers[layerIndex]).fill(0);
        });
        // Deltas (velocity) for momentum
        this.deltas = options.weights || Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            if (layerIndex === 0)
                return null;
            return Array.from({ length: this.layers[layerIndex] }, () => {
                return Array.from({ length: this.layers[layerIndex - 1] }, () => 0);
            });
        });
        // Init GPU
        this.gpuOptions = options.gpuOptions || {};
        this.gpu = new gpu_js_1.GPU({ ...this.gpuOptions });
        // Init layers' kernels
        this.kernels = Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            if (layerIndex === 0)
                return null;
            return this.initKernels(this.layers[layerIndex], this.layers[layerIndex - 1], this.activationFunc, this.outputActivationFunc, this.derivativeFunc, this.outputDerivativeFunc);
        });
    }
    /*//////////////////////////////////////////////////////////////
                                User APIs
    //////////////////////////////////////////////////////////////*/
    feedForward(inputs, options) {
        const enableGPU = options?.enableGPU ?? false;
        // Feed new inputs to our first (input) layer
        this.layerValues[0] = inputs;
        // Propagate layers with layers behind them
        const layers = this.layerValues.length;
        for (let index = 1; index < layers; index++) {
            // Avoid lookups
            const currentLayer = this.layerValues[index];
            const currentLayerSize = currentLayer.length;
            const weights = this.weights[index];
            const biases = this.biases[index];
            const prevLayer = this.layerValues[index - 1];
            const prevlayerSize = prevLayer.length;
            const isOutput = index === this.layers.length - 1;
            if (enableGPU) {
                const { weightedSumAndActivate } = this.kernels[index];
                const { result, weightedSum } = weightedSumAndActivate(prevLayer, prevlayerSize, weights, biases, isOutput, this.reluClip, this.leakyReluAlpha);
                this.preActLayerValues[index] = Array.from(weightedSum);
                this.layerValues[index] = Array.from(result);
            }
            else {
                const preActCurrentLayer = this.preActLayerValues[index];
                for (let nodeIndex = 0; nodeIndex < currentLayerSize; nodeIndex++) {
                    // Avoid lookups
                    const nodeWeights = weights[nodeIndex];
                    // Add bias
                    preActCurrentLayer[nodeIndex] = biases[nodeIndex];
                    // Get weighed sum
                    for (let prevIndex = 0; prevIndex < prevlayerSize; prevIndex++) {
                        const weight = nodeWeights[prevIndex];
                        const prevNode = prevLayer[prevIndex];
                        preActCurrentLayer[nodeIndex] += weight * prevNode;
                    }
                    // Activate
                    if (isOutput) {
                        currentLayer[nodeIndex] = this.outputActivationFunc(preActCurrentLayer[nodeIndex], this.reluClip, this.leakyReluAlpha);
                    }
                    else {
                        currentLayer[nodeIndex] = this.activationFunc(preActCurrentLayer[nodeIndex], this.reluClip, this.leakyReluAlpha);
                    }
                }
            }
        }
        return this.layerValues[this.layerValues.length - 1];
    }
    backPropagate(inputs, target, options) {
        // Init
        const output = this.feedForward(inputs, options);
        const enableGPU = options?.enableGPU ?? false;
        const momentum = options?.momentum || this.momentum;
        const dampening = options?.dampening || this.dampening;
        const nesterov = options?.nesterov ?? this.nesterov;
        const learningRate = options?.learningRate || this.learningRate;
        const lastLayer = this.layerValues.length - 1;
        for (let layer = lastLayer; layer >= 1; layer--) {
            // Avoid lookups
            const nextLayerSize = this.layers[layer + 1];
            const nextLayerWeights = this.weights[layer + 1];
            const nextLayerErrors = this.errors[layer + 1];
            const preActLayerValues = this.preActLayerValues[layer];
            const layerSize = this.layers[layer];
            const layerWeights = this.weights[layer];
            const layerBiases = this.biases[layer];
            const layerDeltas = this.deltas[layer];
            const layerErrors = this.errors[layer];
            const prevLayerValues = this.layerValues[layer - 1];
            const prevLayerSize = this.layers[layer - 1];
            const isLastLayer = layer === lastLayer;
            if (enableGPU) {
                const { calculateErrors, calculateOutputErrors, updateWeights, addBiases } = this.kernels[layer];
                // Calculate errors
                if (isLastLayer) {
                    this.errors[layer] = Array.from(calculateOutputErrors(target, output));
                }
                else {
                    this.errors[layer] = Array.from(calculateErrors(nextLayerSize, nextLayerWeights, nextLayerErrors));
                }
                // Calculate deltas and update weights(
                const { calculateDeltas, result } = updateWeights(layerWeights, layerDeltas, this.errors[layer], preActLayerValues, prevLayerValues, isLastLayer, nesterov, learningRate, dampening, momentum, this.reluClip, this.leakyReluAlpha);
                this.deltas[layer] = calculateDeltas.map((nodeDeltas) => Array.from(nodeDeltas));
                this.weights[layer] = result.map((nodeWeights) => Array.from(nodeWeights));
                // Add biases
                this.biases[layer] = Array.from(addBiases(layerBiases, learningRate, this.errors[layer]));
            }
            else {
                // Calculate errors
                for (let nodeIndex = 0; nodeIndex < layerSize; nodeIndex++) {
                    // Calculate error
                    layerErrors[nodeIndex] = 0;
                    // Output layer error
                    if (isLastLayer) {
                        layerErrors[nodeIndex] = target[nodeIndex] - output[nodeIndex];
                    }
                    // Hidden layer error
                    else {
                        for (let nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++) {
                            layerErrors[nodeIndex] += nextLayerWeights[nextNodeIndex][nodeIndex] * nextLayerErrors[nextNodeIndex];
                        }
                    }
                    const nodeWeights = layerWeights[nodeIndex];
                    const nodeDeltas = layerDeltas[nodeIndex];
                    const nodeError = layerErrors[nodeIndex];
                    // Calculate derivative ahead of time
                    const derivative = isLastLayer ?
                        this.outputDerivativeFunc(preActLayerValues[nodeIndex], this.reluClip, this.leakyReluAlpha) :
                        this.derivativeFunc(preActLayerValues[nodeIndex], this.reluClip, this.leakyReluAlpha);
                    if (nesterov) {
                        for (let prevNodeIndex = 0; prevNodeIndex < prevLayerSize; prevNodeIndex++) {
                            const gradient = nodeError * derivative * prevLayerValues[prevNodeIndex];
                            const effectiveGradient = (1 - dampening) * gradient;
                            let delta = nodeDeltas[prevNodeIndex];
                            nodeDeltas[prevNodeIndex] = momentum * delta + effectiveGradient;
                            // Nesterov look-ahead
                            delta = momentum * delta + effectiveGradient;
                            nodeWeights[prevNodeIndex] += learningRate * delta;
                        }
                    }
                    else {
                        for (let prevNodeIndex = 0; prevNodeIndex < prevLayerSize; prevNodeIndex++) {
                            const gradient = nodeError * derivative * prevLayerValues[prevNodeIndex];
                            nodeDeltas[prevNodeIndex] = momentum * nodeDeltas[prevNodeIndex] + (1 - dampening) * gradient;
                            nodeWeights[prevNodeIndex] += learningRate * nodeDeltas[prevNodeIndex];
                        }
                    }
                    // Update bias for each node
                    layerBiases[nodeIndex] += learningRate * nodeError;
                }
            }
        }
    }
    train(iterations, trainingData, options) {
        const trainingOptions = {
            learningRate: options?.learningRate || this.learningRate,
            decayRate: options?.decayRate || this.decayRate,
            momentum: options?.momentum || this.momentum,
            dampening: options?.dampening || this.dampening,
            nesterov: options?.nesterov ?? this.nesterov,
            shuffle: options?.shuffle ?? this.shuffle,
            enableGPU: options?.enableGPU ?? false
        };
        // Shuffle the dataset first
        if (trainingOptions.shuffle)
            (0, utils_1.shuffle)(trainingData);
        let dataObjectIndex = 0;
        for (let iteration = 0; iteration < iterations; iteration++) {
            if (typeof options?.callback === "function")
                options.callback({ iteration });
            const data = trainingData[dataObjectIndex];
            this.backPropagate(data.inputs, data.outputs, trainingOptions);
            // If we have gone through all of the dataset, reshuffle it and continue training
            if (dataObjectIndex === trainingData.length - 1) {
                if (trainingOptions.shuffle)
                    (0, utils_1.shuffle)(trainingData);
            }
            // Move to the next data object, reset to the first if reached limit
            dataObjectIndex = (dataObjectIndex + 1) % trainingData.length;
            // Update the learning rate
            trainingOptions.learningRate *= trainingOptions.decayRate;
        }
    }
    initKernels(layerSize, prevLayerSize, activationFunc, outputActivationFunc, derivativeFunc, outputDerivativeFunc) {
        const actFuncSource = (0, utils_1.methodToFunc)(activationFunc, "activationFunc");
        const outputActFuncSource = (0, utils_1.methodToFunc)(outputActivationFunc, "outputActivationFunc");
        const derFuncSource = (0, utils_1.methodToFunc)(derivativeFunc, "derivativeFunc");
        const outputDerFuncSource = (0, utils_1.methodToFunc)(outputDerivativeFunc, "outputDerivativeFunc");
        function weightedSum(sum) { return sum; }
        function calculateDeltas(delta) { return delta; }
        return {
            weightedSumAndActivate: this.gpu.createKernelMap({
                weightedSum
            }, function (prevLayer, prevSize, weights, biases, isOutput, clip, alpha) {
                let sum = biases[this.thread.x];
                for (let index = 0; index < prevSize; index++) {
                    sum += prevLayer[index] * weights[this.thread.x][index];
                }
                weightedSum(sum);
                if (isOutput)
                    return outputActivationFunc(sum, clip, alpha);
                return activationFunc(sum, clip, alpha);
            })
                .setFunctions([
                {
                    source: actFuncSource,
                    settings: {}
                },
                {
                    source: outputActFuncSource,
                    settings: {}
                }
            ])
                .setOutput([layerSize]),
            calculateErrors: this.gpu.createKernel(function (nextLayerSize, nextLayerWeights, nextLayerErrors) {
                let errorSum = 0;
                for (let nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++) {
                    errorSum += nextLayerWeights[nextNodeIndex][this.thread.x] * nextLayerErrors[nextNodeIndex];
                }
                return errorSum;
            })
                .setOutput([layerSize]),
            calculateOutputErrors: this.gpu.createKernel(function (target, output) {
                return target[this.thread.x] - output[this.thread.x];
            })
                .setOutput([layerSize]),
            updateWeights: this.gpu.createKernelMap({
                calculateDeltas
            }, function (layerWeights, layerDeltas, layerErrors, preActLayerValues, prevLayerValues, isLastLayer, nesterov, learningRate, dampening, momentum, reluClip, leakyReluAlpha) {
                const derivative = isLastLayer ?
                    outputDerivativeFunc(preActLayerValues[this.thread.x], reluClip, leakyReluAlpha) :
                    derivativeFunc(preActLayerValues[this.thread.x], reluClip, leakyReluAlpha);
                const gradient = layerErrors[this.thread.y] * derivative * prevLayerValues[this.thread.x];
                const effectiveGradient = (1 - dampening) * gradient;
                let delta = momentum * layerDeltas[this.thread.y][this.thread.x] + effectiveGradient;
                calculateDeltas(delta);
                // Nesterov look-ahead
                if (nesterov) {
                    delta = momentum * delta + effectiveGradient;
                }
                return layerWeights[this.thread.y][this.thread.x] + learningRate * delta;
            })
                .setFunctions([
                {
                    source: derFuncSource,
                    settings: {}
                },
                {
                    source: outputDerFuncSource,
                    settings: {}
                }
            ])
                .setOutput([prevLayerSize, layerSize]),
            addBiases: this.gpu.createKernel(function (layerBiases, learningRate, nodeError) {
                return layerBiases[this.thread.x] + learningRate * nodeError[this.thread.x];
            })
                .setOutput([layerSize])
        };
    }
    toJSON() {
        const { layers, weights, biases, weightInit, activation, outputActivation, leakyReluAlpha, reluClip, momentum, dampening, nesterov, learningRate, decayRate, shuffle, gpuOptions } = this;
        return JSON.stringify({
            layers,
            weights,
            biases,
            weightInit,
            activation,
            outputActivation,
            leakyReluAlpha,
            reluClip,
            momentum,
            dampening,
            nesterov,
            learningRate,
            decayRate,
            shuffle,
            gpuOptions
        });
    }
}
exports.CatBrain = CatBrain;
