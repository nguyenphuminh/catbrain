import { GPU, IGPUSettings, IKernelRunShortcut } from "gpu.js";
import { Activation } from "./activation";
import { Rand, weightInitWithAct } from "./rand";
import { methodToFunc, shuffle } from "./utils";

export interface TrainingStatus {
    iteration: number;
}

export interface TrainingOptions {
    learningRate?: number;
    decayRate?: number;
    momentum?: number;
    dampening?: number;
    nesterov?: boolean;
    shuffle?: boolean;
    enableGPU?: boolean;
    callback?: (trainingStatus: TrainingStatus) => void;
}

export interface LayerKernels {
    weightedSum: IKernelRunShortcut;
    activateLayer: IKernelRunShortcut;
    calculateErrors: IKernelRunShortcut;
    calculateOutputErrors: IKernelRunShortcut;
}

export interface CatBrainOptions {
    // Layer config
    layers: number[];

    // Self-submit weights and biases
    weights?: number[][][];
    biases?: number[][];
    weightInit?: string;

    // Activation configuration
    activation?: string;
    outputActivation?: string;
    leakyReluAlpha?: number;
    reluClip?: number;

    // Training configurations
    momentum?: number;
    dampening?: number;
    nesterov?: boolean;
    learningRate?: number;
    decayRate?: number;
    shuffle?: boolean;

    // GPU configuration
    gpuOptions: IGPUSettings;
}

export class CatBrain {
    // Mostly for external use
    public layers: number[];

    public weights: number[][][];
    public biases: number[][];
    public weightInit: string;

    public activation: string;
    public outputActivation: string;
    public leakyReluAlpha: number;
    public reluClip: number;

    public momentum: number;
    public dampening: number;
    public nesterov: boolean;
    public learningRate: number;
    public decayRate: number;
    public shuffle: boolean;
    public gpuOptions: IGPUSettings;
    

    // Mostly for internal use
    public activationFunc: (x: number, reluClip: number, leakyReluAlpha: number) => number;
    public derivativeFunc: (preActValue: number, actValue: number) => number;
    public outputActivationFunc: (x: number, reluClip: number, leakyReluAlpha: number) => number;
    public outputDerivativeFunc: (preActValue: number, actValue: number) => number;
    
    public layerValues: number[][];
    public preActLayerValues: number[][];
    public errors: number[][];
    public deltas: number[][][];
    public kernels: LayerKernels[];
    public gpu: GPU;

    constructor(options: CatBrainOptions) {
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
        this.activationFunc = (Activation as Record<string, any>)[this.activation] || Activation.relu;
        const derivativeMethod = (Activation as Record<string, any>)[this.activation + "Derivative"] || Activation.reluDerivative;
        if (this.activation === "sigmoid" || this.activation === "tanh") {
            this.derivativeFunc = (preActValue: number, actValue: number) => derivativeMethod(actValue);
        } else {
            this.derivativeFunc = (preActValue: number, actValue: number) => derivativeMethod(preActValue, this.reluClip, this.leakyReluAlpha);
        }

        this.outputActivation = options.outputActivation || "sigmoid";
        this.outputActivationFunc = (Activation as Record<string, any>)[this.outputActivation] || Activation.sigmoid;
        const outputDerivativeMethod = (Activation as Record<string, any>)[this.outputActivation + "Derivative"] || Activation.sigmoidDerivative;
        if (this.outputActivation === "sigmoid" || this.outputActivation === "tanh") {
            this.outputDerivativeFunc = (preActValue: number, actValue: number) => outputDerivativeMethod(actValue);
        } else {
            this.outputDerivativeFunc = (preActValue: number, actValue: number) => outputDerivativeMethod(preActValue, this.reluClip, this.leakyReluAlpha);
        }


        // Model configuration
        this.layers = options.layers;

        // Choose weight init function
        this.weightInit = options.weightInit || weightInitWithAct[this.activation] || "basicUniform";
        const weightInit: (inSize: number, outSize?: number) => number = (Rand as Record<string, any>)[this.weightInit];

        // Init layers with the configured size and set them to 0s at first
        this.layerValues = this.layers.map(layerSize => new Array(layerSize).fill(0));
        // Init preactivation layers
        this.preActLayerValues = Array.from({ length: this.layers.length }, (layer, layerIndex) => { 
            if (layerIndex === 0) return null as unknown as number[];

            return new Array(this.layers[layerIndex]).fill(0);
        });
        // Init a list of randomized weights for each node of each layer
        this.weights = options.weights || Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            if (layerIndex === 0) return null as unknown as number[][];

            const outSize = this.layers[layerIndex];

            return Array.from({ length: outSize }, () => {
                const inSize = this.layers[layerIndex - 1];

                return Array.from({ length: inSize }, () => weightInit(inSize, outSize));
            })
        });
        // Init a list of biases for each node of each layer
        this.biases = options.biases || Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            if (layerIndex === 0) return null as unknown as number[];

            return new Array(this.layers[layerIndex]).fill(0);
        });
        // Errors cache
        this.errors = Array.from({ length: this.layers.length }, (layer, layerIndex) => { 
            if (layerIndex === 0) return null as unknown as number[];

            return new Array(this.layers[layerIndex]).fill(0);
        });
        // Deltas (velocity) for momentum
        this.deltas = options.weights || Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            if (layerIndex === 0) return null as unknown as number[][];
            
            return Array.from({ length: this.layers[layerIndex] }, () => {
                return Array.from({ length: this.layers[layerIndex - 1] }, () => 0);
            })
        });

         // Init GPU
        this.gpuOptions = options.gpuOptions || {};
        this.gpu = new GPU({ ...this.gpuOptions });

        // Init layers' kernels
        this.kernels = Array.from({ length: this.layers.length }, (layer, layerIndex) => { 
            if (layerIndex === 0) return null as unknown as LayerKernels;

            return this.initKernels(this.layers[layerIndex], this.activationFunc, this.outputActivationFunc);
        });
    }


    /*//////////////////////////////////////////////////////////////
                                User APIs
    //////////////////////////////////////////////////////////////*/

    feedForward(inputs: number[], options?: TrainingOptions): number[] {
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
            const isOutput = index === this.layers.length-1;

            if (enableGPU) {
                const { weightedSum, activateLayer } = this.kernels[index];

                this.preActLayerValues[index] = Array.from(weightedSum(
                    prevLayer,
                    prevlayerSize,
                    weights,
                    biases,
                ) as Float32Array)

                this.layerValues[index] = Array.from(activateLayer(
                    this.preActLayerValues[index],
                    isOutput,
                    this.reluClip,
                    this.leakyReluAlpha
                ) as Float32Array);
            } else {
                const preActCurrentLayer = this.preActLayerValues[index];
            
                for (let index = 0; index < currentLayerSize; index++) {
                    // Avoid lookups
                    const nodeWeights = weights[index];

                    // Add bias
                    preActCurrentLayer[index] = biases[index];
        
                    // Get weighed sum
                    for (let prevIndex = 0; prevIndex < prevlayerSize; prevIndex++) {
                        const weight = nodeWeights[prevIndex];
                        const prevNode = prevLayer[prevIndex];
        
                        preActCurrentLayer[index] += weight * prevNode;
                    }

                    // Activate
                    if (isOutput) {
                        currentLayer[index] = this.outputActivationFunc(preActCurrentLayer[index], this.reluClip, this.leakyReluAlpha);
                    } else {
                        currentLayer[index] = this.activationFunc(preActCurrentLayer[index], this.reluClip, this.leakyReluAlpha);
                    }
                }
            }
        }

        return this.layerValues[this.layerValues.length-1];
    }

    backPropagate(inputs: number[], target: number[], options: TrainingOptions) {
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
            const layerValues = this.layerValues[layer];
            const layerSize = this.layers[layer];
            const layerWeights = this.weights[layer];
            const layerBiases = this.biases[layer];
            const layerDeltas = this.deltas[layer];
            const prevLayerValues = this.layerValues[layer - 1];
            const prevLayerSize = this.layers[layer - 1];
            const isLastLayer = layer === lastLayer;

            // Calculate errors
            let layerErrors = this.errors[layer];

            if (enableGPU) {
                const { calculateErrors, calculateOutputErrors } = this.kernels[layer];

                if (isLastLayer) {
                    this.errors[layer] = Array.from(calculateOutputErrors(
                        target,
                        output
                    ) as Float32Array);
                } else {
                    this.errors[layer] = Array.from(calculateErrors(
                        nextLayerSize,
                        nextLayerWeights,
                        nextLayerErrors
                    ) as Float32Array);
                }   

                layerErrors = this.errors[layer];
            } else {
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
                }
            }

            // Update weights for each node
            for (let nodeIndex = 0; nodeIndex < layerSize; nodeIndex++) {
                const nodeWeights = layerWeights[nodeIndex];
                const nodeDeltas = layerDeltas[nodeIndex];
                const nodeError = layerErrors[nodeIndex];

                // Calculate derivative ahead of time
                const preActNeuron = preActLayerValues[nodeIndex];
                const actNeuron = layerValues[nodeIndex];
                const derivative = isLastLayer ? 
                                this.outputDerivativeFunc(preActNeuron, actNeuron) :
                                this.derivativeFunc(preActNeuron, actNeuron);

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
                } else {
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

    train(
        iterations: number,
        trainingData: { inputs: number[], outputs: number[] }[],
        options?: TrainingOptions
    ) {
        const trainingOptions = {
            learningRate: options?.learningRate || this.learningRate,
            decayRate: options?.decayRate || this.decayRate,
            momentum: options?.momentum || this.momentum,
            dampening: options?.dampening || this.dampening,
            nesterov: options?.nesterov ?? this.nesterov,
            shuffle: options?.shuffle ?? this.shuffle,
            enableGPU: options?.enableGPU ?? false
        }

        // Shuffle the dataset first
        if (trainingOptions.shuffle) shuffle(trainingData);

        let dataObjectIndex = 0;

        for (let iteration = 0; iteration < iterations; iteration++) {
            if (typeof options?.callback === "function") options.callback({ iteration });

            const data = trainingData[dataObjectIndex];
            this.backPropagate(data.inputs, data.outputs, trainingOptions);

            // If we have gone through all of the dataset, reshuffle it and continue training
            if (dataObjectIndex === trainingData.length - 1) {
                if (trainingOptions.shuffle) shuffle(trainingData);
            }

            // Move to the next data object, reset to the first if reached limit
            dataObjectIndex = (dataObjectIndex + 1) % trainingData.length;

            // Update the learning rate
            trainingOptions.learningRate *= trainingOptions.decayRate;
        }
    }

    initKernels(layerSize: number, activationFunc: Function, outputActivationFunc: Function) {
        const actFuncSource = methodToFunc(activationFunc, "activationFunc");
        const outputActFuncSource = methodToFunc(outputActivationFunc, "outputActivationFunc");

        return {
            weightedSum: this.gpu.createKernel(function(
                prevLayer: number[],
                prevSize: number,
                weights: number[][],
                biases: number[]
            ) {
                let sum = biases[this.thread.x];

                for (let index = 0; index < prevSize; index++) {
                    sum += prevLayer[index] * weights[this.thread.x][index];
                }

                return sum;
            })
            .setOutput([ layerSize ]),

            activateLayer: this.gpu.createKernel(function(
                layer: number[],
                isOutput: boolean,
                clip: number,
                alpha: number
            ) {
                if (isOutput) return outputActivationFunc(layer[this.thread.x], clip, alpha);

                return activationFunc(layer[this.thread.x], clip, alpha);
            })
            .setFunctions([
                {
                    source: actFuncSource,
                    settings: {}
                },
                {
                    source: outputActFuncSource,
                    settings: {}
                }])
            .setOutput([ layerSize ]),

            calculateErrors: this.gpu.createKernel(function(
                nextLayerSize: number,
                nextLayerWeights: number[][],
                nextLayerErrors: number[]
            ) {
                let errorSum = 0;

                for (let nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++) {
                    errorSum += nextLayerWeights[nextNodeIndex][this.thread.x] * nextLayerErrors[nextNodeIndex];
                }

                return errorSum;
            })
            .setOutput([ layerSize ]),

            calculateOutputErrors: this.gpu.createKernel(function(
                target: number[],
                output: number[],
            ) {
                return target[this.thread.x] - output[this.thread.x];
            })
            .setOutput([ layerSize ])
        }
    }

    toJSON() {
        const {
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
        } = this;

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
