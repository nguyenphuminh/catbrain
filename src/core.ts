import { GPU, IGPUSettings, IKernelMapRunShortcut, IKernelRunShortcut, KernelOutput } from "gpu.js";
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
    weightedSumAndActivate: IKernelMapRunShortcut<{ [key: string]: KernelOutput; }>;
    updateWeights: IKernelMapRunShortcut<{ [key: string]: KernelOutput; }>;
    calculateErrors: IKernelRunShortcut;
    calculateOutputErrors: IKernelRunShortcut;
    addBiases: IKernelRunShortcut;
}

export interface CatBrainOptions {
    // Layer config
    layers: number[];

    // Self-submit weights and biases
    weights?: ArrayLike<number>[][];
    biases?: ArrayLike<number>[];
    deltas?: ArrayLike<number>[][];
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
    public derivativeFunc: (x: number, reluClip: number, leakyReluAlpha: number) => number;
    public outputActivationFunc: (x: number, reluClip: number, leakyReluAlpha: number) => number;
    public outputDerivativeFunc: (x: number, reluClip: number, leakyReluAlpha: number) => number;

    // To be exported
    public weights: Float32Array[][];
    public biases: Float32Array[];
    public deltas: Float32Array[][];

    public layerValues: Float32Array[];
    public preActLayerValues: Float32Array[];
    public errors: Float32Array[];
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
        this.derivativeFunc = (Activation as Record<string, any>)[this.activation + "Derivative"] || Activation.reluDerivative;

        this.outputActivation = options.outputActivation || "sigmoid";
        this.outputActivationFunc = (Activation as Record<string, any>)[this.outputActivation] || Activation.sigmoid;
        this.outputDerivativeFunc = (Activation as Record<string, any>)[this.outputActivation + "Derivative"] || Activation.sigmoidDerivative;


        // Model configuration
        this.layers = options.layers;

        // Choose weight init function
        this.weightInit = options.weightInit || weightInitWithAct[this.activation] || "basicUniform";
        const weightInit: (inSize: number, outSize?: number) => number = (Rand as Record<string, any>)[this.weightInit];

        // Init layers with the configured size and set them to 0s at first
        this.layerValues = this.layers.map(layerSize => new Float32Array(layerSize).fill(0));
        // Init preactivation layers
        this.preActLayerValues = Array.from({ length: this.layers.length }, (layer, layerIndex) => { 
            if (layerIndex === 0) return null as unknown as Float32Array;

            return new Float32Array(this.layers[layerIndex]).fill(0);
        });
        // Init a list of randomized weights for each node of each layer
        this.weights = (
            options.weights?.map(layerWeights => layerWeights ? layerWeights.map(nodeWeights => Float32Array.from(nodeWeights)) : layerWeights) ||
            Array.from({ length: this.layers.length }, (layer, layerIndex) => {
                if (layerIndex === 0) return null as unknown as Float32Array[];

                const outSize = this.layers[layerIndex];

                return Array.from({ length: outSize }, () => {
                    const inSize = this.layers[layerIndex - 1];

                    return Float32Array.from({ length: inSize }, () => weightInit(inSize, outSize));
                })
            })
        );
        // Init a list of biases for each node of each layer
        this.biases = (
            options.biases?.map(nodeBiases => nodeBiases ? Float32Array.from(nodeBiases) : nodeBiases) || 
            Array.from({ length: this.layers.length }, (layer, layerIndex) => {
                if (layerIndex === 0) return null as unknown as Float32Array;

                return new Float32Array(this.layers[layerIndex]).fill(0);
            })
        );
        // Deltas (velocity) for momentum
        this.deltas = (
            options.deltas?.map(layerDeltas => layerDeltas ? layerDeltas.map(nodeDeltas => Float32Array.from(nodeDeltas)) : layerDeltas) ||
            Array.from({ length: this.layers.length }, (layer, layerIndex) => {
                if (layerIndex === 0) return null as unknown as Float32Array[];
                
                return Array.from({ length: this.layers[layerIndex] }, () => {
                    return Float32Array.from({ length: this.layers[layerIndex - 1] }, () => 0);
                })
            })
        );
        // Errors cache
        this.errors = Array.from({ length: this.layers.length }, (layer, layerIndex) => { 
            if (layerIndex === 0) return null as unknown as Float32Array;

            return new Float32Array(this.layers[layerIndex]).fill(0);
        });



        // GPU configuration

        // Init GPU
        this.gpuOptions = options.gpuOptions || {};
        this.gpu = new GPU({ ...this.gpuOptions });

        // Init layers' kernels
        this.kernels = Array.from({ length: this.layers.length }, (layer, layerIndex) => { 
            if (layerIndex === 0) return null as unknown as LayerKernels;

            return this.initKernels(
                this.layers[layerIndex],
                this.layers[layerIndex-1],
                this.activationFunc,
                this.outputActivationFunc,
                this.derivativeFunc,
                this.outputDerivativeFunc
            );
        });
    }


    /*//////////////////////////////////////////////////////////////
                                User APIs
    //////////////////////////////////////////////////////////////*/

    feedForward(inputs: ArrayLike<number>, options?: TrainingOptions): Float32Array {
        const enableGPU = options?.enableGPU ?? false;

        // Feed new inputs to our first (input) layer
        this.layerValues[0] = inputs instanceof Float32Array ? inputs : Float32Array.from(inputs);

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
                const { weightedSumAndActivate } = this.kernels[index];

                const { result, weightedSum } = weightedSumAndActivate(
                    prevLayer,
                    prevlayerSize,
                    weights,
                    biases,
                    isOutput,
                    this.reluClip,
                    this.leakyReluAlpha
                );

                this.preActLayerValues[index] = weightedSum as Float32Array;
                this.layerValues[index] = result as Float32Array;
            } else {
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
                    } else {
                        currentLayer[nodeIndex] = this.activationFunc(preActCurrentLayer[nodeIndex], this.reluClip, this.leakyReluAlpha);
                    }
                }
            }
        }

        return this.layerValues[this.layerValues.length-1];
    }

    backPropagate(inputs: ArrayLike<number>, targetInput: ArrayLike<number>, options: TrainingOptions) {
        // Init
        const target = targetInput instanceof Float32Array ? targetInput : Float32Array.from(targetInput);
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
                const {
                    calculateErrors,
                    calculateOutputErrors,
                    updateWeights,
                    addBiases
                } = this.kernels[layer];

                // Calculate errors
                if (isLastLayer) {
                    this.errors[layer] = calculateOutputErrors(
                        target,
                        output
                    ) as Float32Array;
                } else {
                    this.errors[layer] = calculateErrors(
                        nextLayerSize,
                        nextLayerWeights,
                        nextLayerErrors
                    ) as Float32Array;
                }

                // Calculate deltas and update weights(
                const { calculateDeltas, result } = updateWeights(
                    layerWeights,
                    layerDeltas,
                    this.errors[layer],
                    preActLayerValues,
                    prevLayerValues,
                    isLastLayer,
                    nesterov,
                    learningRate,
                    dampening,
                    momentum,
                    this.reluClip,
                    this.leakyReluAlpha
                );

                this.deltas[layer] = calculateDeltas as Float32Array[];
                this.weights[layer] = result as Float32Array[];

                // Add biases
                this.biases[layer] = addBiases(
                    layerBiases,
                    learningRate,
                    this.errors[layer]
                ) as Float32Array;

                // this.biases[layer] = Array.from(gpuBiases);
                // console.log(gpuBiases, this.biases[layer]);
            } else {
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
    }

    train(
        iterations: number,
        trainingData: { inputs: ArrayLike<number>, outputs: ArrayLike<number> }[],
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
            // Call custom callback function
            if (typeof options?.callback === "function") options.callback({ iteration });

            // Backprop training
            const data = trainingData[dataObjectIndex];
            const inputs = data.inputs instanceof Float32Array ? data.inputs : Float32Array.from(data.inputs);
            const outputs = data.outputs instanceof Float32Array ? data.outputs : Float32Array.from(data.outputs);
            this.backPropagate(inputs, outputs, trainingOptions);

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

    initKernels(
        layerSize: number,
        prevLayerSize: number,
        activationFunc: Function,
        outputActivationFunc: Function,
        derivativeFunc: Function,
        outputDerivativeFunc: Function
    ): LayerKernels {
        const actFuncSource = methodToFunc(activationFunc, "activationFunc");
        const outputActFuncSource = methodToFunc(outputActivationFunc, "outputActivationFunc");
        const derFuncSource = methodToFunc(derivativeFunc, "derivativeFunc");
        const outputDerFuncSource = methodToFunc(outputDerivativeFunc, "outputDerivativeFunc");

        function weightedSum(sum: number) { return sum; }
        function calculateDeltas(delta: number) { return delta; }

        return {
            weightedSumAndActivate: this.gpu.createKernelMap(
                {
                    weightedSum
                },
                function(
                    prevLayer: Float32Array,
                    prevSize: number,
                    weights: Float32Array[],
                    biases: Float32Array,
                    isOutput: boolean,
                    clip: number,
                    alpha: number
                ) {
                    let sum = biases[this.thread.x];

                    for (let index = 0; index < prevSize; index++) {
                        sum += prevLayer[index] * weights[this.thread.x][index];
                    }

                    weightedSum(sum);

                    if (isOutput) return outputActivationFunc(sum, clip, alpha);

                    return activationFunc(sum, clip, alpha);
                }
            )
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
            .setOutput([ layerSize ])
            .setOptimizeFloatMemory(true)
            .setTactic("precision"),

            calculateErrors: this.gpu.createKernel(function(
                nextLayerSize: number,
                nextLayerWeights: Float32Array[],
                nextLayerErrors: Float32Array
            ) {
                let errorSum = 0;

                for (let nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++) {
                    errorSum += nextLayerWeights[nextNodeIndex][this.thread.x] * nextLayerErrors[nextNodeIndex];
                }

                return errorSum;
            })
            .setOutput([ layerSize ])
            .setOptimizeFloatMemory(true)
            .setTactic("precision"),

            calculateOutputErrors: this.gpu.createKernel(function(
                target: Float32Array,
                output: Float32Array,
            ) {
                return target[this.thread.x] - output[this.thread.x];
            })
            .setOutput([ layerSize ])
            .setOptimizeFloatMemory(true)
            .setTactic("precision"),

            updateWeights: this.gpu.createKernelMap(
                {
                    calculateDeltas
                },
                function(
                    layerWeights: Float32Array[],
                    layerDeltas: Float32Array[],
                    layerErrors: Float32Array,
                    preActLayerValues: Float32Array,
                    prevLayerValues: Float32Array,
                    isLastLayer: boolean,
                    nesterov: boolean,
                    learningRate: number,
                    dampening: number,
                    momentum: number,
                    reluClip: number,
                    leakyReluAlpha: number,
                ) {
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
                }
            )
            .setFunctions([
                {
                    source: derFuncSource,
                    settings: {}
                },
                {
                    source: outputDerFuncSource,
                    settings: {}
            }])
            .setOutput([ prevLayerSize, layerSize ])
            .setOptimizeFloatMemory(true)
            .setTactic("precision"),

            addBiases: this.gpu.createKernel(function(
                layerBiases: Float32Array,
                learningRate: number,
                nodeError: Float32Array
            ) {
                return layerBiases[this.thread.x] + learningRate * nodeError[this.thread.x];
            })
            .setOutput([ layerSize ])
            .setOptimizeFloatMemory(true)
            .setTactic("precision")
        }
    }

    toJSON(): string {
        const {
            layers,

            weights,
            biases,
            deltas,
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

            weights: weights.map(layerWeights => layerWeights ? layerWeights.map(nodeWeights => Array.from(nodeWeights)) : layerWeights),
            biases: biases.map(nodeBiases => nodeBiases ? Array.from(nodeBiases) : nodeBiases),
            deltas: deltas.map(layerDeltas => layerDeltas ? layerDeltas.map(nodeDeltas => Array.from(nodeDeltas)) : layerDeltas),
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
