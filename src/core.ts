import { Activation, ActivationOptions } from "./activation";
import { Rand, weightInitWithAct } from "./rand";
import { shuffle } from "./utils";

export interface TrainingStatus {
    iteration: number;
}

export interface TrainingOptions {
    learningRate?: number;
    decayRate?: number;
    momentum?: number;
    dampening?: number;
    nesterov?: boolean;
    callback?: (trainingStatus: TrainingStatus) => void;
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
    

    // Mostly for internal use
    public activationFunc: (x: number, options?: ActivationOptions) => number;
    public derivativeFunc: (preActValue: number, actValue: number) => number;
    public outputActivationFunc: (x: number, options?: ActivationOptions) => number;
    public outputDerivativeFunc: (preActValue: number, actValue: number) => number;
    
    public layerValues: number[][];
    public preActLayerValues: number[][];
    public errors: number[][];
    public activationOptions: ActivationOptions;
    public deltas: number[][][];

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
        this.activationOptions = {
            leakyReluAlpha: this.leakyReluAlpha,
            reluClip: this.reluClip
        }

        this.activation = options.activation || "relu";
        this.activationFunc = (Activation as Record<string, any>)[this.activation] || Activation.relu;
        const derivativeMethod = (Activation as Record<string, any>)[this.activation + "Derivative"] || Activation.reluDerivative;
        if (this.activation === "sigmoid" || this.activation === "tanh") {
            this.derivativeFunc = (preActValue: number, actValue: number) => derivativeMethod(actValue);
        } else {
            this.derivativeFunc = (preActValue: number, actValue: number) => derivativeMethod(preActValue, this.activationOptions);
        }

        this.outputActivation = options.outputActivation || "sigmoid";
        this.outputActivationFunc = (Activation as Record<string, any>)[this.outputActivation] || Activation.sigmoid;
        const outputDerivativeMethod = (Activation as Record<string, any>)[this.outputActivation + "Derivative"] || Activation.sigmoidDerivative;
        if (this.outputActivation === "sigmoid" || this.outputActivation === "tanh") {
            this.outputDerivativeFunc = (preActValue: number, actValue: number) => outputDerivativeMethod(actValue);
        } else {
            this.outputDerivativeFunc = (preActValue: number, actValue: number) => outputDerivativeMethod(preActValue, this.activationOptions);
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
    }


    /*//////////////////////////////////////////////////////////////
                                User APIs
    //////////////////////////////////////////////////////////////*/

    feedForward(inputs: number[]): number[] {
        // Avoid lookups
        const { 
            layerValues,
            outputActivationFunc,
            activationFunc,
            preActLayerValues,
            activationOptions,
            weights,
            biases
        } = this;
        const layers = layerValues.length;
        const lastLayer = layers - 1;

        // Feed new inputs to our first (input) layer
        layerValues[0] = inputs;

        // Propagate layers with layers behind them
        for (let layerIndex = 1; layerIndex < layers; layerIndex++) {
            // Avoid lookups
            const currentLayer = layerValues[layerIndex];
            const currentLayerSize = currentLayer.length;
            const currentWeights = weights[layerIndex];
            const currentBiases = biases[layerIndex];
            const prevLayer = layerValues[layerIndex - 1];
            const prevlayerSize = prevLayer.length;
            const preActCurrentLayer = preActLayerValues[layerIndex];
            const currentActivation = layerIndex === lastLayer ? outputActivationFunc : activationFunc;

            for (let nodeIndex = 0; nodeIndex < currentLayerSize; nodeIndex++) {
                // Avoid lookups
                const nodeWeights = currentWeights[nodeIndex];

                // Add bias
                preActCurrentLayer[nodeIndex] = currentBiases[nodeIndex];
    
                // Get weighed sum
                for (let prevIndex = 0; prevIndex < prevlayerSize; prevIndex++) {
                    const weight = nodeWeights[prevIndex];
                    const prevNode = prevLayer[prevIndex];
    
                    preActCurrentLayer[nodeIndex] += weight * prevNode;
                }

                // Activate
                currentLayer[nodeIndex] = currentActivation(preActCurrentLayer[nodeIndex], activationOptions);
            }
        }

        return layerValues[lastLayer];
    }

    backPropagate(inputs: number[], target: number[], options: TrainingOptions) {
        const output = this.feedForward(inputs);

        // Init
        const momentum = options?.momentum || this.momentum;
        const dampening = options?.dampening || this.dampening;
        const nesterov = options?.nesterov || this.nesterov;
        const learningRate = options?.learningRate || this.learningRate;

        // Avoid lookups
        const {
            layers,
            weights,
            errors,
            preActLayerValues,
            layerValues,
            biases,
            deltas,
            outputDerivativeFunc,
            derivativeFunc
        } = this;
        const dampFactor = 1 - dampening;
        const lastLayer = layerValues.length - 1;

        for (let layer = lastLayer; layer >= 1; layer--) {
            // Avoid lookups
            const nextLayer = layer + 1;
            const prevLayer = layer - 1;
            const nextLayerSize = layers[nextLayer];
            const nextLayerWeights = weights[nextLayer];
            const nextLayerErrors = errors[nextLayer];
            const currentPreActLayerValues = preActLayerValues[layer];
            const currentLayerValues = layerValues[layer];
            const currentLayerSize = layers[layer];
            const currentLayerWeights = weights[layer];
            const currentLayerBiases = biases[layer];
            const currentLayerDeltas = deltas[layer];
            const currentLayerErrors = errors[layer];
            const prevLayerValues = layerValues[prevLayer];
            const prevLayerSize = layers[prevLayer];
            const isLastLayer = layer === lastLayer;
            const currentDerivative = isLastLayer ? outputDerivativeFunc : derivativeFunc;

            for (let nodeIndex = 0; nodeIndex < currentLayerSize; nodeIndex++) {
                // Calculate derivative ahead of time
                const preActNeuron = currentPreActLayerValues[nodeIndex];
                const actNeuron = currentLayerValues[nodeIndex];
                const derivative = currentDerivative(preActNeuron, actNeuron);

                // Calculate error
                currentLayerErrors[nodeIndex] = 0;

                // Output layer error
                if (isLastLayer) {
                    currentLayerErrors[nodeIndex] = target[nodeIndex] - output[nodeIndex];
                }
                // Hidden layer error
                else {
                    for (let nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++) {
                        currentLayerErrors[nodeIndex] += nextLayerWeights[nextNodeIndex][nodeIndex] * nextLayerErrors[nextNodeIndex];
                    }
                }

                // Update weights for each node
                const nodeWeights = currentLayerWeights[nodeIndex];
                const nodeDeltas = currentLayerDeltas[nodeIndex];
                const nodeError = currentLayerErrors[nodeIndex];

                // Avoid lookups
                const gradBase = nodeError * derivative;

                if (nesterov) {
                    for (let prevNodeIndex = 0; prevNodeIndex < prevLayerSize; prevNodeIndex++) {
                        const gradient = gradBase * prevLayerValues[prevNodeIndex];
                        const effectiveGradient = dampFactor * gradient;

                        let delta = nodeDeltas[prevNodeIndex];

                        nodeDeltas[prevNodeIndex] = momentum * delta + effectiveGradient;

                        // Nesterov look-ahead
                        delta = momentum * delta + effectiveGradient;

                        nodeWeights[prevNodeIndex] += learningRate * delta;
                    }
                } else {
                    for (let prevNodeIndex = 0; prevNodeIndex < prevLayerSize; prevNodeIndex++) {
                        const gradient = gradBase * prevLayerValues[prevNodeIndex];

                        nodeDeltas[prevNodeIndex] = momentum * nodeDeltas[prevNodeIndex] + dampFactor * gradient;

                        nodeWeights[prevNodeIndex] += learningRate * nodeDeltas[prevNodeIndex];
                    }
                }

                // Update bias for each node
                currentLayerBiases[nodeIndex] += learningRate * nodeError;
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
            nesterov: options?.nesterov || this.nesterov
        }

        // Shuffle the dataset first
        if (this.shuffle) shuffle(trainingData);

        let dataObjectIndex = 0;

        for (let iteration = 0; iteration < iterations; iteration++) {
            if (typeof options?.callback === "function") options.callback({ iteration });

            const data = trainingData[dataObjectIndex];
            this.backPropagate(data.inputs, data.outputs, trainingOptions);

            // If we have gone through all of the dataset, reshuffle it and continue training
            if (dataObjectIndex === trainingData.length - 1) {
                if (this.shuffle) shuffle(trainingData);
            }

            // Move to the next data object, reset to the first if reached limit
            dataObjectIndex = (dataObjectIndex + 1) % trainingData.length;

            // Update the learning rate
            trainingOptions.learningRate *= trainingOptions.decayRate;
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
            shuffle
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
            shuffle
        });
    }
}
