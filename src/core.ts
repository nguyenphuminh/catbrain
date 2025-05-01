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
        // Deltas for momentum
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
                    currentLayer[index] = this.outputActivationFunc(preActCurrentLayer[index], this.activationOptions);
                } else {
                    currentLayer[index] = this.activationFunc(preActCurrentLayer[index], this.activationOptions);
                }
            }
        }

        return this.layerValues[this.layerValues.length-1];
    }

    backPropagate(inputs: number[], target: number[], options: TrainingOptions) {
        const output = this.feedForward(inputs);

        // Avoid lookups
        const lastLayer = this.layerValues.length - 1;
        const momentum = options?.momentum || this.momentum;
        const learningRate = options?.learningRate || this.learningRate;

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
            const layerErrors = this.errors[layer];
            const prevLayerValues = this.layerValues[layer - 1];
            const prevLayerSize = this.layers[layer - 1];
            const isLastLayer = layer === lastLayer;

            for (let nodeIndex = 0; nodeIndex < layerSize; nodeIndex++) {
                // Calculate derivative ahead of time
                const preActNeuron = preActLayerValues[nodeIndex];
                const actNeuron = layerValues[nodeIndex];
                const derivative = isLastLayer ? 
                                   this.outputDerivativeFunc(preActNeuron, actNeuron) :
                                   this.derivativeFunc(preActNeuron, actNeuron);

                // Calculate error
                layerErrors[nodeIndex] = 0;

                // Output layer error
                if (layer === lastLayer) {
                    layerErrors[nodeIndex] = target[nodeIndex] - output[nodeIndex];
                }
                // Hidden layer error
                else {
                    for (let nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++) {
                        layerErrors[nodeIndex] += nextLayerWeights[nextNodeIndex][nodeIndex] * nextLayerErrors[nextNodeIndex];
                    }
                }

                // Update weights for each node
                const nodeWeights = layerWeights[nodeIndex];
                const nodeDeltas = layerDeltas[nodeIndex];
                const nodeError = layerErrors[nodeIndex];

                for (let prevNodeIndex = 0; prevNodeIndex < prevLayerSize; prevNodeIndex++) {
                    const gradient = nodeError * derivative * prevLayerValues[prevNodeIndex];

                    nodeDeltas[prevNodeIndex] = momentum * nodeDeltas[prevNodeIndex] + (1 - momentum) * gradient;

                    nodeWeights[prevNodeIndex] += learningRate * nodeDeltas[prevNodeIndex];
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
            momentum: options?.momentum || this.momentum
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
}
