import { Activation, ActivationOptions } from "./activation";
import { shuffle } from "./utils";

export interface TrainingStatus {
    iteration: number;
}

export interface TrainingOptions {
    learningRate?: number;
    decayRate?: number;
    callback?: (trainingStatus: TrainingStatus) => void;
}

export interface CatBrainOptions {
    // Layer config
    layers: number[];

    // Self-submit weights and biases
    weights?: number[][][];
    biases?: number[][];

    // Activation configuration
    activation?: string;
    outputActivation?: string;
    leakyReluAlpha?: number;
    reluClip?: number;

    // Training configurations
    learningRate?: number;
    decayRate?: number;
    shuffle?: boolean;
}

export class CatBrain {
    public layers: number[];
    public layerValues: number[][];
    public preActLayerValues: number[][];
    public weights: number[][][];
    public biases: number[][];
    public errors: number[][];

    public learningRate: number;
    public decayRate: number;
    public shuffle: boolean;

    public activationOptions: ActivationOptions;
    public activation: (x: number, options: ActivationOptions) => number;
    public derivative: (preActValue: number, actValue: number) => number;
    public outputActivation: (x: number, options: ActivationOptions) => number;
    public outputDerivative: (preActValue: number, actValue: number) => number;

    constructor(options: CatBrainOptions) {
        // Training configuration
        this.learningRate = options.learningRate || 0.01;
        this.decayRate = options.decayRate || 1;
        this.shuffle = options.shuffle ?? true;


        // Activation function configuration
        this.activationOptions = {
            leakyReluAlpha: options.leakyReluAlpha || 0.01,
            reluClip: options.reluClip || 5
        }

        const activation = options.activation || "sigmoid";
        this.activation = (Activation as Record<string, any>)[activation] || Activation.sigmoid;
        const derivativeMethod = (Activation as Record<string, any>)[activation + "Derivative"] || Activation.sigmoidDerivative;
        this.derivative = (preActValue: number, actValue: number) => {
            if (activation === "sigmoid" || activation === "tanh") {
                return derivativeMethod(actValue);
            }

            return derivativeMethod(preActValue, this.activationOptions);
        };

        const outputActivation = options.outputActivation || "sigmoid";
        this.outputActivation = (Activation as Record<string, any>)[outputActivation] || Activation.sigmoid;
        const outputDerivativeMethod = (Activation as Record<string, any>)[outputActivation + "Derivative"] || Activation.sigmoidDerivative;
        this.outputDerivative = (preActValue: number, actValue: number) => {
            if (outputActivation === "sigmoid" || outputActivation === "tanh") {
                return outputDerivativeMethod(actValue);
            }

            return outputDerivativeMethod(preActValue, this.activationOptions);
        };


        // Model configuration
        this.layers = options.layers;

        // Init layers with the configured size and set them to 0s at first
        this.layerValues = this.layers.map(layerSize => new Array(layerSize).fill(0));
        // Init preactivation layers
        this.preActLayerValues = this.layers.map(layerSize => new Array(layerSize).fill(0));
        // Init a list of randomized weights for each node of each layer
        this.weights = options.weights || Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            return Array.from({ length: this.layers[layerIndex] }, () => {
                // Amount of weights of a node is the number of nodes of the previous layer
                return Array.from({ length: this.layers[layerIndex - 1] }, () => Math.random() * 2 - 1);
            })
        });
        // Init a list of biases for each node of each layer
        this.biases = options.biases || Array.from({ length: this.layers.length }, (layer, layerIndex) => {
            return new Array(this.layers[layerIndex]).fill(0);
        });
        // Errors cache
        this.errors = Array.from({ length: this.layers.length }, (layer, layerIndex) => new Array(this.layers[layerIndex]).fill(0));

        // Input weights, biases and pre-act values are non-existent, this is me being lazy
        this.preActLayerValues[0] = null as unknown as number[];
        this.weights[0] = null as unknown as number[][];
        this.biases[0] = null as unknown as number[];
        this.errors[0] = null as unknown as number[];
    }


    /*//////////////////////////////////////////////////////////////
                                User APIs
    //////////////////////////////////////////////////////////////*/

    feedForward(inputs: number[]): number[] {
        // Feed new inputs to our first (input) layer
        this.layerValues[0] = inputs;

        // Propagate layers with layers behind them
        for (let index = 1; index < this.layerValues.length; index++) {
            const currentLayer = this.layerValues[index];
            const weights = this.weights[index];
            const biases = this.biases[index];
            const prevLayer = this.layerValues[index - 1];
            const isOutput = index === this.layers.length-1;
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
                    currentLayer[index] = this.outputActivation(preActCurrentLayer[index], this.activationOptions);
                } else {
                    currentLayer[index] = this.activation(preActCurrentLayer[index], this.activationOptions);
                }
            }
        }

        return this.layerValues[this.layerValues.length-1];
    }

    backPropagate(inputs: number[], target: number[], options: TrainingOptions) {
        const trainingOptions = {
            learningRate: options?.learningRate || this.learningRate
        }

        const output = this.feedForward(inputs);

        const lastLayer = this.layerValues.length - 1;

        for (let layer = lastLayer; layer >= 1; layer--) {
            for (let nodeIndex = 0; nodeIndex < this.layers[layer]; nodeIndex++) {
                // Calculate derivative ahead of time
                const preActNeuron = this.preActLayerValues[layer][nodeIndex]; // layer - 1 because this does not have pre-act input layer
                const actNeuron = this.layerValues[layer][nodeIndex];
                const derivative = layer === lastLayer ? 
                                   this.outputDerivative(preActNeuron, actNeuron) :
                                   this.derivative(preActNeuron, actNeuron);

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

    train(
        iterations: number,
        trainingData: { inputs: number[], outputs: number[] }[],
        options?: TrainingOptions
    ) {
        const trainingOptions = {
            learningRate: options?.learningRate || this.learningRate,
            decayRate: options?.decayRate || this.decayRate
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
