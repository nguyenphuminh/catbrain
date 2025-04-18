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
    // Amount of neurons
    inputAmount: number;
    hiddenAmounts: number[];
    outputAmount: number;

    // Self-submit weights and biases
    hiddenWeights?: number[][][];
    hiddenBiases?: number[][];
    outputWeights?: number[][];
    outputBias?: number[];

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
    public inputAmount: number; // Amount of input nodes
    public hiddenAmounts: number[]; // Amount of hidden nodes in each hidden layer
    public outputAmount: number; // Amount of output nodes

    public learningRate: number;
    public decayRate: number;
    public shuffle: boolean; // Choose whether to shuffle dataset when training

    public activationOptions: ActivationOptions;
    public activation: (x: number, options: ActivationOptions) => number;
    public derivative: (preActValue: number, actValue: number) => number;
    public outputActivation: (x: number, options: ActivationOptions) => number;
    public outputDerivative: (preActValue: number, actValue: number) => number;

    public hiddenLayers: number[][];
    public hiddenWeights: number[][][];
    public hiddenBiases: number[][];

    public outputWeights: number[][];
    public outputBias: number[];

    constructor(options: CatBrainOptions) {
        // Basic configuration
        this.inputAmount = options.inputAmount;
        this.hiddenAmounts = options.hiddenAmounts;
        this.outputAmount = options.outputAmount;


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

        // Init hidden layers with the configured amount of neurons and set them to 0 at first
        this.hiddenLayers = this.hiddenAmounts.map(hiddenAmount => new Array(hiddenAmount).fill(0));
        // Init a list of randomized weights for each node of each layer
        this.hiddenWeights = options.hiddenWeights || Array.from({ length: this.hiddenLayers.length }, (hiddenLayer, layerIndex) => {
            return Array.from({ length: this.hiddenAmounts[layerIndex] }, (hiddenNode, nodeIndex) => {
                // Amount of weights of a node is the number of nodes of the previous layer
                if (layerIndex === 0) {
                    return Array.from({ length: this.inputAmount }, () => Math.random() * 2 - 1);
                } else {
                    return Array.from({ length: this.hiddenAmounts[layerIndex - 1] }, () => Math.random() * 2 - 1);
                }
            })
        });
        // Init a list of biases for each node of each layer
        this.hiddenBiases = options.hiddenBiases || Array.from({ length: this.hiddenLayers.length }, (hiddenLayer, layerIndex) => {
            return new Array(this.hiddenAmounts[layerIndex]).fill(0);
        });


        // Output configuration

        // Init a list of weights targetting the node of that position in the output layer
        this.outputWeights = options.outputWeights || Array.from({ length: this.outputAmount }, () =>
            Array.from({ length: this.hiddenAmounts[this.hiddenAmounts.length - 1] }, () => Math.random() * 2 - 1)
        );
        // Init a list of bias for each output node
        this.outputBias = options.outputBias || new Array(this.outputAmount).fill(0);
    }


    /*//////////////////////////////////////////////////////////////
                                User APIs
    //////////////////////////////////////////////////////////////*/

    feedForward(inputs: number[]): number[];
    feedForward(inputs: number[], getPreActivation: boolean): [number[], number[][]];
    feedForward(inputs: number[], getPreActivation?: boolean) {
        const preActLayers = [];

        // Propagate hidden layers with layers behind them
        for (let index = 0; index < this.hiddenLayers.length; index++) {
            // Get sum
            this.weighedSum(
                this.hiddenLayers[index],
                this.hiddenWeights[index],
                this.hiddenBiases[index],
                index === 0 ? inputs : this.hiddenLayers[index - 1]
            );

            // Push a copy
            if (getPreActivation) preActLayers.push([...this.hiddenLayers[index]]);

            // Activate
            this.activateLayer(this.hiddenLayers[index]);
        }

        // Propagate output from the last hidden layer
        const output = new Array(this.outputAmount);

        // Get sum
        this.weighedSum(
            output,
            this.outputWeights,
            this.outputBias,
            this.hiddenLayers[this.hiddenLayers.length - 1]
        );

        if (getPreActivation) {
            // Push a copy
            preActLayers.push([...output]);

            // Activate
            this.activateLayer(output, true)

            return [output, preActLayers];
        }

        // Activate
        this.activateLayer(output, true);

        return output;
    }

    backPropagate(inputs: number[], target: number[], options: TrainingOptions) {
        const trainingOptions = {
            learningRate: options?.learningRate || this.learningRate
        }

        const [output, preActLayers] = this.feedForward(inputs, true);

        const errorsOutput = new Array(this.outputAmount).fill(0);
        const errorsHidden = Array.from({ length: this.hiddenAmounts.length }, () => new Array(this.hiddenAmounts[0]).fill(0));

        for (let nodeIndex = 0; nodeIndex < this.outputAmount; nodeIndex++) {
            // Calculate error from expected value
            errorsOutput[nodeIndex] = target[nodeIndex] - output[nodeIndex];

            // Pre activation output
            const preActOutput = preActLayers[preActLayers.length-1][nodeIndex];

            // Update weights for each output node
            for (let prevNodeIndex = 0; prevNodeIndex < this.hiddenAmounts[this.hiddenAmounts.length - 1]; prevNodeIndex++) {
                this.outputWeights[nodeIndex][prevNodeIndex] +=
                    trainingOptions.learningRate *
                    errorsOutput[nodeIndex] *
                    this.outputDerivative(preActOutput, output[nodeIndex]) * // Always apply sigmoid in the output layer
                    this.hiddenLayers[this.hiddenLayers.length - 1][prevNodeIndex];
            }

            // Update bias for each output node
            this.outputBias[nodeIndex] += trainingOptions.learningRate * errorsOutput[nodeIndex];
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
                } else {
                    for (let nextNodeIndex = 0; nextNodeIndex < this.hiddenAmounts[hiddenLayer + 1]; nextNodeIndex++) {
                        errorsHidden[hiddenLayer][nodeIndex] +=
                            this.hiddenWeights[hiddenLayer + 1][nextNodeIndex][nodeIndex] *
                            errorsHidden[hiddenLayer + 1][nextNodeIndex];
                    }
                }

                // Update weights for each hidden node
                for (let prevNodeIndex = 0; prevNodeIndex < this.hiddenAmounts[hiddenLayer - 1]; prevNodeIndex++) {
                    this.hiddenWeights[hiddenLayer][nodeIndex][prevNodeIndex] +=
                        trainingOptions.learningRate *
                        errorsHidden[hiddenLayer][nodeIndex] *
                        this.derivative(preActLayers[hiddenLayer][nodeIndex], this.hiddenLayers[hiddenLayer][nodeIndex]) *
                        this.hiddenLayers[hiddenLayer - 1][prevNodeIndex];
                }

                // Update bias for each hidden node
                this.hiddenBiases[hiddenLayer][nodeIndex] += trainingOptions.learningRate * errorsHidden[hiddenLayer][nodeIndex];
            }
        }

        // Backpropagate from hidden layer to input layer
        for (let nodeIndex = 0; nodeIndex < this.hiddenAmounts[0]; nodeIndex++) {
            // Update weights for each hidden node in first hidden layer
            for (let prevNodeIndex = 0; prevNodeIndex < this.inputAmount; prevNodeIndex++) {
                this.hiddenWeights[0][nodeIndex][prevNodeIndex] +=
                    trainingOptions.learningRate *
                    errorsHidden[0][nodeIndex] *
                    this.derivative(preActLayers[0][nodeIndex], this.hiddenLayers[0][nodeIndex]) *
                    inputs[prevNodeIndex];
            }

            // We don't need to update bias here because it's already done in the previous loop
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


    /*//////////////////////////////////////////////////////////////
                                Utilities
    //////////////////////////////////////////////////////////////*/

    weighedSum(
        currentLayer: number[],
        currentWeights: number[][],
        currentBiases: number[],
        prevLayer: number[]
    ) {
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

    activateLayer(
        currentLayer: number[],
        isOutput?: boolean
    ) {
        for (let index = 0; index < currentLayer.length; index++) {
            // Activate
            if (isOutput) {
                // Always apply sigmoid in the output layer
                currentLayer[index] = this.outputActivation(currentLayer[index], this.activationOptions);
            } else {
                currentLayer[index] = this.activation(currentLayer[index], this.activationOptions);
            }
        }
    }
}
