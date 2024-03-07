import { shuffle } from "./utils";

export interface CatBrainOptions {
    inputAmount: number;
    hiddenAmounts: number[];
    outputAmount: number;
    learningRate: number;
}

export class CatBrain {
    // Basic configurations
    public inputAmount: number; // Amount of input nodes
    public hiddenAmounts: number[]; // Hidden layers with their size
    public outputAmount: number; // Amount of output nodes
    public learningRate: number;

    // The hidden layer
    public hiddenLayers: number[][]; // Hidden layers and their nodes' value
    // Weights, layer -> node in layer -> node in previous layer
    public hiddenWeights: number[][][];
    public hiddenBiases: number[][];

    // The output layer
    public outputWeights: number[][];
    public outputBias: number[];
    
    constructor(options: CatBrainOptions) {
        this.inputAmount = options.inputAmount;
        this.hiddenAmounts = options.hiddenAmounts;
        this.outputAmount = options.outputAmount;
        this.learningRate = options.learningRate || 0.01;
        
        // Hidden layer init
        this.hiddenLayers = this.hiddenAmounts.map(hiddenAmount => new Array(hiddenAmount).fill(0));
        // Init a list of layers which contains a list of weights targetting the node of that position in the layer
        this.hiddenWeights = Array.from({ length: this.hiddenLayers.length }, (hiddenLayer, layerIndex) => {
            return Array.from({ length: this.hiddenAmounts[layerIndex] }, (hiddenNode, nodeIndex) => {
                if (layerIndex === 0) {
                    return Array.from({ length: this.inputAmount }, () => Math.random() * 2 - 1);
                } else {
                    return Array.from({ length: this.hiddenAmounts[layerIndex-1] }, () => Math.random() * 2 - 1);
                }
            })
        });
        // Init a list of layers which contains their biases for each of their node
        this.hiddenBiases = Array.from({ length: this.hiddenLayers.length }, (hiddenLayer, layerIndex) => {
            return new Array(this.hiddenAmounts[layerIndex]).fill(0);
        });

        // Init a list of weights targetting the node of that position in the output layer
        this.outputWeights = Array.from({ length: this.outputAmount }, () => 
            Array.from({ length: this.hiddenAmounts[this.hiddenAmounts.length-1] }, () => Math.random() * 2 - 1)
        );
        // Init a list of bias for each output node
        this.outputBias = new Array(this.outputAmount).fill(0);
    }

    // Feed forward
    feedForward(inputs: number[]) {
        // Propagate hidden layer from input nodes
        for (let nodeIndex = 0; nodeIndex < this.hiddenAmounts[0]; nodeIndex++) {
            this.hiddenLayers[0][nodeIndex] = 0;
            
            for (let inputNodeIndex = 0; inputNodeIndex < this.inputAmount; inputNodeIndex++) {
                this.hiddenLayers[0][nodeIndex] += this.hiddenWeights[0][nodeIndex][inputNodeIndex] * inputs[inputNodeIndex];
            }
            
            this.hiddenLayers[0][nodeIndex] += this.hiddenBiases[0][nodeIndex];
            this.hiddenLayers[0][nodeIndex] = this.sigmoid(this.hiddenLayers[0][nodeIndex]);
        }

        // Propagate hidden layer from previous hidden layer
        for (let hiddenLayer = 1; hiddenLayer < this.hiddenLayers.length; hiddenLayer++) {
            // Loop through nodes in the current hidden layer
            for (let nodeIndex = 0; nodeIndex < this.hiddenAmounts[hiddenLayer]; nodeIndex++) {
                this.hiddenLayers[hiddenLayer][nodeIndex] = 0;
                
                for (let prevNodeIndex = 0; prevNodeIndex < this.hiddenAmounts[hiddenLayer-1]; prevNodeIndex++) {
                    this.hiddenLayers[hiddenLayer][nodeIndex] += 
                        this.hiddenWeights[hiddenLayer][nodeIndex][prevNodeIndex] * 
                        this.hiddenLayers[hiddenLayer-1][prevNodeIndex];
                }
                
                this.hiddenLayers[hiddenLayer][nodeIndex] += this.hiddenBiases[hiddenLayer][nodeIndex];
                this.hiddenLayers[hiddenLayer][nodeIndex] = this.sigmoid(this.hiddenLayers[hiddenLayer][nodeIndex]);
            }
        }

        // Propagate output from hidden layer nodes
        const output = new Array(this.outputAmount);

        for (let nodeIndex = 0; nodeIndex < this.outputAmount; nodeIndex++) {
            output[nodeIndex] = 0;

            for (let prevNodeIndex = 0; prevNodeIndex < this.hiddenAmounts[this.hiddenAmounts.length-1]; prevNodeIndex++) {
                output[nodeIndex] += this.outputWeights[nodeIndex][prevNodeIndex] * this.hiddenLayers[this.hiddenLayers.length-1][prevNodeIndex];    
            }

            output[nodeIndex] += this.outputBias[nodeIndex];
            output[nodeIndex] = this.sigmoid(output[nodeIndex]);
        }
        
        return output;
    }

    // Backpropagation training
    backPropagate(inputs: number[], target: number[]) {
        const output = this.feedForward(inputs);

        const errorsOutput = new Array(this.outputAmount).fill(0);
        const errorsHidden = Array.from({ length: this.hiddenAmounts.length }, () => new Array(this.hiddenAmounts[0]).fill(0));

        // Backpropagate from output layer
        for (let nodeIndex = 0; nodeIndex < this.outputAmount; nodeIndex++) {
            // Calculate error from expected value
            errorsOutput[nodeIndex] = target[nodeIndex] - output[nodeIndex];

            // Update weights for each output node
            for (let prevNodeIndex = 0; prevNodeIndex < this.hiddenAmounts[this.hiddenAmounts.length-1]; prevNodeIndex++) {
                this.outputWeights[nodeIndex][prevNodeIndex] +=
                    this.learningRate *
                    errorsOutput[nodeIndex] *
                    output[nodeIndex] *
                    (1 - output[nodeIndex]) *
                    this.hiddenLayers[this.hiddenLayers.length-1][prevNodeIndex];
            }

            // Update bias for each output node
            this.outputBias[nodeIndex] += this.learningRate * errorsOutput[nodeIndex];
        }

        // Backpropagate from hidden layer to hidden layer
        for (let hiddenLayer = this.hiddenLayers.length-1; hiddenLayer >= 0; hiddenLayer--) {
            for (let nodeIndex = 0; nodeIndex < this.hiddenAmounts[hiddenLayer]; nodeIndex++) {
                // We will calculate the hidden errors first
                errorsHidden[hiddenLayer][nodeIndex] = 0;

                // If hidden layer is close to output, calculate hidden errors from output errors
                if (hiddenLayer === this.hiddenLayers.length-1) {
                    for (let outputIndex = 0; outputIndex < this.outputAmount; outputIndex++) {
                        errorsHidden[hiddenLayer][nodeIndex] += 
                            this.outputWeights[outputIndex][nodeIndex] * 
                            errorsOutput[outputIndex];
                    }
                // Or else calculate from the next hidden layer's errors
                } else {
                    for (let nextNodeIndex = 0; nextNodeIndex < this.hiddenAmounts[hiddenLayer+1]; nextNodeIndex++) {
                        errorsHidden[hiddenLayer][nodeIndex] += 
                            this.hiddenWeights[hiddenLayer+1][nextNodeIndex][nodeIndex] * 
                            errorsHidden[hiddenLayer+1][nextNodeIndex];
                    }
                }

                // Update weights for each hidden node
                for (let prevNodeIndex = 0; prevNodeIndex < this.hiddenAmounts[hiddenLayer-1]; prevNodeIndex++) {
                    this.hiddenWeights[hiddenLayer][nodeIndex][prevNodeIndex] +=
                        this.learningRate *
                        errorsHidden[hiddenLayer][nodeIndex] *
                        this.hiddenLayers[hiddenLayer][nodeIndex] *
                        (1 - this.hiddenLayers[hiddenLayer][nodeIndex]) *
                        this.hiddenLayers[hiddenLayer-1][prevNodeIndex];
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
                    this.hiddenLayers[0][nodeIndex] *
                    (1 - this.hiddenLayers[0][nodeIndex]) *
                    inputs[prevNodeIndex];
            }
        }
    }

    // Train with iterations and given data sets
    train(iterations: number, trainingData: { inputs: number[], outputs: number[] }[]) {
        let dataObjectIndex = 0;

        // Shuffle the dataset first
        shuffle(trainingData);

        for (let iteration = 0; iteration < iterations; iteration++) {
            const data = trainingData[dataObjectIndex];
            this.backPropagate(data.inputs, data.outputs);

            // If we have gone through all of the dataset, reshuffle it and continue training
            if (dataObjectIndex === trainingData.length-1) {
                shuffle(trainingData);
            }

            // Move to the next data object, reset to the first if reached limit
            dataObjectIndex = (dataObjectIndex + 1) % trainingData.length;
        }
    }

    // Sigmoid function for activation
    sigmoid(x: number) {
        return 1 / (1 + Math.exp(-x));
    }
}
