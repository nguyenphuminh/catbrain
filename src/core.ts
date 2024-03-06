export interface CatBrainOptions {
    inputAmount: number;
    hiddenAmount: number;
    outputAmount: number;
    learningRate: number;
}

export class CatBrain {
    public weightsInputToHidden: number[][];
    public weightsHiddenToOutput: number[][];
    public biasHidden: number[];
    public biasOutput: number[];
    public hiddenLayer: number[];
    public inputAmount: number;
    public hiddenAmount: number;
    public outputAmount: number;
    public learningRate: number;

    constructor(options: CatBrainOptions) {
        this.inputAmount = options.inputAmount;
        this.hiddenAmount = options.hiddenAmount;
        this.outputAmount = options.outputAmount;
        this.learningRate = options.learningRate || 0.01;
        
        // Hidden layer init
        this.weightsInputToHidden = Array.from({ length: this.hiddenAmount }, () =>
            Array.from({ length: this.inputAmount }, () => Math.random() * 2 - 1)
        );
        this.biasHidden = Array(this.hiddenAmount).fill(0);
        this.hiddenLayer = new Array(this.hiddenAmount).fill(0);

        // Output init
        this.weightsHiddenToOutput = Array.from({ length: this.outputAmount }, () =>
            Array.from({ length: this.hiddenAmount }, () => Math.random() * 2 - 1)
        );
        this.biasOutput = Array(this.outputAmount).fill(0);
    }

    // Feed forward
    feedForward(inputs: number[]) {
        // Propagate hidden layer from input nodes
        for (let i = 0; i < this.hiddenAmount; i++) {
            this.hiddenLayer[i] = 0;
            
            for (let j = 0; j < this.inputAmount; j++) {
                this.hiddenLayer[i] += this.weightsInputToHidden[i][j] * inputs[j];
            }
            
            this.hiddenLayer[i] += this.biasHidden[i];
            this.hiddenLayer[i] = this.sigmoid(this.hiddenLayer[i]);
        }

        // Propagate output from hidden layer nodes
        const output = new Array(this.outputAmount);

        for (let i = 0; i < this.outputAmount; i++) {
            output[i] = 0;

            for (let j = 0; j < this.hiddenAmount; j++) {
                output[i] += this.weightsHiddenToOutput[i][j] * this.hiddenLayer[j];    
            }

            output[i] += this.biasOutput[i];
            output[i] = this.sigmoid(output[i]);
        }
        
        return output;
    }

    // Backpropagation training
    backPropagate(inputs: number[], target: number[]) {
        const output = this.feedForward(inputs);

        const errorsOutput = new Array(this.outputAmount);
        const errorsHidden = new Array(this.hiddenAmount);

        for (let i = 0; i < this.outputAmount; i++) {
            errorsOutput[i] = target[i] - output[i];

            for (let j = 0; j < this.hiddenAmount; j++) {
                this.weightsHiddenToOutput[i][j] +=
                    this.learningRate *
                    errorsOutput[i] *
                    output[i] *
                    (1 - output[i]) *
                    this.hiddenLayer[j];
            }

            this.biasOutput[i] += this.learningRate * errorsOutput[i];
        }

        for (let i = 0; i < this.hiddenAmount; i++) {
            errorsHidden[i] = 0;
            
            for (let j = 0; j < this.outputAmount; j++) {
                errorsHidden[i] += this.weightsHiddenToOutput[j][i] * errorsOutput[j];
            }
            
            this.biasHidden[i] += this.learningRate * errorsHidden[i];
            
            for (let j = 0; j < this.inputAmount; j++) {
                this.weightsInputToHidden[i][j] +=
                    this.learningRate *
                    errorsHidden[i] *
                    this.hiddenLayer[i] *
                    (1 - this.hiddenLayer[i]) *
                    inputs[j];
            }
        }
    }

    // Train with iterations and given data sets
    train(iterations: number, trainingData: { inputs: number[], outputs: number[] }[]) {
        for (let i = 0; i < iterations; i++) {
            const data = trainingData[Math.floor(Math.random() * trainingData.length)];
            this.backPropagate(data.inputs, data.outputs);
        }
    }

    // Sigmoid function for activation
    sigmoid(x: number) {
        return 1 / (1 + Math.exp(-x));
    }
}
