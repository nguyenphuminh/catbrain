import { ActivationOptions } from "./activation";
export interface CatBrainOptions {
    inputAmount: number;
    hiddenAmounts: number[];
    outputAmount: number;
    hiddenWeights?: number[][][];
    hiddenBiases?: number[][];
    outputWeights?: number[][];
    outputBias?: number[];
    activation?: string;
    leakyReluAlpha?: number;
    reluClip?: number;
    learningRate?: number;
    shuffle?: boolean;
}
export declare class CatBrain {
    inputAmount: number;
    hiddenAmounts: number[];
    outputAmount: number;
    learningRate: number;
    shuffle: boolean;
    activationOptions: ActivationOptions;
    activation: Function;
    derivative: Function;
    hiddenLayers: number[][];
    hiddenWeights: number[][][];
    hiddenBiases: number[][];
    outputWeights: number[][];
    outputBias: number[];
    constructor(options: CatBrainOptions);
    feedForward(inputs: number[]): any[];
    backPropagate(inputs: number[], target: number[]): void;
    train(iterations: number, trainingData: {
        inputs: number[];
        outputs: number[];
    }[]): void;
    weighedSum(currentLayer: number[], currentWeights: number[][], currentBiases: number[], prevLayer: number[], isOutput?: boolean): void;
}
