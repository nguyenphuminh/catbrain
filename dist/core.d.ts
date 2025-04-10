import { ActivationOptions } from "./activation";
export interface TrainingOptions {
    learningRate?: number;
    decayRate?: number;
}
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
    decayRate?: number;
    shuffle?: boolean;
}
export declare class CatBrain {
    inputAmount: number;
    hiddenAmounts: number[];
    outputAmount: number;
    learningRate: number;
    decayRate: number;
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
    feedForward(inputs: number[]): number[];
    feedForward(inputs: number[], getPreActivation: boolean): [number[], number[][]];
    backPropagate(inputs: number[], target: number[], options: TrainingOptions): void;
    train(iterations: number, trainingData: {
        inputs: number[];
        outputs: number[];
    }[], options?: TrainingOptions): void;
    weighedSum(currentLayer: number[], currentWeights: number[][], currentBiases: number[], prevLayer: number[]): void;
    activateLayer(currentLayer: number[], isOutput?: boolean): void;
}
