import { ActivationOptions } from "./activation";
export interface TrainingStatus {
    iteration: number;
}
export interface TrainingOptions {
    learningRate?: number;
    decayRate?: number;
    callback?: (trainingStatus: TrainingStatus) => void;
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
    outputActivation?: string;
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
    activation: (x: number, options: ActivationOptions) => number;
    derivative: (preActValue: number, actValue: number) => number;
    outputActivation: (x: number, options: ActivationOptions) => number;
    outputDerivative: (preActValue: number, actValue: number) => number;
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
