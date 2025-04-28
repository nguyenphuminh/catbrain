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
    layers: number[];
    weights?: number[][][];
    biases?: number[][];
    activation?: string;
    outputActivation?: string;
    leakyReluAlpha?: number;
    reluClip?: number;
    learningRate?: number;
    decayRate?: number;
    shuffle?: boolean;
}
export declare class CatBrain {
    layers: number[];
    layerValues: number[][];
    preActLayerValues: number[][];
    weights: number[][][];
    biases: number[][];
    errors: number[][];
    learningRate: number;
    decayRate: number;
    shuffle: boolean;
    activationOptions: ActivationOptions;
    activation: (x: number, options: ActivationOptions) => number;
    derivative: (preActValue: number, actValue: number) => number;
    outputActivation: (x: number, options: ActivationOptions) => number;
    outputDerivative: (preActValue: number, actValue: number) => number;
    constructor(options: CatBrainOptions);
    feedForward(inputs: number[]): number[];
    backPropagate(inputs: number[], target: number[], options: TrainingOptions): void;
    train(iterations: number, trainingData: {
        inputs: number[];
        outputs: number[];
    }[], options?: TrainingOptions): void;
}
