import { ActivationOptions } from "./activation";
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
    layers: number[];
    weights?: number[][][];
    biases?: number[][];
    weightInit?: string;
    activation?: string;
    outputActivation?: string;
    leakyReluAlpha?: number;
    reluClip?: number;
    momentum?: number;
    dampening?: number;
    nesterov?: boolean;
    learningRate?: number;
    decayRate?: number;
    shuffle?: boolean;
}
export declare class CatBrain {
    layers: number[];
    weights: number[][][];
    biases: number[][];
    weightInit: string;
    activation: string;
    outputActivation: string;
    leakyReluAlpha: number;
    reluClip: number;
    momentum: number;
    dampening: number;
    nesterov: boolean;
    learningRate: number;
    decayRate: number;
    shuffle: boolean;
    activationFunc: (x: number, options?: ActivationOptions) => number;
    derivativeFunc: (preActValue: number, actValue: number) => number;
    outputActivationFunc: (x: number, options?: ActivationOptions) => number;
    outputDerivativeFunc: (preActValue: number, actValue: number) => number;
    layerValues: number[][];
    preActLayerValues: number[][];
    errors: number[][];
    activationOptions: ActivationOptions;
    deltas: number[][][];
    constructor(options: CatBrainOptions);
    feedForward(inputs: number[]): number[];
    backPropagate(inputs: number[], target: number[], options: TrainingOptions): void;
    train(iterations: number, trainingData: {
        inputs: number[];
        outputs: number[];
    }[], options?: TrainingOptions): void;
    toJSON(): string;
}
