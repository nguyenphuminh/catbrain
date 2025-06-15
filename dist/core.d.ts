import { GPU, IGPUSettings, IKernelMapRunShortcut, IKernelRunShortcut, KernelOutput } from "gpu.js";
export interface TrainingStatus {
    iteration: number;
}
export interface TrainingOptions {
    learningRate?: number;
    decayRate?: number;
    momentum?: number;
    dampening?: number;
    nesterov?: boolean;
    shuffle?: boolean;
    enableGPU?: boolean;
    callback?: (trainingStatus: TrainingStatus) => void;
}
export interface LayerKernels {
    weightedSumAndActivate: IKernelMapRunShortcut<{
        [key: string]: KernelOutput;
    }>;
    updateWeights: IKernelMapRunShortcut<{
        [key: string]: KernelOutput;
    }>;
    calculateErrors: IKernelRunShortcut;
    calculateOutputErrors: IKernelRunShortcut;
    addBiases: IKernelRunShortcut;
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
    gpuOptions: IGPUSettings;
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
    gpuOptions: IGPUSettings;
    activationFunc: (x: number, reluClip: number, leakyReluAlpha: number) => number;
    derivativeFunc: (x: number, reluClip: number, leakyReluAlpha: number) => number;
    outputActivationFunc: (x: number, reluClip: number, leakyReluAlpha: number) => number;
    outputDerivativeFunc: (x: number, reluClip: number, leakyReluAlpha: number) => number;
    layerValues: number[][];
    preActLayerValues: number[][];
    errors: number[][];
    deltas: number[][][];
    kernels: LayerKernels[];
    gpu: GPU;
    constructor(options: CatBrainOptions);
    feedForward(inputs: number[], options?: TrainingOptions): number[];
    backPropagate(inputs: number[], target: number[], options: TrainingOptions): void;
    train(iterations: number, trainingData: {
        inputs: number[];
        outputs: number[];
    }[], options?: TrainingOptions): void;
    initKernels(layerSize: number, prevLayerSize: number, activationFunc: Function, outputActivationFunc: Function, derivativeFunc: Function, outputDerivativeFunc: Function): LayerKernels;
    toJSON(): string;
}
