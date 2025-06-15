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
    weights?: ArrayLike<number>[][];
    biases?: ArrayLike<number>[];
    deltas?: ArrayLike<number>[][];
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
    weights: Float32Array[][];
    biases: Float32Array[];
    deltas: Float32Array[][];
    layerValues: Float32Array[];
    preActLayerValues: Float32Array[];
    errors: Float32Array[];
    kernels: LayerKernels[];
    gpu: GPU;
    constructor(options: CatBrainOptions);
    feedForward(inputs: ArrayLike<number>, options?: TrainingOptions): Float32Array;
    backPropagate(inputs: ArrayLike<number>, targetInput: ArrayLike<number>, options: TrainingOptions): void;
    train(iterations: number, trainingData: {
        inputs: ArrayLike<number>;
        outputs: ArrayLike<number>;
    }[], options?: TrainingOptions): void;
    initKernels(layerSize: number, prevLayerSize: number, activationFunc: Function, outputActivationFunc: Function, derivativeFunc: Function, outputDerivativeFunc: Function): LayerKernels;
    toJSON(): string;
}
