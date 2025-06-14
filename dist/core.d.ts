import { GPU, IGPUSettings, IKernelRunShortcut } from "gpu.js";
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
    weightedSumAndActivate: any;
    updateWeights: any;
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
    initKernels(layerSize: number, prevLayerSize: number, activationFunc: Function, outputActivationFunc: Function, derivativeFunc: Function, outputDerivativeFunc: Function): {
        weightedSumAndActivate: ((this: import("gpu.js").IKernelFunctionThis<null>, prevLayer: number[], prevSize: number, weights: number[][], biases: number[], isOutput: boolean, clip: number, alpha: number) => import("gpu.js").IMappedKernelResult) & import("gpu.js").IKernelMapRunShortcut<import("gpu.js").ISubKernelObject>;
        calculateErrors: IKernelRunShortcut;
        calculateOutputErrors: IKernelRunShortcut;
        updateWeights: ((this: import("gpu.js").IKernelFunctionThis<null>, layerWeights: number[][], layerDeltas: number[][], layerErrors: number[], preActLayerValues: number[], prevLayerValues: number[], isLastLayer: boolean, nesterov: boolean, learningRate: number, dampening: number, momentum: number, reluClip: number, leakyReluAlpha: number) => import("gpu.js").IMappedKernelResult) & import("gpu.js").IKernelMapRunShortcut<import("gpu.js").ISubKernelObject>;
        addBiases: IKernelRunShortcut;
    };
    toJSON(): string;
}
