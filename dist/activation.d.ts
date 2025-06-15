export declare class Activation {
    static sigmoid(x: number, reluClip?: number, leakyReluAlpha?: number): number;
    static sigmoidDerivative(x: number, reluClip?: number, leakyReluAlpha?: number): number;
    static tanh(x: number, reluClip?: number, leakyReluAlpha?: number): number;
    static tanhDerivative(x: number, reluClip?: number, leakyReluAlpha?: number): number;
    static relu(x: number, reluClip: number, leakyReluAlpha?: number): number;
    static reluDerivative(x: number, reluClip: number, leakyReluAlpha?: number): number;
    static leakyRelu(x: number, reluClip: number, leakyReluAlpha: number): number;
    static leakyReluDerivative(x: number, reluClip: number, leakyReluAlpha: number): number;
    static swish(x: number, reluClip: number, leakyReluAlpha?: number): number;
    static swishDerivative(x: number, reluClip: number, leakyReluAlpha?: number): number;
    static softplus(x: number, reluClip: number, leakyReluAlpha?: number): number;
    static softplusDerivative(x: number, reluClip: number, leakyReluAlpha?: number): number;
    static mish(x: number, reluClip: number, leakyReluAlpha?: number): number;
    static mishDerivative(x: number, reluClip: number, leakyReluAlpha?: number): number;
    static linear(x: number, reluClip?: number, leakyReluAlpha?: number): number;
    static linearDerivative(x?: number, reluClip?: number, leakyReluAlpha?: number): number;
}
