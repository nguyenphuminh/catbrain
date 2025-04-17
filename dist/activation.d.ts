export interface ActivationOptions {
    leakyReluAlpha: number;
    reluClip: number;
}
export declare class Activation {
    static sigmoid(x: number): number;
    static sigmoidDerivative(x: number): number;
    static tanh(x: number): number;
    static tanhDerivative(x: number): number;
    static relu(x: number, options: ActivationOptions): number;
    static reluDerivative(x: number, options: ActivationOptions): number;
    static leakyRelu(x: number, options: ActivationOptions): number;
    static leakyReluDerivative(x: number, options: ActivationOptions): number;
    static swish(x: number, options: ActivationOptions): number;
    static swishDerivative(x: number, options: ActivationOptions): number;
    static softplus(x: number, options: ActivationOptions): number;
    static softplusDerivative(x: number, options: ActivationOptions): number;
    static linear(x: number): number;
    static linearDerivative(): number;
}
