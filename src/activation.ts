export interface ActivationOptions {
    leakyReluAlpha: number;
    reluClip: number;
    activation: string;
}

export class Activation {
    public leakyReluAlpha: number;
    public reluClip: number;
    public activate: Function;
    public derivative: Function;

    constructor(options: ActivationOptions) {
        this.leakyReluAlpha = options.leakyReluAlpha || 0.01;
        this.reluClip = options.reluClip || 5;
        this.activate = (this as Record<string, any>)[options.activation] || this.sigmoid;
        this.derivative = (this as Record<string, any>)[options.activation + "Derivative"] || this.sigmoidDerivative;
    }

    // Sigmoid function for activation
    sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    // Sigmoid derivative
    sigmoidDerivative(x: number) : number {
        // Note that this function expects x to already be sigmoid(x), since this is mostly
        // used with the value of each neuron that has already gone through sigmoid
        return x * (1 - x);
    }

    // Tanh function for activation
    tanh(x: number) : number {
        return Math.tanh(x);
    }

    // Tanh derivative
    tanhDerivative(x: number) : number {
        // Note that this function expects x to already be tanh(x), since this is mostly
        // used with the value of each neuron that has already gone through tanh
        return 1 - x * x;
    }

    // Relu for activation
    relu(x: number) : number {
        return Math.min(this.reluClip, Math.max(x, 0));
    }

    // Relu derivative
    reluDerivative(x: number) : number {
        return 0 < x && x <= this.reluClip ? 1 : 0;
    }

    // Leaky Relu for activation
    leakyRelu(x: number) : number {
        return Math.min(this.reluClip, x > 0 ? x : this.leakyReluAlpha * x);
    }

    // Leaky Rely derivative
    leakyReluDerivative(x: number) {
        if (x > this.reluClip) return 0;
        return x > 0 ? 1 : this.leakyReluAlpha;
    }
}
