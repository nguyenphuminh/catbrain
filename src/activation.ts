export interface ActivationOptions {
    leakyReluAlpha: number;
    reluClip: number;
}

export class Activation {
    // Sigmoid function for activation
    static sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    // Sigmoid derivative
    static sigmoidDerivative(x: number) : number {
        // Note that this function expects x to already be sigmoid(x), since this is mostly
        // used with the value of each neuron that has already gone through sigmoid
        return x * (1 - x);
    }

    // Tanh function for activation
    static tanh(x: number) : number {
        return Math.tanh(x);
    }

    // Tanh derivative
    static tanhDerivative(x: number) : number {
        // Note that this function expects x to already be tanh(x), since this is mostly
        // used with the value of each neuron that has already gone through tanh
        return 1 - x * x;
    }

    // Relu for activation
    static relu(x: number, options: ActivationOptions) : number {
        return Math.min(options.reluClip, Math.max(x, 0));
    }

    // Relu derivative
    static reluDerivative(x: number, options: ActivationOptions) : number {
        return 0 < x && x <= options.reluClip ? 1 : 0;
    }

    // Leaky Relu for activation
    static leakyRelu(x: number, options: ActivationOptions) : number {
        return Math.min(options.reluClip, x > 0 ? x : options.leakyReluAlpha * x);
    }

    // Leaky Rely derivative
    static leakyReluDerivative(x: number, options: ActivationOptions) {
        if (x > options.reluClip) return 0;
        return x > 0 ? 1 : options.leakyReluAlpha;
    }

    // Swish for activation
    static swish(x: number, options: ActivationOptions) {
        return Math.max(-options.reluClip, Math.min(x / (1 + Math.exp(-x)), options.reluClip));
    }

    // Swish derivative
    static swishDerivative(x: number, options: ActivationOptions) {
        const sigmoid = 1 / (1 + Math.exp(-x));

        return -options.reluClip < x && x < options.reluClip ? sigmoid + x * sigmoid * (1 - sigmoid) : 0;
    }
}
