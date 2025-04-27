export interface ActivationOptions {
    leakyReluAlpha: number;
    reluClip: number;
}

export class Activation {
    // Sigmoid function for activation
    static sigmoid(x: number) {
        return 1 / (1 + Math.exp(-x));
    }

    // Sigmoid derivative
    static sigmoidDerivative(x: number) {
        // Note that this function expects x to already be sigmoid(x), since this is mostly
        // used with the value of each neuron that has already gone through sigmoid
        return x * (1 - x);
    }

    // Tanh function for activation
    static tanh(x: number) {
        return Math.tanh(x);
    }

    // Tanh derivative
    static tanhDerivative(x: number) {
        // Note that this function expects x to already be tanh(x), since this is mostly
        // used with the value of each neuron that has already gone through tanh
        return 1 - x * x;
    }

    // Relu for activation
    static relu(x: number, options: ActivationOptions) {
        return Math.min(options.reluClip, Math.max(x, 0));
    }

    // Relu derivative
    static reluDerivative(x: number, options: ActivationOptions) {
        return 0 < x && x <= options.reluClip ? 1 : 0;
    }

    // Leaky Relu for activation
    static leakyRelu(x: number, options: ActivationOptions) {
        if (x > 0) {
            return Math.min(options.reluClip, x);
        } else if (x > -options.reluClip) {
            return options.leakyReluAlpha * x;
        }

        return -options.reluClip * options.leakyReluAlpha;
    }

    // Leaky Rely derivative
    static leakyReluDerivative(x: number, options: ActivationOptions) {
        if (x > options.reluClip || x < -options.reluClip) return 0;
        return x > 0 ? 1 : options.leakyReluAlpha;
    }

    // Swish for activation
    static swish(x: number, options: ActivationOptions) {
        if (x > options.reluClip) {
            return options.reluClip / (1 + Math.exp(-options.reluClip));
        } else if (x < -options.reluClip) {
            return -options.reluClip / (1 + Math.exp(options.reluClip));
        }
        
        return x / (1 + Math.exp(-x));
    }

    // Swish derivative
    static swishDerivative(x: number, options: ActivationOptions) {
        if (x > options.reluClip || x < -options.reluClip) return 0;
        
        const sigmoid = 1 / (1 + Math.exp(-x));
        return sigmoid + x * sigmoid * (1 - sigmoid);
    }

    // Softplus for activation
    static softplus(x: number, options: ActivationOptions) {
        if (x > options.reluClip) return Math.log1p(Math.exp(options.reluClip));

        return Math.log1p(Math.exp(x));
    }

    // Softplus derivative
    static softplusDerivative(x: number, options: ActivationOptions) {
        return x < options.reluClip ? 1 / (1 + Math.exp(-x)) : 0;
    }

    // Mish
    static mish(x: number, options: ActivationOptions) {
        if (x > options.reluClip) {
            return options.reluClip * Math.tanh(Math.log1p(Math.exp(options.reluClip)));
        } else if (x < -options.reluClip) {
            return -options.reluClip * Math.tanh(Math.log1p(Math.exp(-options.reluClip)));
        }
        
        return x * Math.tanh(Math.log1p(Math.exp(x)));
    }

    // Mish derivative
    static mishDerivative(x: number, options: ActivationOptions) {
        if (x > options.reluClip || x < -options.reluClip) return 0;

        const softplus = Math.log1p(Math.exp(x));
        const tanhSoftplus = Math.tanh(softplus);
        const sigmoid = 1 / (1 + Math.exp(-x));
        const sech2Softplus = 1 - tanhSoftplus * tanhSoftplus;

        return tanhSoftplus + x * sech2Softplus * sigmoid;
    }

    // No activation
    static linear(x: number) {
        return x;
    }

    // No activation's derivative is just 1
    static linearDerivative() {
        return 1;
    }
}
