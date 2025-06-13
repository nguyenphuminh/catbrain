export class Activation {
    // Sigmoid function for activation
    static sigmoid(x: number, reluClip?: number, leakyReluAlpha?: number) {
        return 1 / (1 + Math.exp(-x));
    }

    // Sigmoid derivative
    static sigmoidDerivative(x: number, reluClip?: number, leakyReluAlpha?: number) {
        const sigmoid = 1 / (1 + Math.exp(-x));

        return sigmoid * (1 - sigmoid);
    }

    // Tanh function for activation
    static tanh(x: number, reluClip?: number, leakyReluAlpha?: number) {
        return Math.tanh(x);
    }

    // Tanh derivative
    static tanhDerivative(x: number, reluClip?: number, leakyReluAlpha?: number) {
        const tanh = Math.tanh(x);

        return 1 - tanh * tanh;
    }

    // Relu for activation
    static relu(x: number, reluClip: number, leakyReluAlpha?: number) {
        return Math.min(reluClip, Math.max(x, 0));
    }

    // Relu derivative
    static reluDerivative(x: number, reluClip: number, leakyReluAlpha?: number) {
        return 0 < x && x <= reluClip ? 1 : 0;
    }

    // Leaky Relu for activation
    static leakyRelu(x: number, reluClip: number, leakyReluAlpha: number) {
        if (x > 0) {
            return Math.min(reluClip, x);
        } else if (x > -reluClip) {
            return leakyReluAlpha * x;
        }

        return -reluClip * leakyReluAlpha;
    }

    // Leaky Rely derivative
    static leakyReluDerivative(x: number, reluClip: number, leakyReluAlpha: number) {
        if (x > reluClip || x < -reluClip) return 0;
        return x > 0 ? 1 : leakyReluAlpha;
    }

    // Swish for activation
    static swish(x: number, reluClip: number, leakyReluAlpha?: number) {
        if (x > reluClip) {
            return reluClip / (1 + Math.exp(-reluClip));
        } else if (x < -reluClip) {
            return -reluClip / (1 + Math.exp(reluClip));
        }
        
        return x / (1 + Math.exp(-x));
    }

    // Swish derivative
    static swishDerivative(x: number, reluClip: number, leakyReluAlpha?: number) {
        if (x > reluClip || x < -reluClip) return 0;
        
        const sigmoid = 1 / (1 + Math.exp(-x));
        return sigmoid + x * sigmoid * (1 - sigmoid);
    }

    // Softplus for activation
    static softplus(x: number, reluClip: number, leakyReluAlpha?: number) {
        if (x > reluClip) return Math.log1p(Math.exp(reluClip));

        return Math.log1p(Math.exp(x));
    }

    // Softplus derivative
    static softplusDerivative(x: number, reluClip: number, leakyReluAlpha?: number) {
        return x < reluClip ? 1 / (1 + Math.exp(-x)) : 0;
    }

    // Mish
    static mish(x: number, reluClip: number, leakyReluAlpha?: number) {
        if (x > reluClip) {
            return reluClip * Math.tanh(Math.log1p(Math.exp(reluClip)));
        } else if (x < -reluClip) {
            return -reluClip * Math.tanh(Math.log1p(Math.exp(-reluClip)));
        }
        
        return x * Math.tanh(Math.log1p(Math.exp(x)));
    }

    // Mish derivative
    static mishDerivative(x: number, reluClip: number, leakyReluAlpha?: number) {
        if (x > reluClip || x < -reluClip) return 0;

        const softplus = Math.log1p(Math.exp(x));
        const tanhSoftplus = Math.tanh(softplus);
        const sigmoid = 1 / (1 + Math.exp(-x));
        const sech2Softplus = 1 - tanhSoftplus * tanhSoftplus;

        return tanhSoftplus + x * sech2Softplus * sigmoid;
    }

    // No activation
    static linear(x: number, reluClip?: number, leakyReluAlpha?: number) {
        return x;
    }

    // No activation's derivative is just 1
    static linearDerivative(x?: number, reluClip?: number, leakyReluAlpha?: number) {
        return 1;
    }
}
