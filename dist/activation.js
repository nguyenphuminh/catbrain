"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Activation = void 0;
class Activation {
    // Sigmoid function for activation
    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    // Sigmoid derivative
    static sigmoidDerivative(x) {
        // Note that this function expects x to already be sigmoid(x), since this is mostly
        // used with the value of each neuron that has already gone through sigmoid
        return x * (1 - x);
    }
    // Tanh function for activation
    static tanh(x) {
        return Math.tanh(x);
    }
    // Tanh derivative
    static tanhDerivative(x) {
        // Note that this function expects x to already be tanh(x), since this is mostly
        // used with the value of each neuron that has already gone through tanh
        return 1 - x * x;
    }
    // Relu for activation
    static relu(x, options) {
        return Math.min(options.reluClip, Math.max(x, 0));
    }
    // Relu derivative
    static reluDerivative(x, options) {
        return 0 < x && x <= options.reluClip ? 1 : 0;
    }
    // Leaky Relu for activation
    static leakyRelu(x, options) {
        return Math.min(options.reluClip, x > 0 ? x : options.leakyReluAlpha * x);
    }
    // Leaky Rely derivative
    static leakyReluDerivative(x, options) {
        if (x > options.reluClip)
            return 0;
        return x > 0 ? 1 : options.leakyReluAlpha;
    }
}
exports.Activation = Activation;
