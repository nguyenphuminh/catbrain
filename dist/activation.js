"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Activation = void 0;
class Activation {
    leakyReluAlpha;
    reluClip;
    activate;
    derivative;
    constructor(options) {
        this.leakyReluAlpha = options.leakyReluAlpha || 0.01;
        this.reluClip = options.reluClip || 5;
        this.activate = this[options.activation] || this.sigmoid;
        this.derivative = this[options.activation + "Derivative"] || this.sigmoidDerivative;
    }
    // Sigmoid function for activation
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    // Sigmoid derivative
    sigmoidDerivative(x) {
        // Note that this function expects x to already be sigmoid(x), since this is mostly
        // used with the value of each neuron that has already gone through sigmoid
        return x * (1 - x);
    }
    // Tanh function for activation
    tanh(x) {
        return Math.tanh(x);
    }
    // Tanh derivative
    tanhDerivative(x) {
        // Note that this function expects x to already be tanh(x), since this is mostly
        // used with the value of each neuron that has already gone through tanh
        return 1 - x * x;
    }
    // Relu for activation
    relu(x) {
        return Math.min(this.reluClip, Math.max(x, 0));
    }
    // Relu derivative
    reluDerivative(x) {
        return 0 < x && x <= this.reluClip ? 1 : 0;
    }
    // Leaky Relu for activation
    leakyRelu(x) {
        return Math.min(this.reluClip, x > 0 ? x : this.leakyReluAlpha * x);
    }
    // Leaky Rely derivative
    leakyReluDerivative(x) {
        if (x > this.reluClip)
            return 0;
        return x > 0 ? 1 : this.leakyReluAlpha;
    }
}
exports.Activation = Activation;
