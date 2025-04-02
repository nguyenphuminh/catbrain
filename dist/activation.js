"use strict";
// This file will be used for activation functions in the future. For now, there is only sigmoid.
Object.defineProperty(exports, "__esModule", { value: true });
exports.Activation = void 0;
class Activation {
    // Sigmoid function for activation
    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    // Sigmoid derivative
    static sigmoidDerivative(x) {
        return x * (1 - x);
    }
}
exports.Activation = Activation;
