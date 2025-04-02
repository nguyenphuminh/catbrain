// This file will be used for activation functions in the future. For now, there is only sigmoid.

export class Activation {
    // Sigmoid function for activation
    static sigmoid(x: number) {
        return 1 / (1 + Math.exp(-x));
    }

    // Sigmoid derivative
    static sigmoidDerivative(x: number) {
        return x * (1 - x);
    }
}
