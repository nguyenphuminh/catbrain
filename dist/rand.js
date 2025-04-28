"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Rand = exports.weightInitWithAct = void 0;
exports.weightInitWithAct = {
    "tanh": "xavierNormal",
    "relu": "heNormal",
    "leakyRelu": "heNormal",
    "swish": "heNormal",
    "mish": "heNormal",
    "softplus": "heNormal"
};
class Rand {
    static uniform(start, end) {
        return Math.random() * (end - start) + start;
    }
    static normal(mean = 0, stdDev = 1) {
        const u = 1 - Math.random();
        const v = 1 - Math.random();
        const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        return z * stdDev + mean;
    }
    static xavierUniform(inSize, outSize) {
        const limit = Math.sqrt(6 / (inSize + outSize));
        return Rand.uniform(-limit, limit);
    }
    static xavierNormal(inSize, outSize) {
        const stdDev = Math.sqrt(2 / (inSize + outSize));
        return Rand.normal(0, stdDev);
    }
    static heUniform(inSize) {
        const limit = Math.sqrt(6 / inSize);
        return Rand.uniform(-limit, limit);
    }
    static heNormal(inSize) {
        const stdDev = Math.sqrt(2 / inSize);
        return Rand.normal(0, stdDev);
    }
    static lecunUniform(inSize) {
        const limit = Math.sqrt(3 / inSize);
        return Rand.uniform(-limit, limit);
    }
    static lecunNormal(inSize) {
        const stdDev = Math.sqrt(1 / inSize);
        return Rand.normal(0, stdDev);
    }
    static basicUniform() {
        return Math.random() * 2 - 1;
    }
}
exports.Rand = Rand;
