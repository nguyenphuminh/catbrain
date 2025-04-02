"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.weighedSum = exports.shuffle = void 0;
// Fisher-Yates shuffling algorithm
function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}
exports.shuffle = shuffle;
;
// Weighed sum, used for feed forward networks
function weighedSum(currentLayer, currentWeights, currentBiases, prevLayer, activate) {
    for (let index = 0; index < currentLayer.length; index++) {
        currentLayer[index] = 0;
        // Get weighed sum
        for (let prevIndex = 0; prevIndex < prevLayer.length; prevIndex++) {
            const weight = currentWeights[index][prevIndex];
            const prevNode = prevLayer[prevIndex];
            currentLayer[index] += weight * prevNode;
        }
        // Add bias
        currentLayer[index] += currentBiases[index];
        // Activate
        if (typeof activate === "function") {
            currentLayer[index] = activate(currentLayer[index]);
        }
    }
}
exports.weighedSum = weighedSum;
