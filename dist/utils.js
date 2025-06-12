"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.shuffle = shuffle;
exports.methodToFunc = methodToFunc;
// Fisher-Yates shuffling algorithm
function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}
;
// Utility to convert a method of a class to a normal function based on source string
function methodToFunc(func, name) {
    let push = false;
    let oldFuncSource = func.toString();
    let newFuncSource = "function ";
    for (let i = 0; i < oldFuncSource.length; i++) {
        if (oldFuncSource[i] === "(" && !push) {
            push = true;
            newFuncSource += name;
        }
        if (push) {
            newFuncSource += oldFuncSource[i];
        }
    }
    return newFuncSource;
}
