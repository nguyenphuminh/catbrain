# MNIST digit recognition example

## Setup

Install the dependencies:
```
npm install
```

## Train

There is `train.js` which can be used to train the model with 32000 samples through 8 epochs, you can run the file to train and export the model:
```
node train
```

A file called `newModel.json` will be generated, you can then replace the existing `model.json` with it to run the new trained model through `run.js`.

## Try with a sample

In `run.js` I have prepared a random input, you can replace it with your own input to test if the model is working.

Run the file:
```
node run
```
