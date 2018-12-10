#region MIT License

// Copyright (c) 2018-2022 Ai-Beta - Daniel Baumert
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#endregion   

using System;
using System.Collections.Generic;

namespace AiNetwork {
    public class Ai {

        public static Random R;

        private List<Layer> Layers { get; set; }

        private int LayersCount { get => Layers.Count; }
        public double LearningRate { get; }

        static Ai() {
            R = new Random();
        }


        public Ai(double learningRate, int[] nLayers) {
            if (nLayers.Length < 2) {
                throw new Exception("single layer is not posibil");
            }
            Layers = new List<Layer>();

            LearningRate = learningRate;

            for (int layerIndex = 0, n = nLayers.Length; layerIndex < n; layerIndex += 1) {
                int neuronCount = nLayers[layerIndex];
                Layer layer = new Layer(neuronCount - 1);
                Layers.Add(layer);

                for (int nNeuron = 0; nNeuron < neuronCount; nNeuron += 1) {
                    layer.Neurons.Add(new Neuron());
                }

                for (int i = 0; i < layer.NeuronCount; i++) {
                    if (layerIndex == 0) {
                        layer.Neurons[i].Bias = 0;
                    } else {
                        for (int ii = 0; ii < nLayers[layerIndex - 1]; ii++) {
                            layer.Neurons[i].Dendrites.Add(new Dendrite());
                        }
                    }
                }
            }
        }

        public double[] Execute(double[] inputs) {
            if (inputs.Length != Layers[0].NeuronCount) {
                throw new IndexOutOfRangeException("Input is to smal");
            }

            for (int i = 0; i < LayersCount; i++) {
                Layer selectLayer = Layers[i];
                for (int n = 0; n < selectLayer.NeuronCount; n++) {
                    Neuron neuron = selectLayer.Neurons[n];

                    if (i == 0) { //first layer = input
                        neuron.Value = inputs[n];
                    } else {
                        neuron.Value = 0;
                        for (int nNeuron = 0; nNeuron < Layers[i - 1].NeuronCount; nNeuron++) {
                            neuron.Value += Layers[i - 1].Neurons[nNeuron].Value * neuron.Dendrites[nNeuron].Weight;
                        }

                        neuron.Value = Sigmoid(neuron.Value + neuron.Bias);
                    }

                }
            }

            Layer layer = Layers[LayersCount - 1];

            double[] output = new double[layer.NeuronCount];

            for (int i = 0; i < layer.NeuronCount; i++) {
                output[i] = layer.Neurons[i].Value;
            }

            return output;
        }

        public void Train(double[] inputs, double[] outputs) {
            if (inputs.Length != Layers[0].NeuronCount || outputs.Length != Layers[LayersCount - 1].NeuronCount) {
                throw new Exception();
            }

            Execute(inputs);

            for (int i = 0; i < Layers[LayersCount - 1].NeuronCount; i++) {
                Neuron neuron = Layers[LayersCount - 1].Neurons[i];
                neuron.Delta = neuron.Value * (1 - neuron.Value) * (outputs[i] - neuron.Value);

                for (int iLayer = LayersCount - 2; iLayer >= 0; iLayer--) {
                    Layer layer = Layers[iLayer];
                    Layer nLayer = Layers[iLayer + 1];
                    for (int iNeuron = 0; iNeuron < layer.NeuronCount; iNeuron++) {
                        Neuron nNeuron = layer.Neurons[iNeuron];
                        nNeuron.Delta = nNeuron.Value * (1 - nNeuron.Value) * nLayer.Neurons[i].Dendrites[iNeuron].Weight * nLayer.Neurons[i].Delta;
                    }
                }
            }

            for (int ii = LayersCount - 1; ii >= 0; ii--) {
                Layer layer = Layers[ii];
                for (int jj = 0; jj < layer.NeuronCount; jj++) {
                    Neuron neuron = layer.Neurons[jj];
                    neuron.Bias = neuron.Bias + (LearningRate * neuron.Delta);

                    for (int kk = 0; kk < neuron.DendriteCount; kk++) {
                        neuron.Dendrites[kk].Weight = neuron.Dendrites[kk].Weight + (LearningRate * Layers[ii - 1].Neurons[kk].Value * neuron.Delta);
                    }
                }
            }
        }

        private double Sigmoid(double x) => 1d / (1d + Math.Exp(-x));

        private class Dendrite {
            public double Weight { get; set; }
            public Dendrite() {
                Weight = R.NextDouble();
            }
        }
        private class Neuron {
            public List<Dendrite> Dendrites { get; set; }
            public int DendriteCount { get; set; }
            public double Bias { get; set; }
            public double Value { get; set; }
            public double Delta { get; set; }
            public Neuron() {
                Dendrites = new List<Dendrite>();
                Bias = R.NextDouble();
            }
        }
        private class Layer {
            public List<Neuron> Neurons { get; set; }
            public int NeuronCount { get => Neurons.Count; }
            public Layer(int neuronCount) => Neurons = new List<Neuron>(neuronCount);
        }
    }
}
