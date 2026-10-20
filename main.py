from __future__ import annotations

import math
import random
from pprint import pprint

from pydantic import BaseModel, Field, PrivateAttr


def mse(pred: list[float], target: list[float]) -> float:
    if len(pred) != len(target):
        raise ValueError("pred and target length mismatch")
    n = len(pred)
    return sum((p - t) ** 2 for p, t in zip(pred, target)) / n


def mse_grad(pred: list[float], target: list[float]) -> list[float]:
    n = len(pred)
    return [2.0 * (p - t) / n for p, t in zip(pred, target)]


class Neuron(BaseModel):
    weights: list[float]
    bias: float = 0.0

    _last_input: list[float] = PrivateAttr(default_factory=list)
    _last_z: float = PrivateAttr(default=0.0)

    @classmethod
    def new(cls, n_inputs: int) -> Neuron:
        limit = math.sqrt(6.0 / n_inputs)
        weights = [random.uniform(-limit, limit) for _ in range(n_inputs)]
        neuron = cls(weights=weights, bias=0.0)
        neuron._last_input = [0.0] * n_inputs
        return neuron

    def forward(self, inputs: list[float]) -> float:
        if len(inputs) != len(self.weights):
            raise ValueError("Input size mismatch")
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self._last_input = list(inputs)
        self._last_z = z
        return max(z, 0.0)  # ReLU

    def backward(self, grad_output: float, learning_rate: float) -> list[float]:
        relu_grad = 1.0 if self._last_z > 0.0 else 0.0
        delta = grad_output * relu_grad

        # Input gradient is computed BEFORE the weights are updated
        grad_input = [w * delta for w in self.weights]

        for i, x in enumerate(self._last_input):
            self.weights[i] -= learning_rate * delta * x
        self.bias -= learning_rate * delta
        return grad_input


class Layer(BaseModel):
    neurons: list[Neuron] = Field(default_factory=list)

    @classmethod
    def new(cls, n_inputs: int, n_neurons: int) -> Layer:
        return cls(neurons=[Neuron.new(n_inputs) for _ in range(n_neurons)])

    def forward(self, inputs: list[float]) -> list[float]:
        return [n.forward(inputs) for n in self.neurons]

    def backward(self, grad_outputs: list[float], learning_rate: float) -> list[float]:
        if len(grad_outputs) != len(self.neurons):
            raise ValueError("Gradient size mismatch")
        n_inputs = len(self.neurons[0].weights)
        grad_inputs = [0.0] * n_inputs

        for neuron, g in zip(self.neurons, grad_outputs):
            neuron_grad = neuron.backward(g, learning_rate)
            for i, partial in enumerate(neuron_grad):
                grad_inputs[i] += partial
        return grad_inputs


class Network(BaseModel):
    layers: list[Layer] = Field(default_factory=list)

    def forward(self, input: list[float]) -> list[float]:
        x = list(input)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train_step(self, input: list[float], target: list[float], lr: float) -> float:
        pred = self.forward(input)
        loss = mse(pred, target)

        grad = mse_grad(pred, target)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
        return loss

def main() -> None:
    net = Network(
        layers=[
            Layer.new(3, 3),
            Layer.new(3, 4),
            Layer.new(4, 1),
        ]
    )

    pprint(net.model_dump())

    out = net.forward([1.0] * 3)
    print("Output:", out)


if __name__ == "__main__":
    main()
