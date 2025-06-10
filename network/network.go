package network

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
)

type Network struct {
	InputSize   int
	HiddenSize1 int
	HiddenSize2 int
	OutputSize  int
	W1          [][]float64
	W2          [][]float64
	W3          [][]float64
	B1          []float64
	B2          []float64
	B3          []float64
}

func SaveNetwork(network *Network, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}

	defer file.Close()
	encoder := gob.NewEncoder(file)

	return encoder.Encode(network)
}

func LoadNetwork(filename string) (*Network, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("open err: %v", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	var network Network

	err = decoder.Decode(&network)
	if err != nil {
		return nil, fmt.Errorf("err decode: %v", err)
	}

	if network.InputSize <= 0 || network.HiddenSize1 <= 0 || network.HiddenSize2 <= 0 || network.OutputSize <= 0 {
		return nil, fmt.Errorf("incorect ai net size: InputSize=%d, HiddenSize1=%d, HiddenSize2=%d, OutputSize=%d",
			network.InputSize, network.HiddenSize1, network.HiddenSize2, network.OutputSize)
	}

	if len(network.B3) != network.OutputSize {
		return nil, fmt.Errorf("incorect ai net size: B3: waiting %d, got %d", network.OutputSize, len(network.B3))
	}

	if len(network.W3) != network.HiddenSize2 || len(network.W3[0]) != network.OutputSize {
		return nil, fmt.Errorf("incorect ai net size W3: waiting %dx%d, gotten %dx%d",
			network.HiddenSize2, network.OutputSize, len(network.W3), len(network.W3[0]))
	}

	return &network, nil
}

func NewNetwork(inputSize, hiddenSize1, hiddenSize2, outputSize int) *Network {
	network := &Network{
		InputSize:   inputSize,
		HiddenSize1: hiddenSize1,
		HiddenSize2: hiddenSize2,
		OutputSize:  outputSize,
		W1:          make([][]float64, inputSize),
		W2:          make([][]float64, hiddenSize1),
		W3:          make([][]float64, hiddenSize2),
		B1:          make([]float64, hiddenSize1),
		B2:          make([]float64, hiddenSize2),
		B3:          make([]float64, outputSize),
	}

	heW1 := math.Sqrt(2.0 / float64(inputSize))
	for i := range network.W1 {
		network.W1[i] = make([]float64, hiddenSize1)
		for j := range network.W1[i] {
			network.W1[i][j] = rand.NormFloat64() * heW1
		}
	}

	heW2 := math.Sqrt(2.0 / float64(hiddenSize1))
	for i := range network.W2 {
		network.W2[i] = make([]float64, hiddenSize2)
		for j := range network.W2[i] {
			network.W2[i][j] = rand.NormFloat64() * heW2
		}
	}

	xavierW3 := math.Sqrt(6.0 / float64(hiddenSize2+outputSize))
	for i := range network.W3 {
		network.W3[i] = make([]float64, outputSize)
		for j := range network.W3[i] {
			network.W3[i][j] = rand.NormFloat64() * xavierW3
		}
	}

	for i := range network.B1 {
		network.B1[i] = rand.NormFloat64() * 0.1
	}

	for i := range network.B2 {
		network.B2[i] = rand.NormFloat64() * 0.1
	}

	for i := range network.B3 {
		network.B3[i] = rand.NormFloat64() * 0.1
	}

	return network
}

func (n *Network) Forward(input []float64) (hidden1, hidden2, output []float64) {
	if len(input) != n.InputSize {
		panic(fmt.Sprintf("incorect IN size vector: waiting %d, got %d", n.InputSize, len(input)))
	}

	if len(n.B3) != n.OutputSize {
		panic(fmt.Sprintf("incorect IN size B3: waiting %d, got %d", n.OutputSize, len(n.B3)))
	}

	const dropoutRate = 0.1

	hidden1 = make([]float64, n.HiddenSize1)

	for i := 0; i < n.HiddenSize1; i++ {
		for j := 0; j < n.InputSize; j++ {
			hidden1[i] += input[j] * n.W1[j][i]
		}

		hidden1[i] += n.B1[i]
		hidden1[i] = relu(hidden1[i])

		if rand.Float64() < dropoutRate {
			hidden1[i] = 0
		} else {
			hidden1[i] /= (1 - dropoutRate)
		}
	}

	hidden2 = make([]float64, n.HiddenSize2)
	for i := 0; i < n.HiddenSize2; i++ {
		for j := 0; j < n.HiddenSize1; j++ {
			hidden2[i] += hidden1[j] * n.W2[j][i]
		}

		hidden2[i] += n.B2[i]
		hidden2[i] = relu(hidden2[i])

		if rand.Float64() < dropoutRate {
			hidden2[i] = 0
		} else {
			hidden2[i] /= (1 - dropoutRate)
		}
	}

	output = make([]float64, n.OutputSize)
	for i := 0; i < n.OutputSize; i++ {
		for j := 0; j < n.HiddenSize2; j++ {
			output[i] += hidden2[j] * n.W3[j][i]
		}

		output[i] += n.B3[i]
		output[i] = sigmoid(output[i])
	}

	return hidden1, hidden2, output
}

func (n *Network) IsTrendy(word string) bool {
	normalized := Normalize(word)
	if len(normalized) == 0 || isNonInformative(normalized) {
		return false
	}

	vector := WordToVector(normalized)
	_, _, output := n.Forward(vector)

	return output[0] > 0.5
}
