package network

import "math"

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}

	return 0
}

func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}

	return 0
}
