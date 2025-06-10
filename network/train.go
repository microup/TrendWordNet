package network

import (
	"fmt"
	"math"
	"math/rand"
)

type TrainMode string

const (
	ByEpochs TrainMode = "epochs"
	ByError  TrainMode = "error"
)

func (n *Network) Train(dataset, valDataset [][]float64, labels, valLabels []float64, mode TrainMode, maxEpochs int, targetAccuracy, learningRate float64) {
	fmt.Println("starting train...")

	normalizedDataset := normalizeDataset(dataset)
	normalizedValDataset := normalizeDataset(valDataset)

	patience := 100
	patienceValidation := 100
	bestLoss := math.Inf(1)
	bestValAccuracy := 0.0
	noImprovement := 0
	noValImprovement := 0
	l2Lambda := 0.001

	totalSamples := float64(len(labels))
	trendCount := 0.0

	for _, label := range labels {
		if label == 1 {
			trendCount++
		}
	}

	nonTrendCount := totalSamples - trendCount
	trendWeight := totalSamples / (2 * trendCount)
	nonTrendWeight := totalSamples / (2 * nonTrendCount)

	beta1, beta2 := 0.9, 0.999
	epsilon := 1e-8

	mW1 := make([][]float64, n.InputSize)
	vW1 := make([][]float64, n.InputSize)
	mW2 := make([][]float64, n.HiddenSize1)
	vW2 := make([][]float64, n.HiddenSize1)
	mW3 := make([][]float64, n.HiddenSize2)
	vW3 := make([][]float64, n.HiddenSize2)
	mB1 := make([]float64, n.HiddenSize1)
	vB1 := make([]float64, n.HiddenSize1)
	mB2 := make([]float64, n.HiddenSize2)
	vB2 := make([]float64, n.HiddenSize2)
	mB3 := make([]float64, n.OutputSize)
	vB3 := make([]float64, n.OutputSize)

	for i := range mW1 {
		mW1[i] = make([]float64, n.HiddenSize1)
		vW1[i] = make([]float64, n.HiddenSize1)
	}

	for i := range mW2 {
		mW2[i] = make([]float64, n.HiddenSize2)
		vW2[i] = make([]float64, n.HiddenSize2)
	}

	for i := range mW3 {
		mW3[i] = make([]float64, n.OutputSize)
		vW3[i] = make([]float64, n.OutputSize)
	}

	bestNetwork := &Network{
		InputSize:   n.InputSize,
		HiddenSize1: n.HiddenSize1,
		HiddenSize2: n.HiddenSize2,
		OutputSize:  n.OutputSize,
		W1:          make([][]float64, n.InputSize),
		W2:          make([][]float64, n.HiddenSize1),
		W3:          make([][]float64, n.HiddenSize2),
		B1:          make([]float64, n.HiddenSize1),
		B2:          make([]float64, n.HiddenSize2),
		B3:          make([]float64, n.OutputSize),
	}

	for i := range n.W1 {
		bestNetwork.W1[i] = make([]float64, n.HiddenSize1)
		copy(bestNetwork.W1[i], n.W1[i])
	}

	for i := range n.W2 {
		bestNetwork.W2[i] = make([]float64, n.HiddenSize2)
		copy(bestNetwork.W2[i], n.W2[i])
	}

	for i := range n.W3 {
		bestNetwork.W3[i] = make([]float64, n.OutputSize)
		copy(bestNetwork.W3[i], n.W3[i])
	}

	copy(bestNetwork.B1, n.B1)
	copy(bestNetwork.B2, n.B2)
	copy(bestNetwork.B3, n.B3)

	for epoch := 0; epoch < maxEpochs; epoch++ {
		indices := rand.Perm(len(dataset))
		totalLoss := 0.0
		correct := 0

		for _, idx := range indices {
			input := normalizedDataset[idx]
			target := labels[idx]
			hidden1, hidden2, output := n.Forward(input)

			weight := trendWeight
			if target == 0 {
				weight = nonTrendWeight
			}

			loss := weight * (-(target*math.Log(output[0]+1e-10) + (1-target)*math.Log(1-output[0]+1e-10)))

			totalLoss += loss

			if (output[0] > 0.5 && target == 1) || (output[0] <= 0.5 && target == 0) {
				correct++
			}

			deltaOutput := make([]float64, n.OutputSize)
			for j := 0; j < n.OutputSize; j++ {
				deltaOutput[j] = output[j] - target
			}

			deltaHidden2 := make([]float64, n.HiddenSize2)

			for j := 0; j < n.HiddenSize2; j++ {
				for k := 0; k < n.OutputSize; k++ {
					deltaHidden2[j] += deltaOutput[k] * n.W3[j][k]
				}
				deltaHidden2[j] *= reluDerivative(hidden2[j])
			}

			deltaHidden1 := make([]float64, n.HiddenSize1)

			for j := 0; j < n.HiddenSize1; j++ {
				for k := 0; k < n.HiddenSize2; k++ {
					deltaHidden1[j] += deltaHidden2[k] * n.W2[j][k]
				}
				deltaHidden1[j] *= reluDerivative(hidden1[j])
			}

			t := float64(epoch*len(dataset) + idx + 1)

			for j := 0; j < n.HiddenSize2; j++ {
				for k := 0; k < n.OutputSize; k++ {
					grad := deltaOutput[k]*hidden2[j] + l2Lambda*n.W3[j][k]
					mW3[j][k] = beta1*mW3[j][k] + (1-beta1)*grad
					vW3[j][k] = beta2*vW3[j][k] + (1-beta2)*grad*grad
					mHat := mW3[j][k] / (1 - math.Pow(beta1, t))
					vHat := vW3[j][k] / (1 - math.Pow(beta2, t))
					n.W3[j][k] -= learningRate * mHat / (math.Sqrt(vHat) + epsilon)
				}
			}

			for k := 0; k < n.OutputSize; k++ {
				mB3[k] = beta1*mB3[k] + (1-beta1)*deltaOutput[k]
				vB3[k] = beta2*vB3[k] + (1-beta2)*deltaOutput[k]*deltaOutput[k]
				mHat := mB3[k] / (1 - math.Pow(beta1, t))
				vHat := vB3[k] / (1 - math.Pow(beta2, t))
				n.B3[k] -= learningRate * mHat / (math.Sqrt(vHat) + epsilon)
			}

			for j := 0; j < n.HiddenSize1; j++ {
				for k := 0; k < n.HiddenSize2; k++ {
					grad := deltaHidden2[k]*hidden1[j] + l2Lambda*n.W2[j][k]
					mW2[j][k] = beta1*mW2[j][k] + (1-beta1)*grad
					vW2[j][k] = beta2*vW2[j][k] + (1-beta2)*grad*grad
					mHat := mW2[j][k] / (1 - math.Pow(beta1, t))
					vHat := vW2[j][k] / (1 - math.Pow(beta2, t))
					n.W2[j][k] -= learningRate * mHat / (math.Sqrt(vHat) + epsilon)
				}
			}

			for k := 0; k < n.HiddenSize2; k++ {
				mB2[k] = beta1*mB2[k] + (1-beta1)*deltaHidden2[k]
				vB2[k] = beta2*vB2[k] + (1-beta2)*deltaHidden2[k]*deltaHidden2[k]
				mHat := mB2[k] / (1 - math.Pow(beta1, t))
				vHat := vB2[k] / (1 - math.Pow(beta2, t))
				n.B2[k] -= learningRate * mHat / (math.Sqrt(vHat) + epsilon)
			}

			for j := 0; j < n.InputSize; j++ {
				for k := 0; k < n.HiddenSize1; k++ {
					grad := deltaHidden1[k]*input[j] + l2Lambda*n.W1[j][k]
					mW1[j][k] = beta1*mW1[j][k] + (1-beta1)*grad
					vW1[j][k] = beta2*vW1[j][k] + (1-beta2)*grad*grad
					mHat := mW1[j][k] / (1 - math.Pow(beta1, t))
					vHat := vW1[j][k] / (1 - math.Pow(beta2, t))
					n.W1[j][k] -= learningRate * mHat / (math.Sqrt(vHat) + epsilon)
				}
			}

			for k := 0; k < n.HiddenSize1; k++ {
				mB1[k] = beta1*mB1[k] + (1-beta1)*deltaHidden1[k]
				vB1[k] = beta2*vB1[k] + (1-beta2)*deltaHidden1[k]*deltaHidden1[k]
				mHat := mB1[k] / (1 - math.Pow(beta1, t))
				vHat := vB1[k] / (1 - math.Pow(beta2, t))
				n.B1[k] -= learningRate * mHat / (math.Sqrt(vHat) + epsilon)
			}
		}

		avgLoss := totalLoss / float64(len(dataset))
		accuracy := float64(correct) / float64(len(dataset)) * 100

		// check results on test dataset
		valCorrect := 0
		for i, input := range normalizedValDataset {
			_, _, output := n.Forward(input)
			if (output[0] > 0.5 && valLabels[i] == 1) || (output[0] <= 0.5 && valLabels[i] == 0) {
				valCorrect++
			}
		}

		valAccuracy := float64(valCorrect) / float64(len(valDataset)) * 100

		//if epoch%10 == 0 || epoch == 0 {
		if mode == ByError {
			fmt.Printf("epoch %d, average err: %.6f, accuracy (train): %.2f%%, accuracy (validation): %.2f%%\n", epoch+1, avgLoss, accuracy, valAccuracy)
		} else {
			fmt.Printf("epoch %d из %d, average err: %.6f, accuracy (train): %.2f%%, accuracy (validation): %.2f%%\n", epoch+1, maxEpochs, avgLoss, accuracy, valAccuracy)
		}
		//}

		// save best results
		if valAccuracy > bestValAccuracy {
			bestValAccuracy = valAccuracy
			for i := range n.W1 {
				bestNetwork.W1[i] = make([]float64, n.HiddenSize1)
				copy(bestNetwork.W1[i], n.W1[i])
			}

			for i := range n.W2 {
				bestNetwork.W2[i] = make([]float64, n.HiddenSize2)
				copy(bestNetwork.W2[i], n.W2[i])
			}

			for i := range n.W3 {
				bestNetwork.W3[i] = make([]float64, n.OutputSize)
				copy(bestNetwork.W3[i], n.W3[i])
			}

			copy(bestNetwork.B1, n.B1)
			copy(bestNetwork.B2, n.B2)
			copy(bestNetwork.B3, n.B3)

			err := SaveNetwork(n, "data/best_network.gob")
			if err != nil {
				fmt.Println("save err:", err)
			}

			noValImprovement = 0
		} else {
			noValImprovement++
		}

		if mode == ByError && accuracy >= targetAccuracy {
			fmt.Printf("best accuracy %.2f%% (train). stop.\n", accuracy)
			break
		}

		if avgLoss < bestLoss {
			bestLoss = avgLoss
			noImprovement = 0
		} else {
			noImprovement++
			if noImprovement >= patience {
				fmt.Printf("the process was stopped by epoch %d because nothing changed", patience)
				break
			}
		}

		if noValImprovement >= patienceValidation {
			fmt.Printf("any imporove accuracy by %d epoch. stop.\n", patienceValidation)
			break
		}
	}

	for i := range n.W1 {
		n.W1[i] = make([]float64, n.HiddenSize1)
		copy(n.W1[i], bestNetwork.W1[i])
	}

	for i := range n.W2 {
		n.W2[i] = make([]float64, n.HiddenSize2)
		copy(n.W2[i], bestNetwork.W2[i])
	}

	for i := range n.W3 {
		n.W3[i] = make([]float64, n.OutputSize)
		copy(n.W3[i], bestNetwork.W3[i])
	}

	copy(n.B1, bestNetwork.B1)
	copy(n.B2, bestNetwork.B2)
	copy(n.B3, bestNetwork.B3)

	fmt.Printf("Stop. best validate accuracy: %.2f%%\n", bestValAccuracy)
}
