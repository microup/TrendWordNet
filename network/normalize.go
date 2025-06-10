package network

import (
	"math"
	"strings"
)

func Normalize(word string) string {
	word = strings.ToLower(word)
	word = strings.Trim(word, ".,!?-\"'()")
	
	return strings.TrimSpace(word)
}

func normalizeDataset(dataset [][]float64) [][]float64 {
	normalized := make([][]float64, len(dataset))
	mins := make([]float64, len(dataset[0]))
	maxs := make([]float64, len(dataset[0]))

	for i := range mins {
		mins[i] = math.Inf(1)
		maxs[i] = math.Inf(-1)
	}

	for _, vec := range dataset {
		for j, val := range vec {
			if val < mins[j] {
				mins[j] = val
			}

			if val > maxs[j] {
				maxs[j] = val
			}
		}
	}

	for i := range dataset {
		normalized[i] = make([]float64, len(dataset[i]))

		for j, val := range dataset[i] {
			if maxs[j] > mins[j] {
				normalized[i][j] = (val - mins[j]) / (maxs[j] - mins[j])
			} else {
				normalized[i][j] = val
			}
		}
	}

	return normalized
}
