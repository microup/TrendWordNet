package network

import "math"

func WordToVector(word string) []float64 {
	vector := make([]float64, 60)

	if len(word) == 0 {
		return vector
	}

	for _, char := range word {
		if char >= 'а' && char <= 'я' {
			vector[char-'а']++
		} else if char == 'ё' {
			vector[32]++
		} else if char >= 'a' && char <= 'z' {
			vector[33+(char-'a')]++
		}
	}

	max := 1.0

	for i := 0; i < 59; i++ {
		if vector[i] > max {
			max = vector[i]
		}
	}

	if max > 0 {
		for i := 0; i < 59; i++ {
			vector[i] /= max
		}
	}

	vector[59] = math.Min(float64(len(word))/20.0, 1.0)

	return vector
}
