package network

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

func LoadDataset(filename string) ([][]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	var vectors [][]float64
	var labels []float64

	for i, record := range records {
		word := Normalize(record[0])
		if len(word) == 0 || isNonInformative(word) {
			fmt.Printf("dont information words '%s' in record %d\n", record[0], i)
			continue
		}

		vector := WordToVector(word)
		label, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			fmt.Printf("err in record %d: incorecrt label '%s'\n", i, record[1])
			return nil, nil, err
		}

		if i < 5 {
			fmt.Printf("word %d: '%s', vector: %v, label: %v\n", i, word, vector, label)
		}
		vectors = append(vectors, vector)
		labels = append(labels, label)
	}

	return vectors, labels, nil
}

func isNonInformative(word string) bool {
	for _, char := range word {
		if (char >= 'а' && char <= 'я') || char == 'ё' || (char >= 'a' && char <= 'z') {
			return false
		}
	}

	return true
}
