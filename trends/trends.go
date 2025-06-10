package main

import (
	"fmt"
	"log"

	"trendswordnet/network"
)

func main() {
	loadedNetwork, err := network.LoadNetwork("data/network.gob")
	if err != nil {
		log.Fatalf("err load network: %v", err)
	}

	testWords := []string{"кот", "программа", "база", "данные", "it", "идти", "ai", "бегун", "лететь", "data", "воздух", "ленивый", "ноутбук", "главные", "метров", "стране"}

	for _, word := range testWords {
		isTrend := loadedNetwork.IsTrendy(word)

		// only for viewing
		normalizedWord := network.Normalize(word)
		vector := network.WordToVector(normalizedWord)
		_, _, output := loadedNetwork.Forward(vector)

		fmt.Printf("word '%s': %s (accuracy: %.2f%%)\n", word, map[bool]string{true: "is trend", false: "not trend"}[isTrend], output[0]*100)
	}
}
