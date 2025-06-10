package main

import (
	"fmt"
	"log"
	"trendswordnet/network"
)

const dataFile = "data/data.csv"
const testFile = "data/test.csv"

const networkFile = "data/network.gob"
const bestNet = "data/best_network.gob"

func main() {
	dvectors, dlabels, err := network.LoadDataset(dataFile)
	if err != nil {
		fmt.Println("err loaded network: ", err)
		return
	}

	fmt.Printf("loaded %d samples\n", len(dvectors))

	valVectors, valLabels, err := network.LoadDataset(testFile)
	if err != nil {
		fmt.Println("err opens validation data:", err)
		return
	}
	fmt.Printf("loaded %d validate samples\n", len(valVectors))

	trainOnes := 0
	for _, label := range dlabels {
		if label == 1 {
			trainOnes++
		}
	}

	fmt.Printf("balance trained sets: %.2f%% trends, %.2f%% dont trends\n",
		float64(trainOnes)/float64(len(dvectors))*100, float64(len(dvectors)-trainOnes)/float64(len(dvectors))*100)

	numValTrendy := 0
	for _, label := range valLabels {
		if label == 1 {
			numValTrendy++
		}
	}

	fmt.Printf("validation sets: %.2f%% trend words\n", float64(numValTrendy)/float64(len(valLabels))*100)

	n := network.NewNetwork(60, 128, 64, 1)
	n.Train(dvectors, valVectors, dlabels, valLabels, network.ByEpochs, 5000, 95.0, 0.001)

	err = network.SaveNetwork(n, networkFile)
	if err != nil {
		log.Fatalf("err save network:", err)
		return
	}

	fmt.Println("done. the network was saved:", networkFile)
}