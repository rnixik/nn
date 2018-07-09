package main

import (
	"fmt"
	"github.com/fxsjy/gonn/gonn"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
	"time"
)

const hiddenNeuronCount = 4
const outputNeuronCount = 1
const trainInterations = 1000
const dataFilename = "demo.csv"
const nnFilename = "demo.nn"

func readCsv(filename string) (err error, inputs [][]float64, targets [][]float64) {
	f, err := os.Open(filename)
	defer f.Close()
	if err != nil {
		return
	}
	contentBytes, err := ioutil.ReadAll(f)
	if err != nil {
		return
	}

	contents := string(contentBytes)
	lines := strings.Split(contents, "\n")

	for _, line := range lines {
		line = strings.TrimRight(line, "\r\n")
		if len(line) == 0 {
			continue
		}

		tup := strings.Split(line, ",")
		inputsRawSlice := tup[:len(tup)-1]
		targetRaw := tup[len(tup)-1]

		X := make([]float64, 0)
		for _, x := range inputsRawSlice {
			parsedX, _ := strconv.ParseFloat(x, 64)
			X = append(X, parsedX)
		}
		inputs = append(inputs, X)

		Y := make([]float64, 0)
		parsedY, _ := strconv.ParseFloat(targetRaw, 64)
		Y = append(Y, parsedY)
		targets = append(targets, Y)
	}

	return
}

func resultToInt(result []float64) int {
	y := result[0]
	if y > 0.5 {
		return 1
	}
	return 0
}

func main() {

	start := time.Now()
	err, inputs, targets := readCsv(dataFilename)
	if err != nil {
		panic(err)
	}

	trainInputs := make([][]float64, 0)
	trainTargets := make([][]float64, 0)

	testInputs := make([][]float64, 0)
	testTargets := make([][]float64, 0)

	for i, x := range inputs {
		if i%3 == 0 {
			testInputs = append(testInputs, x)
		} else {
			trainInputs = append(trainInputs, x)
		}
	}

	for i, y := range targets {
		if i%3 == 0 {
			testTargets = append(testTargets, y)
		} else {
			trainTargets = append(trainTargets, y)
		}
	}

	fmt.Printf("Train data len: %d\n", len(trainInputs))
	fmt.Printf("Test data len: %d\n", len(testTargets))

	inputsLen := len(trainInputs[0])
	fmt.Printf("Inputs len: %d\n", inputsLen)

	nn := gonn.DefaultNetwork(inputsLen, hiddenNeuronCount, outputNeuronCount, false)
	nn.Train(trainInputs, trainTargets, trainInterations)

	gonn.DumpNN(nnFilename, nn)
	nn = nil

	nn = gonn.LoadNN(nnFilename)

	errCount := 0.0
	for i := 0; i < len(testInputs); i++ {
		result := nn.Forward(testInputs[i])
		resultInt := resultToInt(result)
		expectInt := resultToInt(testTargets[i])
		//fmt.Println(resultInt, expectInt)
		if resultInt != expectInt {
			errCount += 1
		}
	}
	fmt.Println("success rate:", 1.0-errCount/float64(len(testInputs)))

	fmt.Println(time.Since(start))
}
