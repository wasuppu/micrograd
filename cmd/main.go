package main

import (
	"fmt"

	"github.com/wasuppu/micrograd"
)

func main() {
	nn()
	// loss()
}

func loss() {
	n := micrograd.NewMLP(3, []int{4, 4, 1})

	xs := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0},
	}
	ys := []float64{1.0, -1.0, -1.0, 1.0}

	for k := range 20 {
		ypred := make([]*micrograd.Value, len(xs))
		for i, x := range xs {
			xi := make([]*micrograd.Value, len(x))
			for j, val := range x {
				xi[j] = micrograd.NewValue(val, fmt.Sprintf("x%d", j))
			}
			ypred[i] = n.Forward(xi)[0]
		}

		loss := micrograd.NewValue(0.0, "loss")
		for i := range ys {
			ygt := micrograd.NewValue(ys[i], "ygt")
			diff := ypred[i].Sub(ygt).Pow(2.0)
			loss = loss.Add(diff)
		}

		for _, p := range n.Parameters() {
			p.Grad = 0.0
		}

		loss.Backward()

		learningRate := -0.05
		for _, p := range n.Parameters() {
			p.Data += learningRate * p.Grad
		}

		fmt.Printf("Step %2d | Loss: %.6f\n", k, loss.Data)
	}
}

func nn() {
	x := micrograd.NewValues(2.0, 3.0, -1.0)
	n := micrograd.NewMLP(3, []int{4, 4, 1})
	micrograd.DrawDot(n.Forward(x)[0], "nn.png")
}
