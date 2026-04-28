package micrograd

import "math/rand/v2"

type Neuron struct {
	Weight []*Value
	Bias   *Value
}

func NewNeuron(nin int) *Neuron {
	w := make([]*Value, nin)
	for i := range nin {
		w[i] = NewValue(rand.Float64()*2-1, "w")
	}

	b := NewValue(rand.Float64()*2-1, "b")

	return &Neuron{w, b}
}

func (n *Neuron) Forward(x []*Value) *Value {
	act := n.Bias
	for i, xi := range x {
		act = act.Add(n.Weight[i].Mul(xi))
	}
	return act.Tanh()
}

func (n *Neuron) Parameters() []*Value {
	return append(n.Weight, n.Bias)
}

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(nin int, nout int) *Layer {
	var neurons []*Neuron
	for range nout {
		neurons = append(neurons, NewNeuron(nin))
	}
	return &Layer{neurons}
}

func (l *Layer) Forward(x []*Value) []*Value {
	var outs []*Value
	for i := range len(l.Neurons) {
		outs = append(outs, l.Neurons[i].Forward(x))
	}
	return outs
}

func (l *Layer) Parameters() []*Value {
	var params []*Value
	for _, n := range l.Neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}

type MLP struct {
	Layers []*Layer
}

func NewMLP(nin int, nouts []int) *MLP {
	sz := append([]int{nin}, nouts...)

	var layers []*Layer
	for i := range nouts {
		layers = append(layers, NewLayer(sz[i], sz[i+1]))
	}
	return &MLP{layers}
}

func (m *MLP) Forward(xs []*Value) []*Value {
	for _, layer := range m.Layers {
		xs = layer.Forward(xs)
	}
	return xs
}

func (m *MLP) Parameters() []*Value {
	var params []*Value
	for _, layer := range m.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}
