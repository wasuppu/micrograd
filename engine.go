package micrograd

import (
	"fmt"
	"math"
)

type Value struct {
	Data     float64
	Grad     float64
	Label    string
	prev     []*Value
	op       string
	backward func()
}

func NewValue(data float64, label string, prev ...*Value) *Value {
	return &Value{Data: data, Label: label, prev: prev}
}

func NewValues(ds ...float64) []*Value {
	vs := make([]*Value, len(ds))
	for i := range len(ds) {
		vs[i] = NewValue(ds[i], "")
	}
	return vs
}

func (v Value) String() string {
	return fmt.Sprintf("Value(Data=%v)", v.Data)
}

func (v *Value) Add(u *Value) *Value {
	out := &Value{Data: v.Data + u.Data, prev: []*Value{v, u}, op: "+"}
	out.backward = func() {
		v.Grad += out.Grad
		u.Grad += out.Grad
	}
	return out
}

func (v *Value) Addn(n float64) *Value {
	return v.Add(NewValue(n, ""))
}

func (v *Value) Mul(u *Value) *Value {
	out := &Value{Data: v.Data * u.Data, prev: []*Value{v, u}, op: "*"}
	out.backward = func() {
		v.Grad += u.Data * out.Grad
		u.Grad += v.Data * out.Grad
	}
	return out
}

func (v *Value) Muln(n float64) *Value {
	return v.Mul(NewValue(n, ""))
}

func (v *Value) Neg() *Value {
	return v.Mul(NewValue(-1, ""))
}

func (v *Value) Sub(u *Value) *Value {
	return v.Add(u.Neg())
}

func (v *Value) Subn(n float64) *Value {
	return v.Sub(NewValue(n, ""))
}

func (v *Value) Pow(n float64) *Value {
	out := &Value{Data: math.Pow(v.Data, n), prev: []*Value{v}, op: fmt.Sprint("**", n)}
	out.backward = func() {
		v.Grad += n * math.Pow(v.Data, n-1) * out.Grad
	}
	return out
}

func (v *Value) Div(u *Value) *Value {
	return v.Mul(u.Pow(-1))
}

func (v *Value) Exp() *Value {
	x := v.Data
	out := &Value{Data: math.Exp(x), prev: []*Value{v}, op: "exp"}
	out.backward = func() {
		v.Grad += out.Data * out.Grad
	}
	return out
}

func (v *Value) Tanh() *Value {
	x := v.Data
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	out := &Value{Data: t, prev: []*Value{v}, op: "tanh"}
	out.backward = func() {
		v.Grad += (1 - math.Pow(t, 2)) * out.Grad
	}
	return out
}

func (v *Value) Tanh2() *Value {
	e := v.Muln(2).Exp()
	return e.Subn(1).Div(e.Addn(1))
}

func (v *Value) Backward() {
	topo := []*Value{}
	visited := map[*Value]bool{}

	var buildTopo func(*Value)
	buildTopo = func(v *Value) {
		if !visited[v] {
			visited[v] = true
			for _, prev := range v.prev {
				buildTopo(prev)
			}
			topo = append(topo, v)
		}
	}
	buildTopo(v)

	v.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		if len(topo[i].prev) != 0 {
			topo[i].backward()
		}
	}
}
