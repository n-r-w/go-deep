package deep

import (
	"encoding/json"
	"errors"
)

// Dump is a neural network dump
type Dump struct {
	Config  *Config
	Weights [][][]float64
}

var errInvalidWeights = errors.New("invalid weights")

// ApplyWeights sets the weights from a three-dimensional slice
func (n *Neural) ApplyWeights(weights [][][]float64) error {
	if len(n.Layers) != len(weights) {
		return errInvalidWeights
	}

	for i, l := range n.Layers {
		if len(n.Layers[i].Neurons) != len(weights[i]) {
			return errInvalidWeights
		}

		for j := range l.Neurons {
			if len(n.Layers[i].Neurons[j].In) != len(weights[i][j]) {
				return errInvalidWeights
			}

			for k := range l.Neurons[j].In {
				n.Layers[i].Neurons[j].In[k].Weight = weights[i][j][k]
			}
		}
	}

	return nil
}

// Weights returns all weights in sequence
func (n Neural) Weights() [][][]float64 {
	weights := make([][][]float64, len(n.Layers))
	for i, l := range n.Layers {
		weights[i] = make([][]float64, len(l.Neurons))
		for j, n := range l.Neurons {
			weights[i][j] = make([]float64, len(n.In))
			for k, in := range n.In {
				weights[i][j][k] = in.Weight
			}
		}
	}
	return weights
}

// Dump generates a network dump
func (n Neural) Dump() *Dump {
	return &Dump{
		Config:  n.Config,
		Weights: n.Weights(),
	}
}

// FromDump restores a Neural from a dump
func FromDump(dump *Dump) (*Neural, error) {
	n := NewNeural(dump.Config)
	if err := n.ApplyWeights(dump.Weights); err != nil {
		return nil, err
	}

	return n, nil
}

// Marshal marshals to JSON from network
func (n Neural) Marshal() ([]byte, error) {
	return json.Marshal(n.Dump())
}

// Unmarshal restores network from a JSON blob
func Unmarshal(bytes []byte) (*Neural, error) {
	var dump Dump
	if err := json.Unmarshal(bytes, &dump); err != nil {
		return nil, err
	}
	return FromDump(&dump)
}
