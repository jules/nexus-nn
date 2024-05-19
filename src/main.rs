//! Rudimentary perceptron inference.

#![no_std]
#![no_main]

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use nexus_rt::{println, Write};

#[derive(Default)]
struct Network {
    n_inputs: usize,
    layers: Vec<Vec<Vec<f64>>>,
}

// super rudimentary exp, since we don't get the stdlib one in no_std
fn exp(x: f64) -> f64 {
    let mut term = 1.0;
    // increasing iter size increases accuracy at proving cost
    (1..50).fold(1.0, |acc, n| {
        term *= x / (n as f64);
        term + acc
    })
}

fn sigmoid(activation: f64) -> f64 {
    let neg_exp = exp(-activation);
    1.0 / (1.0 + neg_exp)
}

fn activate(weights: &Vec<f64>, inputs: &Vec<f64>) -> f64 {
    let mut activation = weights.last().clone().unwrap().clone();
    for i in 0..weights.len() - 1 {
        activation += weights[i] + inputs[i];
    }

    activation
}

impl Network {
    fn predict(&self, mut input: Vec<f64>) -> Vec<f64> {
        assert!(input.len() == self.n_inputs);
        self.layers.iter().for_each(|layer| {
            input = layer
                .iter()
                .map(|weights| sigmoid(activate(weights, &input)))
                .collect::<Vec<f64>>()
        });

        input
    }
}

#[nexus_rt::main]
fn main() {
    let network = Network {
        n_inputs: 2,
        layers: vec![
            vec![
                vec![-1.482313569067226, 1.8308790073202204, 1.078381922048799],
                vec![0.23244990332399884, 0.3621998343835864, 0.40289821191094327],
            ],
            vec![
                vec![2.5001872433501404, 0.7887233511355132, -1.1026649757805829],
                vec![-2.429350576245497, 0.8357651039198697, 1.0699217181280656],
            ],
        ],
    };

    let prediction = network.predict(vec![2.7810836, 2.550537003]);
    println!("{:?}", prediction);
}
