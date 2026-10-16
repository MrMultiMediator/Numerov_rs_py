#![allow(unused_variables)]
#![allow(non_snake_case)]
use ndarray::Array1;
use std::env;

enum PotentialForm {
    Quadratic,
    DoubleWell,
}

fn main() {
    println!("Hello, world!");

    let x: Array1<f64> = Array1::range(-4.0, 4.0, 0.001);

    let v: Array1<f64> = v_of_x(&x, PotentialForm::Quadratic);

    let s: f64 = 0.001;
    let dguess: f64 = 0.25;
    let tolerance: f64 = 0.000001; // Psi must be smaller than this at the extreme end to be converged

    let (upper_e, lower_e) = parse_args();

    numerov(x, v, s, dguess, tolerance, upper_e, lower_e)
}

struct EFunc {
    upper_e: f64,
    lower_e: f64,
    upper_tail: Option<f64>,
    lower_tail: Option<f64>,
    x: Option<Vec<f64>>,
    psi: Option<Vec<f64>>,
}

impl EFunc {
    // The Setter Method
    // It takes a mutable reference to self, meaning it can alter fields 
    // without losing ownership of the struct instance. 
    pub fn set_x(&mut self, x: Vec<f64>) {
        self.x = Some(x);
    }
    pub fn set_psi(&mut self, psi: Vec<f64>) {
        self.psi = Some(psi);
    }
    pub fn set_upper_e(&mut self, upper_e: f64) {
        self.upper_e = upper_e;
    }
    pub fn set_lower_e(&mut self, lower_e: f64) {
        self.lower_e = lower_e;
    }
    pub fn set_upper_tail(&mut self, upper_tail: f64) {
        self.upper_tail = Some(upper_tail);
    }
    pub fn set_lower_tail(&mut self, lower_tail: f64) {
        self.lower_tail = Some(lower_tail);
    }
    pub fn set_bounds(&mut self, upper_e: f64, lower_e: f64, upper_tail: f64, lower_tail: f64) {
        self.set_upper_e(upper_e);
        self.set_lower_e(lower_e);
        self.set_upper_tail(upper_tail);
        self.set_lower_tail(lower_tail);
    }
    pub fn check_converged(&self, tolerance: f64) -> bool {
        // (self.upper_tail - self.lower_tail).abs() < tolerance
        match(self.upper_tail, self.lower_tail){
            (Some(upper), Some(lower)) => (upper-lower).abs() < tolerance,
            _ => false
        }
    }
    pub fn new(upper_e: f64, lower_e: f64) -> Self {
        Self {
            upper_e,
            lower_e,
            upper_tail: None,
            lower_tail: None,
            x: None,
            psi: None,
        }
    }
}

fn numerov(x: Array1<f64>, v: Array1<f64>, s: f64, dguess: f64, tolerance: f64, upper_e: f64, lower_e: f64){
    /// Run the Numerov Eigensolver procedure
    /// Currently this only does the quadratic potential.
    /// It currently returns nothing and will just write the eigenvals/funcs to disk.

    // Can also do let mut Efuncs = Vec::f64::new();
    let mut Efuncs: Vec<EFunc> = Vec::new();
    let mut all_converged: bool = false;

    while !all_converged {
        if Efuncs.is_empty() {
            initial_scan(lower_e, upper_e, dguess, s, &mut Efuncs)            
        } else {

        }
    }
}

fn initial_scan(lower_e: f64, upper_e: f64, dguess: f64, s: f64, Efuncs: &mut Vec<EFunc>) {
    /// Find all eigenvalues within the range provided
    let current_guess: f64 = lower_e;
    let mut old_psi: Vec<f64> = Vec::new();

    while current_guess <= upper_e {
        let (xx, psi) = compute_psi_of_x(current_guess, s);

        if old_psi.len() < 1 {
            old_psi = psi.clone();
        }

        let old_psi_len = old_psi.len();
        let psi_len = psi.len();

        // The tail end of psi has crossed the origin between iterations, which means
        // we have crossed an eigenfunction
        if (psi[psi_len-1] > 0. && old_psi[old_psi_len-1] < 0.) || (psi[psi_len-1] < 0. && old_psi[old_psi_len-1] > 0.) {
            println!("Found eigenfunction between E = {} and {}", current_guess - dguess, current_guess);
            Efuncs.push(EFunc::new(current_guess, current_guess-dguess));
            
        }
    }
}

fn compute_psi_of_x(guess: f64, s: f64) -> (Vec<f64>, Vec<f64>) {
    let mut xx: Vec<f64> = Vec::new();
    let mut psi: Vec<f64> = Vec::new();

    // --------------------- Set initial points for psi ---------------------
    // Set the initial x position for the scan to be 2 units
    // away from where the guess E intersects with V(x)
    xx.push(-guess.sqrt().abs() - 2.0);
    psi.push(0.0);

    xx.push(xx[0] + s);
    psi.push(0.0001);
    // ----------------------------------------------------------------------

    let mut next_x = xx[0] + 2.0*s;

    // Compute psi, x over the entire relevant domain
    while next_x < guess.sqrt().abs()+2.0 {
        xx.push(next_x);
        psi.push(evaluate_psi(guess, s, &xx, &psi));

        next_x += s;
    }

    let nf: f64 = get_norm_fact(&xx, &psi);

    for val in psi.iter_mut() {
        *val *= nf
    }

    (xx, psi)
}

/// Compute the normalization facttor
fn get_norm_fact(x: &[f64], psi: &[f64]) -> f64 {
    let dx: f64 = x[1] - x[0];
    let mut nf: f64 = 0.;

    for val in psi {
        nf += dx * (val.powi(2));
    }

    1./nf.sqrt()
}


/// Compute the value of psi at a single point in space given previous values 
/// at nearby points in space (iterative procedure)
fn evaluate_psi(guess: f64, s: f64, xx: &Vec<f64>, psi: &Vec<f64>) -> f64 {
    let len = xx.len();
    let Gnm1 = 2. * (xx[len-3].powi(2)) - 2. * guess;
    let Gn = 2. * (xx[len-2].powi(2)) - 2. * guess;
    let Gnp1 = 2. * (xx[len-1].powi(2)) - 2. * guess;
    let denom = 1. - Gnp1*(s.powi(2))/12.;

    let len = psi.len();
    let mut psi_val: f64 = 0.;
    psi_val += (2. * psi[len-1] - psi[len-2]);
    psi_val += 5. * Gn * psi[len-1] * (s.powi(2))/6.;
    psi_val += Gnm1 * psi[len-2] * (s.powi(2))/12.;
    psi_val /= denom;

    psi_val
}

fn v_of_x(x: &Array1<f64>, form: PotentialForm) -> Array1<f64> {
    match form {
        PotentialForm::Quadratic => x*x,
        PotentialForm::DoubleWell => (x*x)*((x*x)-1.0),
    }
}

fn parse_args() -> (f64, f64) {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        panic!("\n\nExpected 2 args\n\n")
    }

    let lower_e: f64 = match args[1].parse() {
        Ok(val) => val,
        Err(_) => panic!("Could not parse {} to f64", args[1]),
    };
    let upper_e: f64 = args[2].parse().unwrap();

    println!("Upper E = {upper_e}, Lower E = {lower_e}");

    (upper_e, lower_e)
}
