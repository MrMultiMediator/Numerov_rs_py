import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class efunc():
    def __init__(self, upperE, lowerE):
        self.upperE = upperE
        self.lowerE = lowerE

    def set_E(self, E):
        self.E = E

    def set_x(self, x):
        self.x = x

    def set_psi(self, psi):
        self.psi = psi

    def set_upperE(self, upperE):
        self.upperE = upperE

    def set_lowerE(self, lowerE):
        self.lowerE = lowerE

    def set_lower_tail(self, lower_tail):
        self.lower_tail = lower_tail

    def set_upper_tail(self, upper_tail):
        self.upper_tail = upper_tail

    def set_bounds(self, upperE, lowerE, upper_tail, lower_tail):
        self.set_upperE(upperE)
        self.set_lowerE(lowerE)
        self.set_upper_tail(upper_tail)
        self.set_lower_tail(lower_tail)

    def converged(self, tolerance):
        "Check if this eigenfunction is converged"
        return abs(self.upper_tail - self.lower_tail) < tolerance

def V_of_x(x, form):
    forms = ["quadratic", "double_well"]
    if form.lower() not in forms:
        raise ValueError(f"Potential form {form} not recognized. Please use one of "
                         f"{forms}.")

    V = {
        "quadratic":x**2.,
        "double_well":(x**2)*((x**2.)-1.)
    }

    return V[form]

def evaluate_psi(guess, s, xx, psi):
    """Compute the value of psi at a single point in space given previous values 
    at nearby points in space (iterative procedure)"""
    Gnm1 = 2.*(xx[-3]**2.) - 2.*guess
    Gn = 2.*(xx[-2]**2.) - 2.*guess
    Gnp1 = 2.*(xx[-1]**2.) - 2.*guess
    denom = 1. - Gnp1*(s**2.)/12.

    psi_val = 0.
    psi_val += (2.*psi[-1] - psi[-2])
    psi_val += 5.*Gn*psi[-1]*(s**2.)/6.
    psi_val += Gnm1*psi[-2]*(s**2.)/12.
    psi_val /= denom

    if denom < 0.:
        print("Negative denominator. Exiting")
        sys.exit()

    return psi_val

def get_norm_fact(x, psi):
    "Compute the normalization facttor"
    dx = x[1] - x[0]
    nf = 0.

    for val in psi:
        nf += dx*(val**2.)

    nf = 1./np.sqrt(nf)

    return nf

def integrate(x, psi):
    "Integrate psi**2."
    dx = x[1] - x[0]
    area = 0.

    for val in psi:
        area += dx*(val**2.)

    return area

def compute_psi_of_x(guess, s):
    xx = []
    psi = []

    # --------------------- Set initial points for psi ---------------------
    # Set the initial x position for the scan to be 2 units
    # away from where the guess E intersects with V(x)
    xx.append(-abs(np.sqrt(guess))-2.)
    # Assume that 2 units away is far enough such that psi is 0 there
    psi.append(0.)

    xx.append(xx[0]+s)
    psi.append(0.0001)
    # ----------------------------------------------------------------------

    next_x = xx[0]+2.*s

    # Parse over the entire relevant domain
    while next_x < abs(np.sqrt(guess))+2.:
        xx.append(next_x)

        # Evaluate psi at the current value of x
        psi.append(evaluate_psi(guess, s, xx, psi))

        next_x += s

    xx = np.asarray(xx)
    psi = np.asarray(psi)*get_norm_fact(xx,psi) # Normalize psi

    return xx, psi

def initial_scan(lowerE, upperE, dguess, s, Efuncs):
    "Find all eigenvalues within he range provided"
    current_guess = lowerE
    old_psi = []

    while current_guess <= upperE:
        xx, psi = compute_psi_of_x(current_guess, s)

        if len(old_psi) < 1:
            old_psi = psi

        # The tail end of psi has crossed the origin between iterations, which means
        # we have crossed an eigenfunction
        if (psi[-1] > 0 and old_psi[-1] < 0) or (psi[-1] < 0 and old_psi[-1] > 0):
            print(f"Found eigenfunction between E = {current_guess-dguess} and "
                  f"{current_guess} and psi[-1] = {psi[-1]} and {old_psi[-1]}")
            Efuncs.append(efunc(current_guess, current_guess-dguess))

        current_guess += dguess
        old_psi = psi

def scan(lowerE, upperE, dguess, s, round_to=11):
    # Round the guess to a high decimal place because to truncate for example 5.256000000000014
    # That extra 1.4E-14 makes it so that the loop may not be inclusive on both siddes of the
    # bounds
    current_guess = round(lowerE, round_to)
    old_psi = []

    while round(current_guess, round_to) <= round(upperE, round_to):
        xx, psi = compute_psi_of_x(current_guess, s)

        if len(old_psi) < 1:
            old_psi = psi

        # The tail end of psi has crossed the origin between iterations, which means
        # we have crossed an eigenfunction.
        if (psi[-1] > 0 and old_psi[-1] < 0) or (psi[-1] < 0 and old_psi[-1] > 0):
            print(f"Found eigenfunction between E = {current_guess-dguess} and "
                  f"{current_guess} and psi[-1] = {psi[-1]} and {old_psi[-1]}")
            return current_guess, current_guess-dguess, psi[-1], old_psi[-1]

        current_guess += dguess
        old_psi = psi

    print("Exiting scan erroneously. No eigenfunction found in provided limits.")
    print("Check that the rounding decimal place in this function is smaller ")
    print("than the tolerance (preferably by a few orders of magnitude).")

def check_all_converged(Efuncs, tolerance):
    for index, eigenfunc in enumerate(Efuncs):
        if not eigenfunc.converged(tolerance):
            print(f"Function {index} not converged. Terminating convergence check.")
            return False
        print(f"Function {index} converged.")
    return True

def set_all_eigenfuncs(s, Efuncs):
    for eigenfunc in Efuncs:
        xx, psi = compute_psi_of_x(eigenfunc.upperE, s)
        eigenfunc.set_psi(psi)
        eigenfunc.set_x(xx)

def plt_all_eigenfuncs(x, V, Efuncs):
    plt.plot(x, V)

    for eigenfunc in Efuncs:
        plt.plot(eigenfunc.x, eigenfunc.psi+eigenfunc.upperE)

    plt.show()


if __name__ == '__main__':
    x = np.arange(-4.,4.,0.001)
    V = V_of_x(x, 'quadratic')

    s = 0.001
    dguess = 0.25
    tolerance = 0.000001 # Psi must be smaller than this at the extreme end to be converged

    # Lowest energy to look at
    lowerE = float(sys.argv[1])
    current_guess = lowerE

    # Highest energy to look at
    upperE = float(sys.argv[2])

    Efuncs = [] # A list of Eigenfunction objects
    all_converged = False

    counter = 0
    while not all_converged:
        if len(Efuncs) == 0:
            initial_scan(lowerE, upperE, dguess, s, Efuncs)
            start = time.time()

        else:
            # Increase density of guesses 10x within range off interest to get closer
            # to the true Eigenvalue
            dguess /= 10.
            for index, eigenfunc in enumerate(Efuncs):
                # counter < 2 is necessary b/c set_bounds need to be called at least
                # once before eigenfunc.converged() can even run (it needs boundaries
                # to check). The "or" allows it to ignore the 2nd condition if the
                # first is true.
                if counter < 2 or not eigenfunc.converged(tolerance):
                    upperE, lowerE, upper_tail, lower_tail = scan(
                        eigenfunc.lowerE, eigenfunc.upperE, dguess, s
                    )
                    eigenfunc.set_bounds(upperE, lowerE, upper_tail, lower_tail)

            all_converged = check_all_converged(Efuncs, tolerance)
        counter += 1
    print(f"All functions converged in {time.time()-start} s")

    set_all_eigenfuncs(s, Efuncs)
    plt_all_eigenfuncs(x, V, Efuncs)
