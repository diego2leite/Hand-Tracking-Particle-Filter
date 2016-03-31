import numpy
import pyparticleest.interfaces as interfaces
import matplotlib.pyplot as plt
import pyparticleest.simulator as simulator
try:
    import pyparticleest.utils.ckalman as kalman
except ImportError:
    print("Falling back to pure python implementaton, expect horrible performance")
    import pyparticleest.utils.kalman as kalman

def generate_dataset(steps, P0, Q, R):
    x = numpy.zeros((steps + 1,))
    y = numpy.zeros((steps,))
    x[0] = 2.0 + 0.0 * numpy.random.normal(0.0, P0)
    for k in range(1, steps + 1):
        x[k] = x[k - 1] + numpy.random.normal(0.0, Q)
        y[k - 1] = x[k] + numpy.random.normal(0.0, R)

    return (x, y)

class Integrator(interfaces.ParticleFiltering):
    # x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
    # y_k = x_k + e_k, e_k ~ N(0,R),
    # x(0) ~ N(0,P0) """

    def __init__(self, P0, Q, R):
        # Define model parameters
        self.P0 = numpy.copy(P0)
        self.Q = numpy.copy(Q)
        self.R = numpy.copy(R)

    def create_initial_estimate(self, N):
        # Create an initial estimate for time 0
        return numpy.random.normal(0.0, self.P0, (N,)
                                   ).reshape((-1, 1))

    def sample_process_noise(self, particles, u, t):
        # Sampled the process noise for each particle
        N = len(particles)
        return numpy.random.normal(0.0, self.Q, (N,)
                                   ).reshape((-1, 1))

    def update(self, particles, u, t, noise):
        # Calculate the estimates for the next time step,
        # using the sampled process noise
        particles += noise

    def measure(self, particles, y, t):
        # Return the log-pdf value of the measurement
        logyprob = numpy.empty(len(particles))
        for k in range(len(particles)):
            logyprob[k] = kalman.lognormpdf(particles[k, 0] - y,
                                            self.R)
        return logyprob


if __name__ == '__main__':
    steps = 50
    num = 50
    P0 = 1.0
    Q = 1.0
    R = numpy.asarray(((1.0,),))

    # Make realization deterministic
    numpy.random.seed(1)
    (x, y) = generate_dataset(steps, P0, Q, R)

    model = Integrator(P0, Q, R)
    # No input signals (u) for this model class
    sim = simulator.Simulator(model, u=None, y=y)

    # Use 'num' forward particles, and create 'num'
    # backward trajectories by taking the ancestral
    # paths for the end-time particles
    sim.simulate(num, num, smoother='ancestor')

    # Plot true state trajectory
    plt.plot(range(steps + 1), x, 'r-')

    # Plot measurements
    plt.plot(range(1, steps + 1), y, 'bx')

    # Get the filtered particle estimates
    (vals, _) = sim.get_filtered_estimates()

    # Plot filtered estimates
    plt.plot(range(steps + 1), vals[:, :, 0],
             'k.', markersize=0.8)

    # Get "smoothed" ancestral trajectories
    svals = sim.get_smoothed_estimates()

    # Plot particles ancestral trajectories to
    # illustrate that the particle filter suffers
    # from degeneracy when estimating the smoothing
    # distribution
    plt.plot(range(steps + 1), svals[:, :, 0], 'b--')
    plt.xlabel('t')
    plt.ylabel('x')

    plt.show()
