'''
Created on Nov 11, 2013

@author: Jerker Nordh
'''

import numpy
import math
import pyparticleest.models.mlnlg as mlnlg
import pyparticleest.simulator as simulator

C_theta = numpy.array([[ 0.0, 0.04, 0.044, 0.008], ])
def calc_Ae_fe(eta, t):
    Ae = eta / (1 + eta ** 2) * C_theta
    fe = 0.5 * eta + 25 * eta / (1 + eta ** 2) + 8 * math.cos(1.2 * t)
    return (Ae, fe)

def calc_h(eta):
    return 0.05 * eta ** 2


def generate_dataset(length):
    Az = numpy.array([[3.0, -1.691, 0.849, -0.3201],
                      [2.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.5, 0.0]])

    C = numpy.array([[0.0, 0.0, 0.0, 0.0]])

    Qe = numpy.diag([ 0.005])
    Qz = numpy.diag([ 0.01, 0.01, 0.01, 0.01])
    R = numpy.diag([0.1, ])

    e_vec = numpy.zeros((1, length + 1))
    z_vec = numpy.zeros((4, length + 1))

    e = numpy.array([[0.0, ]])
    z = numpy.zeros((4, 1))

    e_vec[:, 0] = e.ravel()
    z_vec[:, 0] = z.ravel()

    y = numpy.zeros((1, length))
    t = 0
    h = calc_h(e)

    for i in range(1, length + 1):
        (Ae, fe) = calc_Ae_fe(e, t)

        e = (fe + Ae.dot(z) +
             numpy.random.multivariate_normal(numpy.zeros((1,)),
                                              Qe))

        wz = numpy.random.multivariate_normal(numpy.zeros((4,)),
                                              Qz).ravel().reshape((-1, 1))

        z = Az.dot(z) + wz
        t = t + 1
        h = calc_h(e)
        y[:, i - 1] = (h + C.dot(z) +
                       numpy.random.multivariate_normal(numpy.zeros((1,)),
                                                        R)).ravel()
        e_vec[:, i] = e.ravel()
        z_vec[:, i] = z.ravel()

    return (y.T.tolist(), e_vec, z_vec)

class ParticleLSB(mlnlg.MixedNLGaussianMarginalizedInitialGaussian):
    def __init__(self):
        # Define all model variables
        # No uncertainty in initial state
        xi0 = numpy.zeros((1, 1))
        z0 = numpy.zeros((4, 1))
        P0 = numpy.zeros((4, 4))

        Az = numpy.array([[3.0, -1.691, 0.849, -0.3201],
                          [2.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.5, 0.0]])

        # Set these in constructor since they are
        # not time varying
        Qxi = numpy.diag([ 0.005])
        Qz = numpy.diag([ 0.01, 0.01, 0.01, 0.01])
        R = numpy.diag([0.1, ])

        super(ParticleLSB, self).__init__(xi0=xi0, z0=z0,
                                          Pz0=P0, Az=Az,
                                          R=R, Qxi=Qxi,
                                          Qz=Qz,)

    def get_nonlin_pred_dynamics(self, particles, u, t):
        # Return A_xi, f_xi and Q_xi for all particles.
        # Expected return value is a tuple, where the first
        # element is a 3-dimensional array containing
        # A_xi for all particles, the second f_xi and the
        # third Q_xi. Returning None indicates to reuse the
        # constant values set in the constructor
        tmp = numpy.vstack(particles)[:, numpy.newaxis, :]
        xi = tmp[:, :, 0]

        # Create Axi matrices using vectorized operations
        Axi = (xi / (1 + xi ** 2)).dot(C_theta)
        Axi = Axi[:, numpy.newaxis, :] # Expand to 3d array

        # Create fxi vectors using vectorized operations
        fxi = (0.5 * xi +
               25 * xi / (1 + xi ** 2) +
               8 * math.cos(1.2 * t))
        fxi = fxi[:, numpy.newaxis, :] # Expand to 3d array

        return (Axi, fxi, None)


    def get_meas_dynamics(self, particles, y, t):
        # Same concept as in get_nonlin_pred_dynamics,
        # but with (h, Az, Rz).
        # First value is the measurement value,
        # this allows for some pre-processing to
        # make sure it is in the expected format
        if (y == None):
            return (y, None, None, None)
        else:
            tmp = 0.05 * particles[:, 0] ** 2
            h = tmp[:, numpy.newaxis, numpy.newaxis]

        # Ensure that the measurement value is returned
        # as a column vector
        return (numpy.asarray(y).reshape((-1, 1)),
                None, h, None)

if __name__ == '__main__':

    num = 300
    nums = 50

    # How many steps forward in time should our simulation run
    steps = 100

    sims = 1000
    sqr_err_eta = numpy.zeros((sims, steps + 1))
    sqr_err_theta = numpy.zeros((sims, steps + 1))

    print "Simulates %d realisations of model B from Lindsten and Sch\\\"{o}n (2011)" % (sims,)
    print "Note, this will take quite some time, each iteration takes around a minute on a typical desktop computer"
    print "iteration RMSE_\\eta RMSE_\\theta"

    for k in range(sims):
        # Create reference
        numpy.random.seed(k)
        (y, e, z) = generate_dataset(steps)

        model = ParticleLSB()

        # Create an array for our particles
        sim = simulator.Simulator(model=model, u=None, y=y)
        sim.simulate(num, nums, res=0.67, filter='PF', smoother='mcmc')

        smean = sim.get_smoothed_mean()
        theta_mean = 25.0 + C_theta.dot(smean[:, 1:5].T).T
        theta = 25.0 + C_theta.dot(z.reshape((4, -1)))
        sqr_err_eta[k, :] = (smean[:, 0] - e[0, :]) ** 2
        sqr_err_theta[k, :] = (theta_mean[:, 0] - theta) ** 2

        rmse_eta = numpy.sqrt(numpy.mean(sqr_err_eta[k, :]))
        rmse_theta = numpy.sqrt(numpy.mean(sqr_err_theta[k, :]))
        print "%d %f %f" % (k, numpy.mean(rmse_eta), numpy.mean(rmse_theta))
