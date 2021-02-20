import sys
import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Lhs

import cantera as ct

np.random.seed(1)

gas = ct.Solution('../mech/JP10skeletal.cti')
i_var = [gas.species_index(s) for s in [
    "C10H16", "H2", "CH4", "C2H4", "C3H6", "C5H6", "C6H6", "C6H5CH3", "O2", "H2O"]]

lhs = Lhs(lhs_type="classic", criterion=None)
space = Space([(1100., 1400.), (1., 5.), ('0', '1', '2')])

nsamples = 10
x = lhs.generate(space.dimensions, nsamples)

# comp = ['C10H16:0.01,n2:0.99',
#         'C10H16:0.02,n2:0.98',
#         'C10H16:0.01,o2:0.014,h2o:0.1,n2:0.9',
#         'C10H16:0.01,o2:0.028,h2o:0.1,n2:0.9',
#         'C10H16:0.01,o2:0.007,h2o:0.1,n2:0.9']

comp = ['C10H16:0.01,o2:0.014,h2o:0.1,n2:0.9',
        'C10H16:0.01,o2:0.028,h2o:0.1,n2:0.9',
        'C10H16:0.01,o2:0.007,h2o:0.1,n2:0.9']

for i in range(nsamples):

    print(i, x[i][0], x[i][1], comp[np.int16(x[i][2])])

    gas.TPX = x[i][0], x[i][1] * ct.one_atm, comp[np.int16(x[i][2])]
    Y_fuel_0 = gas.Y[gas.species_index('C10H16')]

    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    time = 0.0
    states = ct.SolutionArray(gas, extra=['t'])

    # print('%10s %10s %10s %14s' % ('t [s]', 'T [K]', 'P [Pa]', 'u [J/kg]'))
    for n in range(10000):
        states.append(r.thermo.state, t=sim.time)
        # time += 1.e-4
        # sim.advance(time)
        sim.step()

        if r.thermo.Y[gas.species_index('C10H16')] < Y_fuel_0 * 0.05:
            break

        # print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T,
        #                                        r.thermo.P, r.thermo.u))

    X = states.Y[:, i_var].T
    nodedata = np.vstack((states.t, states.T, states.P, X)).T
    np.savetxt('data/data_'+str(i+1), nodedata)

    if False:
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.plot(states.t, states.T, 'o')
        plt.xlabel('Time (ms)')
        plt.ylabel('Temperature (K)')
        plt.subplot(2, 2, 2)
        plt.plot(states.t, states.X[:, gas.species_index('C10H16')])
        plt.xlabel('Time (ms)')
        plt.ylabel('C10H16 Mole Fraction')
        plt.subplot(2, 2, 3)
        plt.plot(states.t, states.X[:, gas.species_index('C2H4')])
        plt.xlabel('Time (ms)')
        plt.ylabel('C2H4 Mole Fraction')
        plt.subplot(2, 2, 4)
        plt.plot(states.t, states.X[:, gas.species_index('CH3')])
        plt.plot(states.t, states.X[:, gas.species_index('H')])
        plt.xlabel('Time (ms)')
        plt.ylabel('CH3/H Mole Fraction')
        plt.title(str(i))
        plt.tight_layout()
        plt.show()
    else:
        print("To view a plot of these results, run this script with the option --plot")
