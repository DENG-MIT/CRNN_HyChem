import sys, os
import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Lhs

import cantera as ct

np.random.seed(1)

gas = ct.Solution('../../mech/JP10skeletal.cti')
i_var = [gas.species_index(s) for s in [
    "C10H16", "H", "H2", "CH3", "CH4", "aC3H5", "C2H4", "C3H6", "C5H6", "C6H6", "C6H5CH3", "N2",
    "OH", "H2O", "O2", "HO2", "O", "H2O2"]]

# for i in range(3,232):
#     gas.set_multiplier(i, 0)

lhs = Lhs(lhs_type="classic", criterion=None)
space = Space([(1100., 1400.), (1., 5.), ('0', '1', '2', '3', '4')])

nsamples = 10
x = lhs.generate(space.dimensions, nsamples)

comp = ['C10H16:0.01,n2:0.99',
        'C10H16:0.02,n2:0.98',
        'C10H16:0.01,o2:0.014,h2o:0.1,n2:0.9',
        'C10H16:0.01,o2:0.028,h2o:0.1,n2:0.9',
        'C10H16:0.01,o2:0.007,h2o:0.1,n2:0.9']

for i in range(1):

    gas.TPX = x[i][0], x[i][1] * ct.one_atm, comp[np.int16(x[i][2])]
    
    Y_fuel_0 = gas.X[gas.species_index('C10H16')]

    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    time = 0.0
    states = ct.SolutionArray(gas, extra=['t'])
    
    print(x[i][0], x[i][1], comp)

    print('%10s %10s %10s %14s' % ('t [s]', 'T [K]', 'P [Pa]', 'u [J/kg]'))
    for n in range(10000):
        states.append(r.thermo.state, t=sim.time)
        # time += 1.e-4
        # sim.advance(time)
        sim.step()

        if r.thermo.X[gas.species_index('C10H16')] < Y_fuel_0 * 0.75:
            break

        # print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T,
        #                                        r.thermo.P, r.thermo.u))
        
    element = 'O'

    diagram = ct.ReactionPathDiagram(gas, element)
    diagram.title = 'Reaction path diagram following {} @ {:.1f} K {:.1f} atm {}'.format(element, x[i][0], x[i][1], comp[np.int16(x[i][2])])
    diagram.label_threshold = 0.05
    diagram.show_details = True
    
    dot_file = 'rxnpath.dot'
    img_file = 'rxnpath.png'
    img_path = os.path.join(os.getcwd(), img_file)
    
    diagram.write_dot(dot_file)
    # print(diagram.get_data())
    
    print("Wrote graphviz input file to '{0}'.".format(os.path.join(os.getcwd(), dot_file)))
    # dot .\rxnpath.dot -Tpng -o rxnpath.png -Gdpi=300
    
    os.system('dot {0} -Tpng -o{1} -Gdpi=200'.format(dot_file, img_file))
    print("Wrote graphviz output file to '{0}'.".format(img_path))