# Stage-wise independent finite discrete problem
def construct_nvid():
    from msppy.msp import MSLP
    nvid = MSLP(T=2, sense=-1, bound=20)
    for t in range(2):
        m = nvid[t]
        if t == 0:
            buy_now, _ = m.addStateVar(name='bought', obj=-1.0)
        else:
            _, buy_past = m.addStateVar(name='bought')
            sold = m.addVar(name='sold', obj=2)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=0.5)
            m.addConstr(sold + unsatisfied == 0,
                uncertainty={'rhs':range(11)})
            m.addConstr(sold + recycled == buy_past)
    return nvid

# Stage-wise independent continuous problem
def construct_nvic():
    from msppy.msp import MSLP
    import numpy as np
    nvic = MSLP(T=2, sense=-1, bound=100)
    def f(random_state):
        return random_state.lognormal(mean=np.log(4),sigma=2)
    for t in range(2):
        m = nvic[t]
        buy_now, buy_past = m.addStateVar(name='bought', obj=-1.0)
        if t == 1:
            sold = m.addVar(name='sold', obj=2)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=0.5)
            m.addConstr(sold + unsatisfied == 0, uncertainty={'rhs':f})
            m.addConstr(sold + recycled == buy_past)
    return nvic

# Stage-wise independent finite discrete risk averse problem
def construct_nvida():
    from msppy.msp import MSLP
    import numpy as np
    nvica = MSLP(T=2, sense=-1, bound=100)
    for t in range(2):
        m = nvica[t]
        buy_now, buy_past = m.addStateVar(name='bought', obj=-1.0)
        if t == 1:
            sold = m.addVar(name='sold', obj=2)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=0.5)
            m.addConstr(sold + unsatisfied == 0,
                uncertainty={'rhs':range(11)})
            m.addConstr(sold + recycled == buy_past)
    nvica.set_AVaR(l=0.5, a=0.1, method='direct')
    return nvica

# Stage-wise independent finite discrete integer problem
def construct_nvidi():
    from msppy.msp import MSIP
    import numpy as np
    nvidi = MSIP(T=2, sense=-1, bound=100)
    for t in range(2):
        m = nvidi[t]
        buy_now, buy_past = m.addStateVar(name='bought', obj=-1.0, vtype='I')
        if t == 1:
            sold = m.addVar(name='sold', obj=2, vtype='I')
            unsatisfied = m.addVar(name='unsatisfied', vtype='I')
            recycled = m.addVar(name='recycled', obj=0.5, vtype='I')
            m.addConstr(sold + unsatisfied == 0,
                uncertainty={'rhs':range(11)})
            m.addConstr(sold + recycled == buy_past)
    return nvidi

# Stage-wise independent finite continuous integer problem
def construct_nvici():
    from msppy.msp import MSIP
    import numpy as np
    nvici = MSIP(T=2, sense=-1, bound=100)
    def f(random_state):
        return random_state.choice(np.arange(100))    
    for t in range(2):
        m = nvici[t]
        buy_now, buy_past = m.addStateVar(name='bought', obj=-1.0, vtype='I')
        if t == 1:
            sold = m.addVar(name='sold', obj=2, vtype='I')
            unsatisfied = m.addVar(name='unsatisfied', vtype='I')
            recycled = m.addVar(name='recycled', obj=0.5, vtype='I')
            m.addConstr(sold + unsatisfied == 0, uncertainty={'rhs':f})
            m.addConstr(sold + recycled == buy_past)
    return nvici

# Stage-wise independent finite discrete infinite horizon problem
def construct_nvidinf():
    from msppy.msp import MSLP
    rhs = [[0,2,4,6,8],[1,3,5,7,9]]
    nvidinf = MSLP(T=3, discount=0.9, sense=-1, bound=20, infinity=3)
    for t in range(3):
        m = nvidinf[t]
        if t == 0:
            buy_now, _ = m.addStateVar(name='bought', obj=-1.0)
        else:
            _, buy_past = m.addStateVar(name='bought')
            sold = m.addVar(name='sold', obj=2)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=0.5)
            m.addConstr(sold + unsatisfied == 0,
                uncertainty={'rhs':rhs[t-1]})
            m.addConstr(sold + recycled == buy_past)
    return nvidinf

# Stage-wise independent finite discrete infinite horizon integer problem
def construct_nvidinfi():
    from msppy.msp import MSIP
    rhs = [[0,2,4,6,8],[1,3,5,7,9]]
    nvidinfi = MSIP(T=3, discount=0.9, sense=-1, bound=20, infinity=3)
    for t in range(3):
        m = nvidinfi[t]
        if t == 0:
            buy_now, _ = m.addStateVar(name='bought', obj=-1.0, vtype='I')
        else:
            _, buy_past = m.addStateVar(name='bought', vtype='I')
            sold = m.addVar(name='sold', obj=2, vtype='I')
            unsatisfied = m.addVar(name='unsatisfied', vtype='I')
            recycled = m.addVar(name='recycled', obj=0.5, vtype='I')
            m.addConstr(sold + unsatisfied == 0,
                uncertainty={'rhs':rhs[t-1]})
            m.addConstr(sold + recycled == buy_past)
    return nvidinfi

# Markov chain problem
def construct_nvmc():
    from msppy.msp import MSLP
    nvmc = MSLP(T=3, sense=-1, bound=100)
    nvmc.add_MC_uncertainty(
        Markov_states=[[[0]],[[4],[6]],[[4],[6]]],
        transition_matrix=[
            [[1]],
            [[0.5,0.5]],
            [[0.3,0.7],[0.7,0.3]]
        ]
    )
    for t in range(3):
        m = nvmc[t]
        buy_now, buy_past = m.addStateVar(name='bought', obj=-1.0)
        if t != 0:
            sold = m.addVar(name='sold', obj=2)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=0.5)
            m.addConstr(sold + unsatisfied == 0,
                uncertainty_dependent={'rhs':0})
            m.addConstr(sold + recycled == buy_past)
    return nvmc

# Markovian continuous problem
def construct_nvm():
    from msppy.msp import MSLP
    import numpy as np
    nvm = MSLP(T=3, sense=-1, bound=500)
    def sample_path_generator(random_state, size):
        a = np.zeros([size,3,1])
        for t in range(1,3):
            a[:,t,:] = (0.5 * a[:,t-1,:]
                + random_state.lognormal(2.5,1,size=[size,1]))
        return a
    nvm.add_Markovian_uncertainty(sample_path_generator)
    for t in range(3):
        m = nvm[t]
        buy_now, buy_past = m.addStateVar(name='bought', obj=-1.0)
        if t != 0:
            sold = m.addVar(name='sold', obj=2)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=0.5)
            m.addConstr(sold + unsatisfied == 0, uncertainty_dependent={'rhs':0})
            m.addConstr(sold + recycled == buy_past)
    return nvm
