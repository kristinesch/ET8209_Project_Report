import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from helpers import *
import numpy as np
import random

"""Select method: (uncomment which method to run)"""
option = "SDP"
#option = "SDDP"


"""Input data"""

#Constants
C = 3600/1e6 #conversion from m^3/s to Mm^3
E = 0.981 #MWh/m^3
P_max = 100 #MW
Q_max = 100 #m^3/s
V_0 = 5 #Mm^3
V_max = 10 #Mm^3
WV = 13000 #EUR/Mm^3
f_1 = 50 #m^3/s
pi_omega = 0.2

def rho(t):
    return 50 + t

def f_2(omega):
    return 25 * omega

#sets
T_1 = range(1,25)
T_2 = range(25, 49)
T_3 = range(49, 73)
Omega = range(0,5)


"""Model building"""

def build_model_3_stage():
    MP = pyo.ConcreteModel()
    SP = pyo.ConcreteModel()

    #Sets (vector indices)
    MP.T_1 = pyo.Set(initialize = T_1) 
    SP.T_2 = pyo.Set(initialize = T_2)
    SP.Omega = pyo.Set(initialize = Omega)

    #Master problem
    MP.p_1 = pyo.Var(MP.T_1, within=pyo.NonNegativeReals, bounds=(0, P_max)) #production
    MP.q_1 = pyo.Var(MP.T_1, within=pyo.NonNegativeReals, bounds=(0, Q_max)) #discharge
    MP.sigma_1 = pyo.Var(MP.T_1, within=pyo.NonNegativeReals, bounds=(0, V_max)) #spillage
    MP.v_1 = pyo.Var(MP.T_1, within = pyo.NonNegativeReals, bounds=(0, V_max)) #rsv vol
    MP.v_1_last = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, V_max))
    MP.alpha = pyo.Var(within=pyo.NonNegativeReals, bounds = (0, 1e6))

    #Stage 2
    SP.p_2 = pyo.Var(SP.Omega, SP.T_2, within=pyo.NonNegativeReals, bounds=(0, P_max))
    SP.q_2 = pyo.Var(SP.Omega, SP.T_2, within=pyo.NonNegativeReals, bounds=(0, Q_max)) 
    SP.sigma_2 = pyo.Var(SP.Omega, SP.T_2, within=pyo.NonNegativeReals, bounds=(0, V_max)) 
    SP.v_2 = pyo.Var(SP.Omega, SP.T_2, within = pyo.NonNegativeReals, bounds=(0, V_max))
    SP.v_2_init = pyo.Var(within = pyo.NonNegativeReals, bounds=(0, V_max))
    SP.beta = pyo.Var(SP.Omega, within = pyo.NonNegativeReals, bounds = (0, 1e6))
    SP.v_2_last = pyo.Var(SP.Omega, within = pyo.NonNegativeReals, bounds=(0, V_max))

    #Stage 3 /Final stage
    SP2 = pyo.ConcreteModel()
    SP2.T_3 = pyo.Set(initialize = T_3)
    SP2.Omega = pyo.Set(initialize = Omega)
    SP2.p_3 = pyo.Var(SP2.Omega, SP2.T_3, within=pyo.NonNegativeReals, bounds=(0, P_max))
    SP2.q_3 = pyo.Var(SP2.Omega, SP2.T_3, within=pyo.NonNegativeReals, bounds=(0, Q_max)) 
    SP2.sigma_3 = pyo.Var(SP2.Omega, SP2.T_3, within=pyo.NonNegativeReals, bounds=(0, V_max)) 
    SP2.v_3 = pyo.Var(SP2.Omega, SP2.T_3, within = pyo.NonNegativeReals, bounds=(0, V_max))
    SP2.v_3_init = pyo.Var(within = pyo.NonNegativeReals, bounds=(0, V_max))

    #Add constraints
    add_constraints_stage_1(MP)
    add_constraints_stage_2(SP)
    add_volume_coupling_2last(SP)
    add_constraints_stage_3(SP2)

    #Add objectives
    add_mp_objective(MP)
    add_sp_objective_multistage(SP)
    add_sp2_objective_multistage(SP2)
    
    return MP, SP, SP2


"""Constraints"""

def add_constraints_stage_1(model):

    def mass_conservation_1(model, t): 
        if t == model.T_1.first():
            return model.v_1[t] == V_0 + C*(f_1 - model.q_1[t] - model.sigma_1[t])
        return model.v_1[t] == model.v_1[t-1] + C*(f_1 - model.q_1[t] - model.sigma_1[t])
    model.mass_conservation = pyo.Constraint(model.T_1, rule = mass_conservation_1)

    def volume_coupling_1last(model):
        return model.v_1_last == model.v_1[model.T_1.last()]
    model.volume_coupling_1last = pyo.Constraint(rule = volume_coupling_1last)

    def pq_relation_1(model, t):
        return model.p_1[t] == E*model.q_1[t]
    model.pq_relation_1 = pyo.Constraint(model.T_1, rule = pq_relation_1)

def add_volume_coupling_1last_2init(model): #sp
    def volume_coupling_1last_2init(model):
        return model.v_2_init == model.v_1_last
    model.volume_coupling_1last_2init = pyo.Constraint(rule = volume_coupling_1last_2init)

def add_volume_coupling_2last(model): #sp
    def volume_coupling_2last(model, omega):
        return model.v_2_last[omega] == model.v_2[omega, model.T_2.last()]
    model.volume_coupling_2last = pyo.Constraint(model.Omega, rule = volume_coupling_2last)

def add_constraints_stage_2(model):

    def mass_conservation_2(model, omega, t): 
        if t == model.T_2.first():
            return model.v_2[omega, t] == model.v_2_init + C*(f_2(omega) - model.q_2[omega, t] - model.sigma_2[omega, t])
        return model.v_2[omega,t] == model.v_2[omega, t-1] + C*(f_2(omega) - model.q_2[omega, t] - model.sigma_2[omega, t])  
    model.mass_conservation_2 = pyo.Constraint(model.Omega, model.T_2, rule = mass_conservation_2)

    def pq_relation_2(model, omega, t):
        return model.p_2[omega, t] == E*model.q_2[omega, t]
    model.pq_relation_2 = pyo.Constraint(model.Omega, model.T_2, rule = pq_relation_2)

def add_volume_coupling_2last_3init(model): #sp2
    def volume_coupling_2last_3init(model):
        return model.v_3_init == model.v_2_last
    model.volume_coupling_2last_3init = pyo.Constraint(rule = volume_coupling_2last_3init)

def add_constraints_stage_3(model):

    def mass_conservation_3(model, omega, t): 
        if t == model.T_3.first():
            return model.v_3[omega, t] == model.v_3_init + C*(f_2(omega) - model.q_3[omega, t] - model.sigma_3[omega, t])
        return model.v_3[omega, t] == model.v_3[omega, t-1] + C*(f_2(omega) - model.q_3[omega,  t] - model.sigma_3[omega, t])  
    model.mass_conservation_3 = pyo.Constraint(model.Omega,  model.T_3, rule = mass_conservation_3)

    def pq_relation_3(model, omega,t):
        return model.p_3[omega, t] == E*model.q_3[omega, t]
    model.pq_relation_3 = pyo.Constraint(model.Omega, model.T_3, rule = pq_relation_3)

"""Objectives"""

def add_mp_objective(model):
    def objective(model):
        return sum(rho(t)*model.p_1[t] for t in model.T_1) + model.alpha
    model.objective = pyo.Objective(rule=objective, sense = pyo.maximize)

def add_sp_objective_multistage(model):
    def objective(model):
        return sum(((pi_omega*sum(rho(t)*model.p_2[omega,t] for t in model.T_2)) + pi_omega*model.beta[omega] for omega in model.Omega)) #piomega*beta?
    model.objective = pyo.Objective(rule=objective, sense = pyo.maximize)

def add_sp2_objective_multistage(model):
    def objective(model):
        stage_3 = sum((pi_omega*sum(rho(t)*model.p_3[omega,  t] for t in model.T_3)) for omega in model.Omega)
        wv = sum((pi_omega*WV*model.v_3[omega, model.T_3.last()]) for omega in model.Omega)
        return stage_3 + wv
    model.objective = pyo.Objective(rule=objective, sense = pyo.maximize)


"""Functions for solving models"""

def solve_model(model):
    opt = pyo.SolverFactory('gurobi')
    opt.solve(model) 
    return pyo.value(model.objective)

def solve_mp(MP):
    obj = solve_model(MP)
    mp_end_vol = MP.v_1_last.value
    return obj, mp_end_vol

def solve_sp(SP, mp_end_vol):

    #remove old values
    if hasattr(SP, 'v_1_last'):
        SP.del_component(SP.v_1_last)
        SP.del_component(SP.volume_coupling_1last_2init)
        SP.del_component(SP.dual)

    #update values
    SP.v_1_last = pyo.Param(initialize = mp_end_vol)
    add_volume_coupling_1last_2init(SP)
    SP.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)

    obj = solve_model(SP)
    return obj


def solve_sp2(SP2, v):

    #remove old values
    if hasattr(SP2, 'v_2_last'):
        SP2.del_component(SP2.v_2_last)
        SP2.del_component(SP2.volume_coupling_2last_3init)
        SP2.del_component(SP2.dual)

    #update values
    SP2.v_2_last = pyo.Param(initialize = v)
    add_volume_coupling_2last_3init(SP2)
    SP2.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)

    obj = solve_model(SP2)
    return obj 

def final_forward_pass(MP, SP, SP2):
    print("Forward pass final")
    mp_obj, mp_end_vol = solve_mp(MP)
    v1=extract_results(MP.v_1)
    v2={}
    v3={}
    sp_obj = solve_sp(SP, mp_end_vol)
    for omega in Omega:
        v2[omega] = convert_to_array(extract_results(SP.v_2)["v_2"])[omega]
        v = SP.v_2_last[omega].value
        sp_obj_stage_3 = solve_sp2(SP2, v) 
        for omega2 in Omega:
            v3[omega, omega2] = convert_to_array(extract_results(SP2.v_3)["v_3"], stage = 3)[omega2]
    print("\nObjective: ", mp_obj)
    return mp_obj, v1, v2, v3


"""Solution algorithms"""

def solve_multistage_SDP(MP, SP, SP2, guesses = 10):
    v_guesses = np.linspace(0,V_max,guesses)
    MP.cuts = pyo.ConstraintList()
    SP.cuts = pyo.ConstraintList()

    print("Stage 3")
    for i, v in enumerate(v_guesses):
        print(f'calculating for v {v}...')
        sp2_obj = solve_sp2(SP2, v)
        dual = SP2.dual.get(SP2.volume_coupling_2last_3init)
        print("dual: ", dual, " obj: ", sp2_obj)
        for omega in Omega:
            SP.cuts.add(SP.beta[omega] <= sp2_obj + dual * SP.v_2[omega, SP.T_2.last()] - dual * v)

    print("Stage 2")  
    for i, v in enumerate(v_guesses):   
        print(f'calculating for v {v}...')   
        sp_obj = solve_sp(SP, v)
        dual = SP.dual.get(SP.volume_coupling_1last_2init)
        print("dual: ", dual, " obj: ", sp_obj)
        MP.cuts.add(MP.alpha <= sp_obj + dual * MP.v_1[MP.T_1.last()] - dual * v)

    obj, v1, v2, v3 =final_forward_pass(MP, SP, SP2)
    return obj, v1, v2, v3


def solve_SDDP(MP, SP, SP2, i_max = 20, tol = 0.1):
    MP.cuts = pyo.ConstraintList()
    SP.cuts = pyo.ConstraintList()
    mp_obj, mp_end_vol = solve_mp(MP)
    for i in range(i_max):

        #Forward pass
        print("Forward pass ", i+1)
        omega = random.choice(Omega)
        sp_obj = solve_sp(SP, mp_end_vol)
        sp_end_vol = SP.v_2_last[omega].value
        beta = SP.beta[omega].value
        omega_2 = random.choice(Omega)
        sp2_obj = solve_sp2(SP2, sp_end_vol)
        sp2_end_vol = SP2.v_3[omega_2, SP2.T_3.last()]

        #Calculate UB
        UB = mp_obj
        LB = mp_obj - MP.alpha.value + sp_obj - beta + sp2_obj
        print("UB = ", mp_obj)
        print("LB = ", LB)

        #Backward pass
        print("Backward pass ", i+1)
        for omega in Omega:
            dual = SP2.dual.get(SP2.volume_coupling_2last_3init)
            SP.cuts.add(SP.beta[omega] <= sp2_obj + dual * SP.v_2[omega, SP.T_2.last()] - dual * sp_end_vol)
        sp_obj = solve_sp(SP, mp_end_vol)
        dual = SP.dual.get(SP.volume_coupling_1last_2init)
        MP.cuts.add(MP.alpha <= sp_obj + dual * MP.v_1[MP.T_1.last()] - dual * mp_end_vol)
        mp_obj, mp_end_vol = solve_mp(MP)
        print("objective:", mp_obj)
        
        if UB-LB <= tol:
            print("Convergence has been reached")
            break

    #Forward pass
    obj, v1, v2, v3 = final_forward_pass(MP, SP, SP2)
    return obj, v1, v2, v3



