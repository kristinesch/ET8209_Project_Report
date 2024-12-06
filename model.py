import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from helpers import *
import numpy as np


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

def build_model(option, scenario = None, multistage = False):
    
    if option == "basic":
        model = pyo.ConcreteModel()

        #Sets (vector indices)
        model.T_1 = pyo.Set(initialize = T_1) 
        model.T_2 = pyo.Set(initialize = T_2)
        if scenario is not None:
            model.Omega = pyo.Set(initialize = [scenario])
        else:
            model.Omega = pyo.Set(initialize = Omega)

        #variables
        model.p_1 = pyo.Var(model.T_1, within=pyo.NonNegativeReals, bounds=(0, P_max)) #production
        model.p_2 = pyo.Var(model.Omega, model.T_2, within=pyo.NonNegativeReals, bounds=(0, P_max))
        model.q_1 = pyo.Var(model.T_1, within=pyo.NonNegativeReals, bounds=(0, Q_max)) #discharge
        model.q_2 = pyo.Var(model.Omega, model.T_2, within=pyo.NonNegativeReals, bounds=(0, Q_max)) 
        model.sigma_1 = pyo.Var(model.T_1, within=pyo.NonNegativeReals, bounds=(0, V_max)) #spillage
        model.sigma_2 = pyo.Var(model.Omega, model.T_2, within=pyo.NonNegativeReals, bounds=(0, V_max)) 
        model.v_1 = pyo.Var(model.T_1, within = pyo.NonNegativeReals, bounds=(0, V_max)) #rsv vol
        model.v_2 = pyo.Var(model.Omega, model.T_2, within = pyo.NonNegativeReals, bounds=(0, V_max))
        model.v_1_last = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, V_max))

        if multistage: #add third stage
            model.Omega_2 = pyo.Set(initialize = Omega)
            model.T_3 = pyo.Set(initialize = T_3)
            model.p_3 = pyo.Var(model.Omega, model.Omega_2, model.T_3, within=pyo.NonNegativeReals, bounds=(0, P_max)) #production
            model.q_3 = pyo.Var(model.Omega, model.Omega_2, model.T_3, within=pyo.NonNegativeReals, bounds=(0, Q_max)) #discharge
            model.sigma_3 = pyo.Var(model.Omega, model.Omega_2, model.T_3, within=pyo.NonNegativeReals, bounds=(0, V_max)) #spillage
            model.v_3 = pyo.Var(model.Omega, model.Omega_2, model.T_3, within = pyo.NonNegativeReals, bounds=(0, V_max)) #rsv vol
            model.v_2_last = pyo.Var(model.Omega, within=pyo.NonNegativeReals, bounds=(0, V_max))
    
    if option == "benders" or option == "SDP" or option == "SDDP":
        MP = pyo.ConcreteModel()
        SP = pyo.ConcreteModel()

        #Sets (vector indices)
        MP.T_1 = pyo.Set(initialize = T_1) 
        SP.T_2 = pyo.Set(initialize = T_2)

        if scenario is not None:
            SP.Omega = pyo.Set(initialize = [scenario])
        else:
            SP.Omega = pyo.Set(initialize = Omega)

        #variables
        MP.p_1 = pyo.Var(MP.T_1, within=pyo.NonNegativeReals, bounds=(0, P_max)) #production
        MP.q_1 = pyo.Var(MP.T_1, within=pyo.NonNegativeReals, bounds=(0, Q_max)) #discharge
        MP.sigma_1 = pyo.Var(MP.T_1, within=pyo.NonNegativeReals, bounds=(0, V_max)) #spillage
        MP.v_1 = pyo.Var(MP.T_1, within = pyo.NonNegativeReals, bounds=(0, V_max)) #rsv vol
        MP.v_1_last = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, V_max))
        MP.alpha = pyo.Var(within=pyo.NonNegativeReals, bounds = (0, 1e6))

        SP.p_2 = pyo.Var(SP.Omega, SP.T_2, within=pyo.NonNegativeReals, bounds=(0, P_max))
        SP.q_2 = pyo.Var(SP.Omega, SP.T_2, within=pyo.NonNegativeReals, bounds=(0, Q_max)) 
        SP.sigma_2 = pyo.Var(SP.Omega, SP.T_2, within=pyo.NonNegativeReals, bounds=(0, V_max)) 
        SP.v_2 = pyo.Var(SP.Omega, SP.T_2, within = pyo.NonNegativeReals, bounds=(0, V_max))
        SP.v_2_init = pyo.Var(within = pyo.NonNegativeReals, bounds=(0, V_max))

    if option == "basic":
        add_constraints_stage_1(model)
        add_constraints_stage_2(model, option)
        if scenario is None and not multistage:
            add_objective_basic(model)
        if scenario is not None and not multistage:
            add_objective_deterministic(model)
        if multistage:
            add_constraints_stage_3(model)
            add_objective_basic_multistage(model)
            add_volume_coupling_2last(model)
        return model

    elif option == "benders" or option == "SDP":
        add_constraints_stage_1(MP)
        add_constraints_stage_2(SP, option)
        add_mp_objective(MP)
        if scenario is None:
            add_sp_objective(SP)
        else: #if scenario is not none, deterministic
            add_sp_objective_deterministic(SP)
        return MP, SP

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

def add_constraints_stage_2(model, option):

    if option == "basic":
        def mass_conservation_2(model, omega, t): 
            if t == model.T_2.first():
                return model.v_2[omega, t] == model.v_1_last + C*(f_2(omega) - model.q_2[omega, t] - model.sigma_2[omega, t])
            return model.v_2[omega,t] == model.v_2[omega, t-1] + C*(f_2(omega) - model.q_2[omega, t] - model.sigma_2[omega, t])  
        model.mass_conservation_2 = pyo.Constraint(model.Omega, model.T_2, rule = mass_conservation_2)

    elif option == "benders" or option == "SDP":
        def mass_conservation_2(model, omega, t): 
            if t == model.T_2.first():
                return model.v_2[omega, t] == model.v_2_init + C*(f_2(omega) - model.q_2[omega, t] - model.sigma_2[omega, t])
            return model.v_2[omega,t] == model.v_2[omega, t-1] + C*(f_2(omega) - model.q_2[omega, t] - model.sigma_2[omega, t])  
        model.mass_conservation_2 = pyo.Constraint(model.Omega, model.T_2, rule = mass_conservation_2)

    def pq_relation_2(model, omega, t):
        return model.p_2[omega, t] == E*model.q_2[omega, t]
    model.pq_relation_2 = pyo.Constraint(model.Omega, model.T_2, rule = pq_relation_2)

def add_constraints_stage_3(model):

    def mass_conservation_3(model, omega, omega_2, t): 
        if t == model.T_3.first():
            return model.v_3[omega, omega_2, t] == model.v_2_last[omega] + C*(f_2(omega_2) - model.q_3[omega, omega_2, t] - model.sigma_3[omega, omega_2, t])
        return model.v_3[omega, omega_2, t] == model.v_3[omega, omega_2, t-1] + C*(f_2(omega_2) - model.q_3[omega, omega_2, t] - model.sigma_3[omega, omega_2, t])  
    model.mass_conservation_3 = pyo.Constraint(model.Omega, model.Omega_2, model.T_3, rule = mass_conservation_3)

    def pq_relation_3(model, omega, omega_2, t):
        return model.p_3[omega, omega_2, t] == E*model.q_3[omega, omega_2, t]
    model.pq_relation_3 = pyo.Constraint(model.Omega, model.Omega_2, model.T_3, rule = pq_relation_3)

def add_objective_basic(model):
    def objective(model):
        stage_1 = sum(rho(t)*model.p_1[t] for t in model.T_1) 
        stage_2 = sum(((pi_omega*sum(rho(t)*model.p_2[omega,t] for t in model.T_2) + pi_omega*WV*model.v_2[omega, model.T_2.last()]) for omega in model.Omega))
        return stage_1 + stage_2
    model.objective = pyo.Objective(rule=objective, sense = pyo.maximize)

def add_objective_deterministic(model):
    def objective(model):
        stage_1 = sum(rho(t)*model.p_1[t] for t in model.T_1) 
        stage_2 = sum(((sum(rho(t)*model.p_2[omega,t] for t in model.T_2) + WV*model.v_2[omega, model.T_2.last()]) for omega in model.Omega))
        return stage_1 + stage_2
    model.objective = pyo.Objective(rule=objective, sense = pyo.maximize)

def add_objective_basic_multistage(model):
    def objective(model):
        stage_1 = sum(rho(t)*model.p_1[t] for t in model.T_1) 
        stage_2 = sum((pi_omega*sum(rho(t)*model.p_2[omega, t] for t in model.T_2)) for omega in model.Omega)
        stage_3 = sum((pi_omega*sum((pi_omega*sum(rho(t)*model.p_3[omega, omega_2, t] for t in model.T_3)) for omega in model.Omega)) for omega_2 in model.Omega_2)
        wv = sum(pi_omega*(sum((pi_omega*WV*model.v_3[omega, omega_2, model.T_3.last()]) for omega in model.Omega)) for omega_2 in model.Omega_2)
        return stage_1 + stage_2 + stage_3 +wv
    model.objective = pyo.Objective(rule=objective, sense = pyo.maximize)

def add_mp_objective(model):
    def objective(model):
        return sum(rho(t)*model.p_1[t] for t in model.T_1) + model.alpha
    model.objective = pyo.Objective(rule=objective, sense = pyo.maximize)

def add_mp_objective_initial(model):
    def objective(model):
        return sum(rho(t)*model.p_1[t] for t in model.T_1)
    model.objective = pyo.Objective(rule=objective, sense = pyo.maximize)

def add_sp_objective(model):
    def objective(model):
        return sum(((pi_omega*sum(rho(t)*model.p_2[omega,t] for t in model.T_2) + pi_omega*WV*model.v_2[omega, model.T_2.last()]) for omega in model.Omega))
    model.objective = pyo.Objective(rule=objective, sense = pyo.maximize)

def add_sp_objective_deterministic(model):
    def objective(model):
        return sum(((sum(rho(t)*model.p_2[omega,t] for t in model.T_2) + WV*model.v_2[omega, model.T_2.last()]) for omega in model.Omega))
    model.objective = pyo.Objective(rule=objective, sense = pyo.maximize)

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

def solve_benders(MP, SP, i_max = 10, tol = 0.1):
    cuts={}
    MP.cuts = pyo.ConstraintList()
    mp_obj, mp_end_vol = solve_mp(MP)
    for i in range(i_max):
        sp_obj = solve_sp(SP, mp_end_vol)
        dual = SP.dual.get(SP.volume_coupling_1last_2init)
        MP.cuts.add(MP.alpha <= sp_obj + dual * MP.v_1[MP.T_1.last()] - dual * mp_end_vol)
        cuts[i]=(dual, sp_obj - dual * MP.v_1_last.value)
        LB = mp_obj - MP.alpha.value + sp_obj
        print(f'LB: {LB}')

        mp_obj, mp_end_vol = solve_mp(MP)
        UB = mp_obj
        print(f'UB: {UB}')

        if UB-LB < tol:
            print("solution has converged")
            break
        if i == i_max-1:
            print("max number of iterations has been reached")
    return mp_obj, cuts

def solve_SDP(MP, SP, guesses):
    v_guesses = np.linspace(0,V_max,guesses)
    cuts={}
    MP.cuts = pyo.ConstraintList()
    for i, v in enumerate(v_guesses):
        sp_obj = solve_sp(SP, v)
        dual = SP.dual.get(SP.volume_coupling_1last_2init)
        MP.cuts.add(MP.alpha <= sp_obj + dual * MP.v_1[MP.T_1.last()] - dual * v)
        cuts[i]=(dual, sp_obj - dual * v)
    mp_obj, mp_end_vol = solve_mp(MP)
    sp_obj = solve_sp(SP, mp_end_vol)
    return mp_obj, cuts



