from model import *
from model_multistage import *
from helpers import *
from plotting import *

scenario = None

"""Run code: uncomment which alternative to run"""

#option = "basic"
#scenario = 4 #for deterministic case

# option = "basic_multistage"

# option = "benders"

# option = "SDP"
n_points = 10

# option = "SDP_multistage"

option = "SDDP"



if option == "basic" and scenario is None:
    model = build_model("basic")
    solve_model(model)
    v_1 = extract_results(model.v_1)
    v_2 = extract_results(model.v_2)
    q_1 = extract_results(model.q_1)
    q_2 = extract_results(model.q_2)
    plot_q_and_v(v_1, v_2, q_1, q_2, Omega, T_1, T_2, "Discharge and volume")
    print("objective value: ", pyo.value(model.objective))

if option == "basic" and scenario is not None:
    model = build_model("basic", scenario)
    solve_model(model)
    v_1 = extract_results(model.v_1)
    v_2 = extract_results(model.v_2)
    q_1 = extract_results(model.q_1)
    q_2 = extract_results(model.q_2)
    plot_q_and_v(v_1, v_2, q_1, q_2, [scenario], T_1, T_2, f"Discharge and volume, scenario {scenario} ")
    print("objective value: ", pyo.value(model.objective))

if option == "benders" and scenario is None:
    MP, SP = build_model(option)
    obj, cuts = solve_benders(MP, SP)
    v_1 = extract_results(MP.v_1)
    v_2 = extract_results(SP.v_2)
    q_1 = extract_results(MP.q_1)
    q_2 = extract_results(SP.q_2)
    plot_q_and_v(v_1, v_2, q_1, q_2, Omega, T_1, T_2, f"Discharge and volume, Benders")
    print("objective value: ", obj)
    plot_cuts(cuts, f"Benders cuts")
    print(cuts)

if option == "benders" and scenario is not None:
    MP, SP = build_model(option, scenario)
    obj, cuts = solve_benders(MP, SP)
    v_1 = extract_results(MP.v_1)
    v_2 = extract_results(SP.v_2)
    q_1 = extract_results(MP.q_1)
    q_2 = extract_results(SP.q_2)
    plot_q_and_v(v_1, v_2, q_1, q_2, [scenario], T_1, T_2, f"Discharge and volume, Benders scenario {scenario}")
    print("objective value: ", obj)
    plot_cuts(cuts, f"Benders cuts scenario {scenario}")
    print(cuts)
    
if option == "SDP":
    MP, SP = build_model(option)
    obj, cuts = solve_SDP(MP, SP, n_points)
    v_1 = extract_results(MP.v_1)
    v_2 = extract_results(SP.v_2)
    q_1 = extract_results(MP.q_1)
    q_2 = extract_results(SP.q_2)
    plot_q_and_v(v_1, v_2, q_1, q_2, Omega, T_1, T_2, f"Discharge and volume, SDP")
    print("objective value: ", obj)
    plot_cuts(cuts, f"SDP cuts, {n_points} discrete points")
    print(cuts)

if option == "basic_multistage":
    model = build_model("basic", multistage=True)
    solve_model(model)
    v_1 = extract_results(model.v_1)
    v_2 = extract_results(model.v_2)
    v_3 = extract_results(model.v_3)
    plot_v_basic_multistage(v_1, v_2, v_3, Omega, T_1, T_2, T_3, "Volume, 3-stage")
    print("objective value: ", pyo.value(model.objective))


if option == "SDDP":
    MP, SP, SP2 = build_model_3_stage()
    mp_obj, v1, v2, v3 = solve_SDDP(MP, SP, SP2)
    plot_v_multistage(v1, v2, v3, Omega, T_1, T_2, T_3, "Volume, SDDP 3-stage")
    print("objective value: ", mp_obj)

if option == "SDP_multistage":
    MP, SP, SP2 = build_model_3_stage()
    mp_obj, v1, v2, v3 = solve_multistage_SDP(MP, SP, SP2)
    plot_v_multistage(v1, v2, v3, Omega, T_1, T_2, T_3, "Volume, SDP 3-stage")
    print("objective value: ", mp_obj)