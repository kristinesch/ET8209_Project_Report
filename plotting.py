from matplotlib import pyplot as plt 
import numpy as np
from helpers import *



def plot_q_and_v(v_1, v_2, q_1, q_2, Omega, T_1, T_2, title):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(T_1, v_1.values, label = "$v_{1,t}$", color = "darkblue")
    ax2.plot(T_1, q_1.values, color = "darkred")
    v_2_arr = convert_to_array(v_2["v_2"])
    q_2_arr = convert_to_array(q_2["q_2"])
    if len(Omega) == 1:
        o = Omega[0]
        ax.plot([24]+list(T_2), v_1.values[-1].tolist()+v_2_arr[o].tolist(), label = '$v_{2,t}$', color = plt.colormaps["Blues"](100))
        ax2.plot([24]+list(T_2), q_1.values[-1].tolist()+q_2_arr[o].tolist(), label = '$q_{2,t}$', color = plt.colormaps["Reds"](100))
    else:
        for o in Omega:
            ax.plot([24]+list(T_2), v_1.values[-1].tolist()+v_2_arr[o].tolist(), label = '$v_{2,t}$, $\omega =$'+ str(o), color = plt.colormaps["Blues"](50+o*50))
            ax2.plot([24]+list(T_2), q_1.values[-1].tolist()+q_2_arr[o].tolist(), color = plt.colormaps["Reds"](50+o*50))
        #for legend only...
        ax.plot([],[], label = "$q_{1,t}$", color = "darkred") 
        for o in Omega:
            ax.plot([],[], label = '$q_{2,t}$, $\omega =$ '+ str(o), color = plt.colormaps["Reds"](50+o*50))
        ax.legend(bbox_to_anchor=(1.15, 0.9))
    fig.suptitle(title)
    ax.set_ylabel("volume [Mm^3]")
    ax2.set_ylabel("discharge [m^3/s]")
    ax.set_xlabel("time [hour]")
    fig.tight_layout()
    fig.savefig(title)
    plt.show()

def plot_cuts(cuts, title, V_max = 10):
    fig, ax = plt.subplots()
    v = np.array([v for v in range(V_max +1)])
    for i in range(len(cuts)):
        a, b = cuts[i]
        ax.plot(v, a*v + b, label = f'cut {i}')
    ax.set_xlabel("v [Mm^3]")
    ax.set_ylabel("Objective value [Euro]")
    fig.suptitle(title)
    fig.legend(loc = "right")
    fig.tight_layout()
    fig.savefig(title)
    plt.show()

def plot_v_basic_multistage(v_1, v_2, v_3, Omega, T_1, T_2, T_3, title):
    fig, ax = plt.subplots()
    ax.plot(T_1, v_1.values, label = "$v_{1,t}$", color = "darkblue")
    v_2_arr = convert_to_array(v_2["v_2"])
    
    for o in Omega:
        ax.plot([24]+list(T_2), v_1.values[-1].tolist()+v_2_arr[o].tolist(), label = '$v_{2,t}$, $omega =$'+ str(o), color = plt.colormaps["Blues"](70+o*50))
        for o2 in Omega:
            v_3_list=[]
            for t in T_3:
                v_3_list.append(v_3["v_3"][o,o2,t])
            ax.plot([48]+list(T_3), [v_2_arr[o][-1]]+v_3_list, label = '$v_{3,t}$, $omega =$'+ str(o), color = plt.colormaps["Blues"](70+o*40))

    fig.suptitle(title)
    ax.set_ylabel("volume [Mm^3]")
    ax.set_xlabel("time [hour]")
    fig.tight_layout()
    fig.savefig(title)
    plt.show()

def plot_v_multistage(v1, v2, v3, Omega, T_1, T_2, T_3, title):
    fig, ax = plt.subplots()
    ax.plot(T_1, v1.values, label = "$v_{1,t}$", color = "darkblue")
    for o in Omega:
        v2_t = v2[o]
        ax.plot([24]+list(T_2), v1.values[-1].tolist()+v2[o].tolist(), label = '$v_{2,t}$, $omega =$'+ str(o), color = plt.colormaps["Blues"](70+o*50))
        for o2 in Omega:
            v3_t = v3[o, o2]
            ax.plot([48]+list(T_3), [v2_t[-1]]+v3_t.tolist(), label = '$v_{3,t}$, $omega =$'+ str(o), color = plt.colormaps["Blues"](70+o*40))
    fig.suptitle(title)
    ax.set_ylabel("volume [Mm^3]")
    ax.set_xlabel("time [hour]")
    fig.tight_layout()
    fig.savefig(title)
    plt.show()