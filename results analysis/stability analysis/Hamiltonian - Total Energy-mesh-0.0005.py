import numpy as np
import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))
import utils.funciones as fns   

dt_view = "dt_0.0005"

base_dir = f"results analysis/Training-results-mesh-0.0005/"

G_const = 4*(np.pi)**2  #(AU^3/(M_sun*year^2)))

mass = np.array([
    1988410E24,     #Sun
    3.302E23,       #Mercury
    48.685E23,      #Venus
    5.97219E24,     #Earth
    6.4171E23,      #Mars
    18.9819E26,     #Jupiter
    5.6834E26,      #Saturn
    86.813E24,      #Uranus
    102.409E24,     #Neptune
    1.303E22        #Pluto
    ], dtype=np.float64)/1988410E24

function_group = [
    'log(cosh(x))', 
    'log1p(xtanh(x))', 
    'sqrt(x^2+1)-1', 
    'tanh(x)', 
    'xtanh(x)'
]

for i in function_group:
    r = np.load(base_dir + i + '/master_integration_800_years.npz')['r'].reshape(-1, 10, 3)
    p = np.load(base_dir + i + '/master_integration_800_years.npz')['p'].reshape(-1, 10, 3)

    H_total_planetas = fns.Hamiltoniano_potencial(r, p, mass, G_const, "all")

    np.savez_compressed(f"results analysis/stability analysis/{dt_view}/Total_Energy_{i}.npz", H=H_total_planetas)
