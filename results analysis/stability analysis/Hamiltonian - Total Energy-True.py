import numpy as np
import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))
import utils.funciones as fns   

st_view = "True"

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


# Data
start_val = 6_400_000
years_val = 800
val_steps = int(years_val / 0.0005) + 1
fin_val = start_val + val_steps

with np.load("Data/Z.npz", mmap_mode='r') as data:
    Zt = data['z'][start_val:fin_val, ...].astype('float64')

r = Zt[:, :30].reshape(-1, 10, 3)
p = Zt[:, 30:].reshape(-1, 10, 3)

H_total_planetas = fns.Hamiltoniano_potencial(r, p, mass, G_const, "all")

np.savez_compressed(f"results analysis/stability analysis/{st_view}/Total_Energy.npz", H=H_total_planetas)
