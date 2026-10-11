import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================================
# 1. CONFIGURATION GLOBALE
# ==========================================================
NB_STEPS = 300            # Nombre de pas de relaxation par noyau
VALMAX_CALCULEE_A = 250   # Taille maximale A
file_path = "./mass_1.mas20.txt"

if not os.path.exists(file_path):
    print("\n" + "!" * 80)
    print(f"ERREUR CRITIQUE : Le fichier '{file_path}' est INTROUVABLE.")
    print("!" * 80 + "\n")
    raise FileNotFoundError(f"Fichier manquant : '{file_path}'")

# ==========================================================
# 2. EXTRACTION DES DONNÉES EXPÉRIMENTALES
# ==========================================================
A_exp_all, BA_exp_all_mev = [], []
start = False

with open(file_path, "r") as f:
    for line in f:
        if "MASS LIST" in line:
            start = True
            continue
        if not start or len(line) < 100 or line.strip().startswith('.'):
            continue
        try:
            A_str = line[16:20].strip()
            BA_str = line[54:65].strip()
            if not A_str or not BA_str or '*' in BA_str:
                continue
            A_val = int(A_str)
            if A_val >= 2:
                A_exp_all.append(A_val)
                BA_exp_all_mev.append(float(BA_str.replace("#", "")) / 1000.0)
        except ValueError:
            continue

A_exp_all = np.array(A_exp_all)
BA_exp_all_mev = np.array(BA_exp_all_mev)

mask_bounds = (A_exp_all >= 2) & (A_exp_all <= VALMAX_CALCULEE_A)
A_exp = A_exp_all[mask_bounds]
BA_exp_mev = BA_exp_all_mev[mask_bounds]

# ==========================================================
# 3. MOTEUR DE SIMULATION : INITIALISATION & RELAXATION
# ==========================================================
def simuler_noyau_complet(A, rp=0.877, steps=NB_STEPS):
    if A <= 1:
        return 0.0, 0.0, 0.0, 0.0, 0

    d_min = 2.0 * rp
    kappa_physique = 1.0 / rp

    n_grid = int(np.ceil(A**(1/3))) + 3
    positions_kepler = []

    for i in range(-n_grid, n_grid + 1):
        for j in range(-n_grid, n_grid + 1):
            for k in range(-n_grid, n_grid + 1):
                x = (i + 0.5 * (j + k)) * d_min
                y = (np.sqrt(3)/2.0 * (j + k/3.0)) * d_min
                z = (np.sqrt(2.0/3.0) * k) * d_min
                positions_kepler.append([x, y, z])

    positions_kepler = np.array(positions_kepler)
    distances_centre = np.linalg.norm(positions_kepler, axis=1)
    indices_proches = np.argsort(distances_centre)[:A]

    pos = positions_kepler[indices_proches].copy()
    vel = np.zeros((A, 3))

    dt = 0.01
    damping = 0.15
    k_barriere = 1500.0

    for _ in range(steps):
        forces = np.zeros((A, 3))
        for i in range(A):
            diff = pos[i] - pos
            dist = np.linalg.norm(diff, axis=1)
            dist[i] = 1e10

            forces[i] -= np.sum(diff / (dist[:, np.newaxis]**3), axis=0)

            overlap = dist < d_min
            if np.any(overlap):
                ecart = d_min - dist[overlap]
                forces[i] += np.sum(k_barriere * ecart[:, np.newaxis] * (diff[overlap] / dist[overlap, np.newaxis]), axis=0)

        vel = (vel + forces * dt) * (1.0 - damping)
        pos += vel * dt

    matrice_dist = np.zeros((A, A))
    for i in range(A):
        diff_i = pos[i] - pos
        dist_i = np.linalg.norm(diff_i, axis=1)
        dist_i[i] = 1e10
        matrice_dist[i] = dist_i

    nb_paires = 0
    somme_r = 0.0
    E1_pure = 0.0
    E_yukawa = 0.0

    for i in range(A):
        for j in range(i):
            rij = matrice_dist[i, j]
            if rij < 1e9:
                nb_paires += 1
                somme_r += rij
                E1_pure += 1.0 / rij
                E_yukawa += np.exp(-kappa_physique * rij) / rij

    r_moy = somme_r / nb_paires if nb_paires > 0 else 0.0
    inv_r_moy = E1_pure / nb_paires if nb_paires > 0 else 0.0

    return E1_pure, E_yukawa, r_moy, inv_r_moy, nb_paires

# ==========================================================
# 4. BOUCLE PRINCIPALE DE CALCUL
# ==========================================================
A_points = np.arange(2, VALMAX_CALCULEE_A + 1, dtype=int)

print(f"\nExécution de la simulation pour {len(A_points)} noyaux (NB_STEPS = {NB_STEPS})...")
E1_points, Ey_points = [], []
rmean_points, invr_points, pairs_points = [], [], []

for a in A_points:
    e1, ey, rm, irm, npairs = simuler_noyau_complet(a, rp=0.877, steps=NB_STEPS)
    E1_points.append(e1)
    Ey_points.append(ey)
    rmean_points.append(rm)
    invr_points.append(irm)
    pairs_points.append(npairs)

    if a % 20 == 0 or a == 2:
        print(f" -> [A = {a:3d}] État relaxé atteint")

E1_points = np.array(E1_points)
Ey_points = np.array(Ey_points)
rmean_points = np.array(rmean_points)
invr_points = np.array(invr_points)
pairs_points = np.array(pairs_points)

mask_exp = np.isin(A_exp, A_points)
A_exp_fit = A_exp[mask_exp]
BA_exp_mev_fit = BA_exp_mev[mask_exp]

# ==========================================================
# 5. OPTIMISATION DE GAMMA PAR BALAYAGE SUR LES ENVELOPPES
# ==========================================================
plt.figure(1, figsize=(8, 5))
plt.scatter(A_exp, BA_exp_mev, color="blue", s=1.5, alpha=0.3)

# --- 1. Courbe Orange : gamma = 9.2 (fixe) ---
gamma1 = 9.2
def fonction_fit_g1(A_in, K1):
    idx = [np.where(A_points == int(a))[0][0] for a in A_in if a in A_points]
    return K1 * E1_points[idx] / (A_in * (A_in + gamma1))

popt1, _ = curve_fit(fonction_fit_g1, A_exp_fit, BA_exp_mev_fit, p0=[100.0])
K1_g1 = popt1[0]
plt.scatter(A_points, K1_g1 * E1_points / (A_points * (A_points + gamma1)), 
            color="orange", s=2, zorder=5)

# --- Fonction utilitaire pour optimiser gamma par grid search sur une enveloppe ---
def optimiser_gamma_enveloppe(A_data, BA_data, gamma_range):
    # Pré-calcul des indices
    idx_data = [np.where(A_points == int(a))[0][0] for a in A_data if a in A_points]
    A_sub = A_data[np.isin(A_data, A_points)]
    BA_sub = BA_data[np.isin(A_data, A_points)]
    E1_sub = E1_points[[np.where(A_points == int(a))[0][0] for a in A_sub]]

    best_gamma = gamma_range[0]
    best_K1 = 0.0
    min_chi2 = float('inf')

    for g in gamma_range:
        # Moindres carrés linéaires pour K1 : K1 = sum(y * model) / sum(model^2)
        model_unit = E1_sub / (A_sub * (A_sub + g))
        denom = np.sum(model_unit**2)
        if denom == 0:
            continue
        K1_test = np.sum(BA_sub * model_unit) / denom
        
        # Calcul du chi2 (somme des carrés des résidus)
        pred = K1_test * model_unit
        chi2 = np.sum((BA_sub - pred)**2)

        if chi2 < min_chi2:
            min_chi2 = chi2
            best_gamma = g
            best_K1 = K1_test

    return best_gamma, best_K1

# --- 2. Courbe Rouge : Optimisation de gamma sur l'enveloppe HAUTE (> 60e percentile) ---
mask_haute = BA_exp_mev > np.percentile(BA_exp_mev, 60)
gamma_grid_haute = np.linspace(1.0, 50.0, 491)
gamma2_opt, K1_g2 = optimiser_gamma_enveloppe(A_exp[mask_haute], BA_exp_mev[mask_haute], gamma_grid_haute)
print(f"Gamma optimal (enveloppe HAUTE - Rouge) : {gamma2_opt:.3f} (K1 = {K1_g2:.3f})")

plt.scatter(A_points, K1_g2 * E1_points / (A_points * (A_points + gamma2_opt)), 
            color="red", s=2, zorder=5)

# --- 3. Courbe Bleue : Optimisation de gamma sur l'enveloppe BASSE (< 40e percentile) ---
mask_basse = BA_exp_mev < np.percentile(BA_exp_mev, 40)
gamma_grid_basse = np.linspace(30.0, 150.0, 1201)
gamma3_opt, K1_g3 = optimiser_gamma_enveloppe(A_exp[mask_basse], BA_exp_mev[mask_basse], gamma_grid_basse)
print(f"Gamma optimal (enveloppe BASSE - Bleue) : {gamma3_opt:.3f} (K1 = {K1_g3:.3f})")

plt.scatter(A_points, K1_g3 * E1_points / (A_points * (A_points + gamma3_opt)), 
            color="blue", s=2, zorder=5)

# --- 4. Courbe Verte : Modèle de Yukawa discret ---
def fonction_fit_yukawa(A_in, K_y):
    idx = [np.where(A_points == int(a))[0][0] for a in A_in if a in A_points]
    return K_y * Ey_points[idx] / A_in

popt_y, _ = curve_fit(fonction_fit_yukawa, A_exp_fit, BA_exp_mev_fit, p0=[100.0])
K_y_val = popt_y[0]
plt.scatter(A_points, K_y_val * Ey_points / A_points, 
            color="green", s=2, zorder=5)

# ==========================================================
# 6. CALCUL ET DIAGNOSTIC DU PARAMÈTRE E0 (à A_ref = 50)
# ==========================================================
A_ref = 50
idx_ref = np.where(A_points == A_ref)[0][0]

mask50 = (A_exp >= 45) & (A_exp <= 55)
BA_ref = np.mean(BA_exp_mev[mask50])
E1_ref = E1_points[idx_ref]

E0_direct = BA_ref * A_ref * (A_ref + gamma2_opt) / E1_ref

print("\n" + "=" * 60)
print("DIAGNOSTIC PHYSIQUE : EXTRACTION DE E0")
print("=" * 60)
print(f"Noyau de référence A      : {A_ref}")
print(f"Énergie de liaison exp.   : {BA_ref:.6f} MeV/nucléon")
print(f"Somme simulée E1(A)      : {E1_ref:.6f} fm^-1")
print(f"Nombre total de paires    : {pairs_points[idx_ref]}")
print(f"Distance moyenne <r>      : {rmean_points[idx_ref]:.6f} fm")
print(f"Moyenne <1/r>             : {invr_points[idx_ref]:.6f} fm^-1")
print(f"Constante d'ajustement K1 : {K1_g2:.6f} MeV.fm")
print(f"Constante d'échelle E0    : {E0_direct:.6f} MeV.fm")
print("=" * 60)

plt.xlabel("Mass number (A)", fontsize=11)
plt.ylabel("Binding energy per nucleon B/A (MeV)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("Curves.png", dpi=300)
plt.show()
