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
# 2. EXTRACTION ET FILTRAGE DE L'ENVELOPPE HAUTE (VALEURS BRUTES AME)
# ==========================================================
donnees_brutes_par_A = {}

with open(file_path, "r") as f:
    start = False
    for line in f:
        if "MASS LIST" in line:
            start = True
            continue
        if not start or len(line) < 100 or line.strip().startswith('.'):
            continue
        try:
            N_str = line[4:9].strip()
            Z_str = line[9:14].strip()
            A_str = line[14:19].strip()
            BA_str = line[54:67].strip()
            
            if not N_str or not Z_str or not A_str or not BA_str or '*' in BA_str:
                continue
            
            n_val = int(N_str)
            z_val = int(Z_str)
            a_val = int(A_str)
            ba_brut_mev = float(BA_str.replace("#", "")) / 1000.0

            if 2 <= a_val <= VALMAX_CALCULEE_A:
                if a_val not in donnees_brutes_par_A or ba_brut_mev > donnees_brutes_par_A[a_val][2]:
                    donnees_brutes_par_A[a_val] = (n_val, z_val, ba_brut_mev)
        except ValueError:
            continue

A_exp = np.array(sorted(donnees_brutes_par_A.keys()))
N_exp = np.array([donnees_brutes_par_A[a][0] for a in A_exp])
Z_exp = np.array([donnees_brutes_par_A[a][1] for a in A_exp])
BA_brut_exp = np.array([donnees_brutes_par_A[a][2] for a in A_exp])

# ==========================================================
# 3. AFFICHAGE DU TABLEAU INTERMÉDIAIRE & CORRECTION ÉLECTROMAGNÉTIQUE
# ==========================================================
titre_tableau = "TABLEAU INTERMEDIAIRE : VALEURS BRUTES EXTRAITES D'AME2020 (ENVELOPPE HAUTE)"
print("\n" + "=" * 80)
print(f"{titre_tableau:^80}")
print("=" * 80)
print(f"{'N (Neutrons)':<15} | {'Z (Protons)':<15} | {'A (Masse)':<15} | {'B/A Brut (MeV)':<20}")
print("-" * 80)
for n, z, a, ba in zip(N_exp, Z_exp, A_exp, BA_brut_exp):
    print(f"{n:<15d} | {z:<15d} | {a:<15d} | {ba:<20.4f}")
print("=" * 80 + "\n")

ac = 0.72
coulomb_correction = ac * (Z_exp * (Z_exp - 1)) / (A_exp ** (4/3))
BA_exp_mev = BA_brut_exp + coulomb_correction

# ==========================================================
# 4. MOTEUR DE SIMULATION : INITIALISATION & RELAXATION
# ==========================================================
RP_VAL = 0.877

def simuler_noyau_complet(A, rp=RP_VAL, steps=NB_STEPS):
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
# 5. BOUCLE PRINCIPALE DE CALCUL
# ==========================================================
A_points = np.arange(2, VALMAX_CALCULEE_A + 1, dtype=int)

print(f"Exécution de la simulation pour {len(A_points)} noyaux (NB_STEPS = {NB_STEPS})...")
E1_points, Ey_points = [], []
rmean_points, invr_points, pairs_points = [], [], []

for a in A_points:
    e1, ey, rm, irm, npairs = simuler_noyau_complet(a, rp=RP_VAL, steps=NB_STEPS)
    E1_points.append(e1)
    Ey_points.append(ey)
    rmean_points.append(rm)
    invr_points.append(irm)
    pairs_points.append(npairs)

    if a % 10 == 0 or a == 2:
        print(f" -> [A = {a:3d} / {VALMAX_CALCULEE_A}] État relaxé atteint")

E1_points = np.array(E1_points)
Ey_points = np.array(Ey_points)
rmean_points = np.array(rmean_points)
invr_points = np.array(invr_points)
pairs_points = np.array(pairs_points)

mask_exp = np.isin(A_exp, A_points)
A_exp_fit = A_exp[mask_exp]
BA_exp_mev_fit = BA_exp_mev[mask_exp]

# ==========================================================
# 6. OPTIMISATION ET TRACÉ GRAPHIQUE
# ==========================================================
plt.figure(1, figsize=(9, 6))

# Points AME épaissis de 15% (s = 22 * 1.15 = 25.3 -> arrondi à 25)
plt.scatter(A_exp, BA_exp_mev, color="blue", s=25, alpha=0.5)

gamma1 = 9.05
def fonction_fit_g1(A_in, K1):
    idx = [np.where(A_points == int(a))[0][0] for a in A_in if a in A_points]
    return K1 * E1_points[idx] / (A_in * (A_in + gamma1))

popt1, _ = curve_fit(fonction_fit_g1, A_exp_fit, BA_exp_mev_fit, p0=[100.0])
K1_g1 = popt1[0]

# Modèle avec eta = 9.05 tracé en vert
plt.scatter(A_points, K1_g1 * E1_points / (A_points * (A_points + gamma1)), 
            color="green", s=10, zorder=5)

def optimiser_gamma_enveloppe(A_data, BA_data, gamma_range):
    A_sub = A_data[np.isin(A_data, A_points)]
    BA_sub = BA_data[np.isin(A_data, A_points)]
    E1_sub = E1_points[[np.where(A_points == int(a))[0][0] for a in A_sub]]

    best_gamma = gamma_range[0]
    best_K1 = 0.0
    min_chi2 = float('inf')

    for g in gamma_range:
        model_unit = E1_sub / (A_sub * (A_sub + g))
        denom = np.sum(model_unit**2)
        if denom == 0:
            continue
        K1_test = np.sum(BA_sub * model_unit) / denom
        
        pred = K1_test * model_unit
        chi2 = np.sum((BA_sub - pred)**2)

        if chi2 < min_chi2:
            min_chi2 = chi2
            best_gamma = g
            best_K1 = K1_test

    return best_gamma, best_K1

gamma_grid_haute = np.linspace(1.0, 50.0, 491)
gamma2_opt, K1_g2 = optimiser_gamma_enveloppe(A_exp_fit, BA_exp_mev_fit, gamma_grid_haute)

plt.scatter(A_points, K1_g2 * E1_points / (A_points * (A_points + gamma2_opt)), 
            color="red", s=12, zorder=6)

def fonction_fit_yukawa(A_in, K_y):
    idx = [np.where(A_points == int(a))[0][0] for a in A_in if a in A_points]
    return K_y * Ey_points[idx] / A_in

popt_y, _ = curve_fit(fonction_fit_yukawa, A_exp_fit, BA_exp_mev_fit, p0=[100.0])
K_y_val = popt_y[0]

# Modèle Yukawa tracé en noir
plt.scatter(A_points, K_y_val * Ey_points / A_points, 
            color="black", s=10, zorder=5)

# ==========================================================
# 7. AFFICHAGE DES RÉSULTATS ET SYNTHÈSE DES PARAMÈTRES
# ==========================================================
print("\n" + "=" * 60)
print("RÉCAPITULATIF DES PARAMÈTRES DES MODÈLES")
print("=" * 60)
print(f"1. Modèle Environnant (Courbe Verte) :")
print(f"   - Paramètre η = {gamma1:.2f}")
print(f"   - Constante K1 = {K1_g1:.4f} MeV·fm")
print("-" * 60)
print(f"2. Modèle Environnant Fit Enveloppe Haute (Courbe Rouge) :")
print(f"   - Paramètre optimal γ = {gamma2_opt:.3f}")
print(f"   - Constante K1 = {K1_g2:.4f} MeV·fm")
print("-" * 60)
print(f"3. Modèle Yukawa (Courbe Noire) :")
print(f"   - Rayon r_y = r_p = {RP_VAL} fm")
print(f"   - Constante K1 (K_y) = {K_y_val:.4f} MeV·fm")
print("=" * 60 + "\n")

plt.xlabel("Mass number (A)", fontsize=26)
plt.ylabel("Strong binding energy per nucleon B/A (MeV)", fontsize=26)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("Curves_PureStrong_UpperEnvelope.png", dpi=300)
plt.show()
