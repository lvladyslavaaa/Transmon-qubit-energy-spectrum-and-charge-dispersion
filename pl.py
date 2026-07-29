import numpy as np
import matplotlib.pyplot as plt

def Hamiltonian(N=200, EJ=20, ng=0):

    phi = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    dx = phi[1] - phi[0]
    t  = 4 / dx**2
    U = -EJ*np.cos(phi) 

    main = U + 8 / dx**2
    H = np.diag(main) + np.diag(-t*np.ones(N-1), 1) + np.diag(-t*np.ones(N-1), -1)   

    H[0, -1] = -t
    H[-1, 0] = -np.conj(t)

    ng_off = -t * dx * 1j * ng *np.ones(N-1)
    ng_main = np.ones(N) * 4 * ng**2
    H_ng = np.diag(ng_main) + np.diag(ng_off, 1) + np.diag(np.conj(ng_off), -1)    

    H_ng[0,  -1] =  ng_off[0] 
    H_ng[-1,  0] = np.conj(ng_off[0]) 
    
    return H + H_ng

N, EJ, ng = 200, 20, 0
phi_centered = np.linspace(-np.pi, np.pi, N, endpoint=False)
U_centered = -EJ * np.cos(phi_centered)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

H = Hamiltonian(N=N, EJ=EJ, ng=ng)
evals, evecs = np.linalg.eigh(H)

evecs_centered = np.fft.fftshift(evecs, axes=0)
scale_psi = 10.0 

ax1.plot(phi_centered / np.pi, U_centered, 'k--', label=r'$U(\phi) = -E_J \cos(\phi)$', alpha=0.7, lw=2)

for k in range(4):
    E_k = evals[k]
    psi_k = evecs_centered[:, k].real
    
    max_idx = np.argmax(np.abs(psi_k))
    if psi_k[max_idx] < 0:
        psi_k = -psi_k
    
    ax1.axhline(E_k, color=f'C{k}', linestyle='--', alpha=0.5, label=f'$E_{k} = {E_k:.2f} E_C$')
    ax1.plot(phi_centered / np.pi, E_k + scale_psi * psi_k, color=f'C{k}', lw=2)
    ax1.fill_between(phi_centered / np.pi, E_k, E_k + scale_psi * psi_k, color=f'C{k}', alpha=0.2)

ax1.set_xlabel(r'Phase $\phi / \pi$', fontsize=12)
ax1.set_ylabel(r'Energy / $E_C$', fontsize=12)
ax1.set_title(r'Transmon energy levels & wavefunctions ($E_J/E_C = 20$)', fontsize=13, fontweight='bold')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-EJ - 3, evals[3] + 10)
ax1.legend(loc='upper right', frameon=True)

ng_vals = np.linspace(0, 1, 200)
EJ_list = [1, 5, 10, 15, 20, 50]

for EJ_val in EJ_list:
    splitting = []
    for ng_val in ng_vals:
        H_temp = Hamiltonian(N=N, EJ=EJ_val, ng=ng_val)
        evals_temp = np.linalg.eigvalsh(H_temp)
        w01 = evals_temp[1] - evals_temp[0]
        splitting.append(w01)
    ax2.plot(ng_vals, splitting, lw=2, label=f'$E_J/E_C = {EJ_val}$')

ax2.set_xlabel(r'Offset charge $n_g$', fontsize=12)
ax2.set_ylabel(r'Qubit Splitting $\omega_{01} = E_1 - E_0$ [$E_C$]', fontsize=12)
ax2.set_title(r'Qubit Splitting $\omega_{01}$ vs Offset Charge $n_g$', fontsize=13, fontweight='bold')
ax2.set_xlim(0, 1)
ax2.legend(loc='center right', frameon=True)

plt.tight_layout()
plt.savefig('transmon_figure.png', dpi=300)
plt.show()