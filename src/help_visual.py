import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

def plot_grid_topology(grid):
    """
    Affiche la position physique des nœuds de la grille.
    Permet de vérifier l'orientation (i correspond à X, j correspond à Y).
    """
    # X varie avec l'index j (colonnes), Y varie avec l'index i (lignes)
    X = grid.X.T
    Y = grid.Y.T
    plt.figure(figsize=(6, 6))
    
    # 1. Dessiner les points
    plt.scatter(X, Y, c='black', s=100/len(X[0]), zorder=3, label='Nœuds (Nodes)')
    
    # 2. Annoter quelques points pour se repérer
    # On annote (0,0), (1,0) et (0,1) pour voir les directions
    offset = grid.L * 0.02
    plt.text(X[0,0]+offset, Y[0,0]+offset, "(0,0)", color='red', fontsize=12, fontweight='bold')
    plt.text(X[1,0]+offset, Y[1,0]+offset, "(1,0)\n(+x)", color='blue', fontsize=10)
    plt.text(X[0,1]+offset, Y[0,1]+offset, "(0,1)\n(+y)", color='green', fontsize=10)
    
    # 3. Grille légère en fond
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title(f"Topologie de la Grille ({grid.N}x{grid.N})\n Convention: X=Index 0, Y=Index 1")
    plt.xlabel("Coordonnée X (Physique)")
    plt.ylabel("Coordonnée Y (Physique)")
    plt.legend()
    plt.axis('equal')
    plt.show()

def plot_flux_on_edges(grid, phi_dict, scale_arrows=True):
    """
    Affiche les flux scalaires (phi) directement sur les arêtes du graphe.
    C'est la vérité terrain de ce que voit le VQA.
    """
    # X varie avec l'index j (colonnes), Y varie avec l'index i (lignes)
    X, Y = grid.X.T, grid.Y.T
    phi_h = phi_dict['phi_horizontal']
    phi_v = phi_dict['phi_vertical']
    N = grid.N
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    segments = []
    colors = []
    
    # --- 1. Construction des Arêtes ---
    # On parcourt toute la grille pour créer les lignes
    for i in range(N):
        for j in range(N):
            # Coordonnée du point courant
            x_curr, y_curr = X[i, j], Y[i, j]
            
            # --- Arête Horizontale (vers voisin j+1) ---
            # Connecte (i,j) -> (i, j+1). Periodicité via modulo
            x_next_h, y_next_h = X[i, (j+1)%N], Y[i, (j+1)%N]
            
            # Cas spécial affichage périodique : on ne dessine pas la ligne qui traverse tout
            # si on est au bord droit, sauf si on veut voir le "loop"
            if j < N - 1: 
                segments.append([(x_curr, y_curr), (x_next_h, y_next_h)])
                colors.append(phi_h[i, j])
            else :
                dx = X[i, j] - X[i, j-1]
                segments.append([(x_curr, y_curr), (x_curr + 0.5*dx, y_curr)])
                colors.append(phi_h[i, j])
                segments.append([(x_next_h, y_next_h), (x_next_h - 0.5*dx, y_next_h)])
                colors.append(phi_h[i, j])

            
            # --- Arête Verticale (vers voisin i+1) ---
            # Connecte (i,j) -> (i+1, j)
            x_next_v, y_next_v = X[(i+1)%N, j], Y[(i+1)%N, j]
            
            if i < N - 1:
                segments.append([(x_curr, y_curr), (x_next_v, y_next_v)])
                colors.append(phi_v[i, j])
            else :
                dy = Y[i, j] - Y[i-1, j]
                segments.append([(x_curr, y_curr), (x_curr, Y[i, j] + 0.5*dy)])
                colors.append(phi_v[i, j])
                segments.append([(x_next_v, y_next_v), (x_next_v, y_next_v - 0.5*dy)])
                colors.append(phi_v[i, j])

    # --- 2. Création de la Collection de Lignes ---
    # Normalisation des couleurs pour bien voir les flux forts
    norm = plt.Normalize(vmin=0, vmax=max(np.max(phi_h), np.max(phi_v)))
    lc = LineCollection(segments, cmap='RdBu', norm=norm, linewidths=3)
    lc.set_array(np.array(colors))
    ax.add_collection(lc)
    
    # --- 3. Ajout des Nœuds par dessus ---
    ax.scatter(X, Y, c='gray', s=100/len(segments), zorder=5, alpha=0.5)
    
    # Barre de couleur
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label(r'Stress Flux $\Phi_{ij}$ (Magnitude)')
    
    ax.set_title("VQA Input: Flux on Graph Edges")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    margin = 0.6 # slightly more than 0.5 to give breathing room
    ax.set_xlim(X.min() - margin, X.max() + margin)
    ax.set_ylim(Y.min() - margin, Y.max() + margin)
    # Ajustement des limites pour tout voir
    plt.show()

def visualize_vqa_step(local_h, local_v, bounds, depth, prob_map=None):
    """
    Affiche le patch local de flux que le VQA est en train d'analyser.
    """
    # Calcul de la magnitude du flux local pour l'affichage
    # On gère les dimensions potentiellement différentes dues au slicing
    h, w = local_h.shape
    magnitude = np.sqrt(local_h**2 + local_v**2)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # 1. Affichage du Flux Local (Physique)
    # cmap='magma' pour bien voir les intensités
    im = ax.imshow(magnitude, origin='lower', cmap='magma', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Flux Stress Magnitude')
    
    # 2. Dessiner la grille 3x3 virtuelle (ce que le VQA va découper)
    step_y = h / 3
    step_x = w / 3
    for i in range(1, 3):
        # Lignes horizontales
        ax.axhline(i * step_y, color='cyan', linestyle='--', alpha=0.5)
        # Lignes verticales
        ax.axvline(i * step_x, color='cyan', linestyle='--', alpha=0.5)

    # 3. (Optionnel) Superposer les probabilités VQA si disponibles
    if prob_map is not None:
        # On affiche le score au centre de chaque case de la grille 3x3
        for i in range(3):
            for j in range(3):
                # Coordonnées du centre de la case
                center_y = (i + 0.5) * step_y
                center_x = (j + 0.5) * step_x
                score = prob_map[i, j]
                
                # Couleur du texte : Rouge si > seuil, Blanc sinon
                color = 'red' if score > 0.6 else 'white'
                weight = 'bold' if score > 0.6 else 'normal'
                
                ax.text(center_x, center_y, f"{score:.2f}", 
                        color=color, ha='center', va='center', 
                        fontsize=10, fontweight=weight, 
                        bbox=dict(facecolor='black', alpha=0.3, edgecolor='none'))

    # Titres et Infos
    y_s, y_e, x_s, x_e = bounds
    ax.set_title(f"Depth {depth} | Bounds: Y[{y_s}:{y_e}], X[{x_s}:{x_e}]\nSize: {h}x{w} px")
    
    # Affichage bloquant ou avec pause pour créer une animation
    plt.pause(0.5) # Pause de 0.5 sec pour laisser le temps de voir
    # plt.show() # Décommente si tu veux devoir fermer la fenêtre manuellement à chaque pas