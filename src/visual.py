import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_amr_state(sim, active_patches, t_val, target_dim, verbose = False, save_dir=None, suffix=""):
    if not verbose and save_dir is None:
        return
    """
    Affiche l'état MHD global (Courant Jz) et superpose les boîtes d'attention VQA.
    Style: RdBu (Rouge/Bleu) + Cadres Rouges pointillés + Indication de Zoom.
    """
    # 1. Calcul du Courant Jz pour l'esthétique "Fluide" (Curl of B)
    # Jz = dBy/dx - dBx/dy
    #
    # D-68 : le commentaire disait ici « axis=1 est d/dx (colonnes), axis=0
    # est d/dy (lignes) ». C'est la convention indexing='xy', qui n'est PAS
    # celle du depot. `grid.py` fait foi : AXIS_X = 0, AXIS_Y = 1, et
    # `MHDSolver.get_fluxes` forme bien Jz = dBy/dX - dBx/dY avec axis=0
    # pour X. Le tableau Jz est donc indexe [X, Y].
    _, _, _, _, Jz = sim.get_fluxes().values()

    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # 2. Affichage du champ (Fond)
    # On utilise RdBu centré sur 0 (Blanc = Calme, Rouge/Bleu = Fort courant)
    """
    max_val_Jz = np.max(np.abs(Jz)) 
    max_val_Bx = np.max(np.abs(Bx))
    max_val_By = np.max(np.abs(By))
    max_val_vx = np.max(np.abs(vx))
    max_val_vy = np.max(np.abs(vy))
    max_val = max(max_val_Jz, max_val_Bx, max_val_By, max_val_vx, max_val_vy)
    """
    # D-68, resolution : on TRANSPOSE pour mettre X en horizontal.
    #
    # `imshow` place l'axe 0 du tableau en vertical. `Jz` etant indexe
    # [X, Y], l'image non transposee portait Y en horizontal — l'inverse
    # des deux AUTRES traceurs du depot (`plot_recursive_state` ligne ~171
    # trace `state['Jz'].T`, `help_visual.plot_field` trace `grid.X.T` et
    # etiquette « X » en horizontal). Cette fonction etait la seule des
    # trois a lire dans l'autre sens.
    #
    # L'objection « cela change la geometrie de PNG deja publies » ne tient
    # plus : toutes les figures sont regenerees apres la campagne, donc la
    # coherence est gratuite maintenant et couteuse plus tard.
    im = ax.imshow(Jz.T, origin='lower', cmap='RdBu', interpolation='nearest')

    # Barre de couleur
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Current Density $J_z$ (Vorticity)", rotation=270, labelpad=15)
    
    # 3. Dessin des Patchs VQA (AMR)
    # On trie par profondeur pour dessiner les petits par-dessus les gros
    active_patches_sorted = sorted(active_patches, key=lambda p: p.get('depth', 0))
    
    for p in active_patches_sorted:
        # Récupération des coord (Format Dictionnaire)
        # Noms honnetes : `bounds` porte (i_start, i_end, j_start, j_end),
        # i indexant l'axe 0 (= X selon grid.py) et j l'axe 1 (= Y). Les
        # anciens noms `ys/ye` pour i et `xs/xe` pour j disaient l'inverse
        # de ce qu'ils contenaient — c'est cette inversion de vocabulaire
        # qui a rendu D-68 invisible pendant tout le projet.
        if 'bounds' in p:
            i_start, i_end, j_start, j_end = p['bounds']
            depth = p.get('depth', 0)
        else:
            # Fallback (Ancien format au cas où)
            i_start, i_end = p['i_start'], p['i_start'] + p['width']
            j_start, j_end = p['j_start'], p['j_start'] + p['width']
            depth = 0 # Inconnu

        # Image transposee : l'horizontal porte X (=i), le vertical Y (=j).
        width = i_end - i_start
        height = j_end - j_start
        
        # Calcul du facteur de zoom pour l'affichage (Base 3 car découpage 3x3)
        zoom_factor = target_dim**depth
        
        # Style visuel (Rouge pointillé comme l'ancienne)
        # On rend le trait plus fin si le zoom est profond pour ne pas cacher la physique
        line_width = max(1, 2.5 - 0.5 * depth) 
        
        # Rectangle
        rect = patches.Rectangle((i_start, j_start), width, height,
                                 linewidth=line_width, edgecolor='red', 
                                 facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        # Annotation du Zoom (Uniquement si ce n'est pas tout le domaine)
        if depth > 0:
            # On place le texte un peu au-dessus du cadre
            label_text = f"x{zoom_factor}"
            ax.text(i_start, j_end + 1, label_text,
                    color='red', fontsize=8 + depth, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # Titres et Labels
    total_patches = len(active_patches)
    ax.set_title(f"VQA-Driven AMR Simulation\nTime: {t_val:.3f} | Active Zones: {total_patches}")
    # D-68 clos. L'image est transposee (voir `imshow` plus haut), donc ces
    # deux etiquettes disent maintenant la verite ET s'accordent avec les
    # deux autres traceurs du depot. Mesure : une structure placee en
    # X=10, Y=40 au sens de grid.py se lisait « X=40, Y=10 » avant, et se
    # lit « X=10, Y=40 » apres. Epingle par
    # `tests/pipeline/test_amr_figure_axes.py`.
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"amr_t{t_val:.4f}_{suffix}.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        if verbose:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.pause(0.01)
        plt.show()


def plot_recursive_state(coarse_sim, fine_solvers, t, dt):
    """
    Affiche la grille grossière et les boîtes imbriquées.
    """
    state = coarse_sim.get_fluxes()
    Jz_coarse = state['Jz'].T
    N = coarse_sim.grid.N
    L = 2 * np.pi
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Fond
    X, Y = np.meshgrid(np.linspace(0, L, N+1), np.linspace(0, L, N+1))
    ax.pcolormesh(X, Y, Jz_coarse, cmap='RdBu', shading='flat', alpha=0.3)
    
    # Dessiner les boîtes
    dx_base = L / N
    
    # On trie par profondeur pour dessiner les petits par dessus les gros
    fine_solvers.sort(key=lambda s: s.meta['depth'])

    for sim in fine_solvers:
        meta = sim.meta
        depth = meta['depth']
        if depth == 0: continue # On ne redessine pas le coarse global s'il est dans la liste
        
        # Coordonnées (approximation pour la démo, suppose un mapping direct indices -> pos)
        # Dans le code complet, il faut passer les coords exactes lors de la récursion
        # Ici on utilise les clés stockées dans meta
        
        # Note: Pour que ce soit exact visuellement, il faut traquer (x,y) lors de la récursion.
        # J'ai ajouté abs_i/abs_j dans la fonction recursive_vqa_check pour ça.
        
        # Calcul position réelle (Attention: logique simplifiée pour l'exemple)
        # On suppose que abs_i réfère à l'index dans la grille grossière d'origine
        # C'est une simplification, une vraie implém QuadTree a besoin de bounds (xmin, ymin)
        
        # Pour cet affichage, on va juste montrer qu'il y a des niveaux différents
        # On ne peut pas placer parfaitement sans passer les bounds (xmin, xmax, ymin, ymax)
        # dans la récursion. *Je vais corriger la récursion ci-dessous pour inclure les bounds.*
        pass 

    # --- CORRECTION VISU ---
    # Je réintègre la logique de position dans la fonction récursive pour le plot
    # On suppose que `sim` contient les bounds dans meta
    
    ax.set_title(f"Q-HAS Recursive Depth View (t={t*dt:.2f})")
    plt.show()

# NOTE : Pour que le plot fonctionne parfaitement, je simplifie le main pour utiliser 
# une fonction de plot qui dessine juste les patchs actifs de la liste retournée.

def simple_hierarchical_plot(coarse_sim, fine_solvers, t, dt):
    L = 2 * np.pi
    N = coarse_sim.grid.N
    
    fig, ax = plt.subplots(figsize=(8, 8))
    state = coarse_sim.get_fluxes()
    ax.imshow(state['Jz'].T, origin='lower', extent=[0, L, 0, L], cmap='RdBu', alpha=0.5)
    
    # On compte les patchs par profondeur
    depth_counts = {}
    
    # Comme le tracking de position exact en récursion numpy est verbeux,
    # on va simuler l'affichage des "leafs".
    # Dans votre vraie implémentation, passez (x_min, y_min, x_max, y_max) dans recursive_vqa_check
    
    total_leafs = len(fine_solvers)
    max_depth_found = max([s.meta['depth'] for s in fine_solvers]) if fine_solvers else 0
    
    ax.set_title(f"Step {t}: {total_leafs} Active Solvers (Max Depth {max_depth_found})")
    plt.draw()
    plt.pause(0.01)
    plt.clf()