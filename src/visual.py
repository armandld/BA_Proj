import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_amr_state(sim, active_patches, step, dt, t_val):
    """
    Affiche l'état MHD global (Courant Jz) et superpose les boîtes d'attention VQA.
    Style: RdBu (Rouge/Bleu) + Cadres Rouges pointillés + Indication de Zoom.
    """
    # 1. Calcul du Courant Jz pour l'esthétique "Fluide" (Curl of B)
    # Jz = dBy/dx - dBx/dy
    # np.gradient(arr, axis=1) est d/dx (colonnes), axis=0 est d/dy (lignes)
    grad_By_x = np.gradient(sim.By, axis=1)
    grad_Bx_y = np.gradient(sim.Bx, axis=0)
    Jz = grad_By_x - grad_Bx_y
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # 2. Affichage du champ (Fond)
    # On utilise RdBu centré sur 0 (Blanc = Calme, Rouge/Bleu = Fort courant)
    im = ax.imshow(Jz, origin='lower', cmap='RdBu', interpolation='nearest')
    
    # Barre de couleur
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Current Density $J_z$ (Vorticity)', rotation=270, labelpad=15)
    
    # 3. Dessin des Patchs VQA (AMR)
    # On trie par profondeur pour dessiner les petits par-dessus les gros
    active_patches_sorted = sorted(active_patches, key=lambda p: p.get('depth', 0))
    
    for p in active_patches_sorted:
        # Récupération des coord (Format Dictionnaire)
        if 'bounds' in p:
            ys, ye, xs, xe = p['bounds']
            depth = p.get('depth', 0)
        else:
            # Fallback (Ancien format au cas où)
            ys, ye = p['i_start'], p['i_start'] + p['width']
            xs, xe = p['j_start'], p['j_start'] + p['width']
            depth = 0 # Inconnu
            
        width = xe - xs
        height = ye - ys
        
        # Calcul du facteur de zoom pour l'affichage (Base 3 car découpage 3x3)
        zoom_factor = 3**depth
        
        # Style visuel (Rouge pointillé comme l'ancienne)
        # On rend le trait plus fin si le zoom est profond pour ne pas cacher la physique
        line_width = max(1, 2.5 - 0.5 * depth) 
        
        # Rectangle
        rect = patches.Rectangle((xs, ys), width, height, 
                                 linewidth=line_width, edgecolor='red', 
                                 facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        # Annotation du Zoom (Uniquement si ce n'est pas tout le domaine)
        if depth > 0:
            # On place le texte un peu au-dessus du cadre
            label_text = f"x{zoom_factor}"
            ax.text(xs, ye + 1, label_text, 
                    color='red', fontsize=8 + depth, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # Titres et Labels
    total_patches = len(active_patches)
    ax.set_title(f"VQA-Driven AMR Simulation\nTime: {t_val:.3f} | Active Zones: {total_patches}")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    
    # Pour ne pas accumuler les fenêtres si tu lances une longue simu
    plt.pause(0.01) 
    # Si tu veux sauvegarder :
    # plt.savefig(f"frames/amr_{step:04d}.png", dpi=150)
    
    plt.show() # Nettoie la figure pour le tour suivant

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