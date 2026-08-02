# Paquet V4 : reponse experimentale a l'audit scientifique (blocs P0/P1).
#
# V1 (src/)      = pipeline de production, lecture seule.
# V2 (study/)    = etude de falsification, phases 1..13, lecture seule.
# V3 (study/v3/) = couche d'evaluation reparee (protocole pre-enregistre).
# V4 (study/v4/) = tests manquants identifies par l'audit :
#                  attribution quantique, statistiques confirmatoires,
#                  equivariance, ablations causales, validation numerique.
#
# Regle de continuite : aucun symbole de V1/V2/V3 n'est redefini ici ;
# tout ce qui existe est importe (solveur, mappers, build_ising_terms,
# exact_diag, QAOA, metriques CE(b), bootstrap trajectoire).
