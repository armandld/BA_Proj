# Blocages actuels

## B1 — résultats antérieurs non admissibles

Les artefacts déjà présents dans `results/` ont été produits avant plusieurs
corrections de `src` et `study`. Ils servent aux tests de compatibilité et à
l’historique, mais aucun nombre ne doit être repris dans le préprint.

Résolution : relancer le panel DNS, les analyses et la comparaison fermée sur
un commit propre, puis conserver ensemble code, environnement et sorties.

## B2 — validation finale du commit de campagne

Le commit destiné à la location n’est validé qu’après : suite rapide, suite
lente, répétition du journal Optuna et préflight des coefficients. Un seul
échec maintient ce blocage.

## B3 — absence de résultat confirmatoire actuel

Le protocole statistique est implémenté, mais les huit folds × trois
trajectoires n’ont pas encore été recalculés sur le code revu. Aucun verdict
d’avantage, de désavantage ou d’équivalence n’est donc actuellement permis.

Ces trois points sont des blocages de campagne, pas des invitations à ajuster
le protocole après observation.
