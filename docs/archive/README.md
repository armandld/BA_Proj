# Archive

Documents de campagnes antérieures à l'audit de contrat.

**Tous les nombres qu'ils contiennent sont obsolètes.** Ils ont été obtenus
sur du code dont on sait maintenant qu'il calculait autre chose que ce qu'il
annonçait, et dont le code d'étude n'était pas testé — voir `DEFAUTS.md` et
`RESULTS.md`.

Ils sont conservés parce qu'ils documentent **l'histoire du projet** : ce
qui a été cru, et quand. Ils ne documentent pas son état.

**Ne rien en citer.** Tout nombre qu'on veut réutiliser doit être remesuré
par la commande qui le produit.

Exceptions restées à la racine de `docs/` : `protocol_v3_evaluation.md` et
`protocol_deviations.md` décrivent un **protocole**, pas des résultats.

## `handoff_v4.md` — déplacé le jour de l'audit du script d'entraînement

Ce fichier vivait à la racine du dépôt et se présentait comme « the single
entry point ». Il ne l'est plus : les six documents de `docs/` se partagent
ce rôle depuis, sans recouvrement.

Il était de surcroît **activement trompeur**. Sa liste de vérification exigeait
que `git diff -- src/` soit **vide** — « V1 read-only ». C'était vrai quand
`src/` était l'objet gelé de l'étude ; ce ne l'est plus depuis que les défauts
D-6 à D-35 y ont été corrigés, chacun mesuré et verrouillé par un test. Un
relecteur qui suivrait sa checklist rejetterait exactement le travail qu'il
faut accepter.

Ses tableaux de défauts (D1–D13, numérotation **différente** de celle de
`RESULTS.md`) et ses chemins (`study/v4/RESULTS.md`,
`docs/EVALUATION_CRITIQUE.md`, `docs/CODE_REVIEW_GUIDE.md`) datent d'avant la
réorganisation.
