import optuna
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Upload Local SQLite to Neon")
    parser.add_argument("--train-dir", default="../Train_results", help="Directory where local DBs are")
    parser.add_argument("--in-url", default="", help="Neon URL")
    parser.add_argument("--LocalToNeon", action="store_true", help="Upload from Local to Neon instead of Neon to Local")
    parser.add_argument("--ResetNeon", action="store_true", help="Reset Neon data instead of importing")
    parser.add_argument("--ResetLocal", action="store_true", help="Reset local data instead of importing")
    args = parser.parse_args()

    # Destination : le PostgreSQL Neon.
    #
    # L'URL complete, mot de passe compris, etait ecrite ici en dur. Ce depot
    # est PUBLIC, et elle est dans son historique git : la retirer d'ici ne
    # la retire pas de l'historique. Le mot de passe doit etre change cote
    # Neon — voir `docs/DEFAUTS.md`. Ce changement-ci ne fait qu'empecher que
    # le suivant soit publie a son tour.
    neon_url = args.in_url or os.environ.get("NEON_DB_URL", "")
    if not neon_url:
        parser.error(
            "aucune URL de base : passer --in-url, ou exporter NEON_DB_URL. "
            "Aucune valeur par defaut n'est codee ici — un identifiant en "
            "dur dans un depot public est publie a chaque clone."
        )

    studies = ["q_has_v2_phase1", "q_has_v2_phase1_agr", "q_has_v2_phase1b", "q_has_v2_phase1b_agr",
               "q_has_v2_phase2", "q_has_v2_phase2_agr", "q_has_v2_phase3",
               "classical_v2_phase1", "classical_v2_phase2","classical_v2_phase3"]

    # D-64 : un import qui n'a rien importe doit etre discernable d'un import
    # reussi. Les echecs restent per-etude — une etude absente ne doit pas
    # empecher les neuf autres — mais le processus sort non nul a la fin.
    failed = []

    for study_name in studies:
        db_path = os.path.join(args.train_dir, f"{study_name}.db")
        local_url = f"sqlite:///{db_path}"

        # 1. Vérifier si le fichier local existe
        if not os.path.exists(db_path):
            print(f"For {study_name}: No local file found at {db_path}")

        try:
            if args.ResetNeon or args.ResetLocal:
                if args.ResetNeon:
                    if study_name=="q_has_v2_phase1": 
                        optuna.delete_study(study_name=study_name, storage=neon_url)
                        print(f"🗑️  Existing study '{study_name}' deleted in Neon due to reset flag.")
                if args.ResetLocal:
                    optuna.delete_study(study_name=study_name, storage=local_url)
                    print(f"🗑️  Existing study '{study_name}' deleted in local due to reset flag.")
            else:
                if args.LocalToNeon:
                    to_storage = neon_url
                    from_storage = local_url
                else:
                    to_storage = local_url
                    from_storage = neon_url

                # D-64 : charger la SOURCE avant de supprimer la DESTINATION.
                # L'ordre inverse detruisait l'etude de destination, puis
                # rattrapait l'echec de la copie par un message et un code de
                # sortie 0. Mesure : 5 essais dans la destination, source sans
                # l'etude -> destination effacee, `code 0`. C'est la seule
                # suppression d'etude du depot, et c'est l'empreinte que
                # portent 8 des 10 bases gelees : schema complet, zero ligne,
                # et pour deux d'entre elles 274 ko / 299 ko la ou un schema
                # neuf pese 114 ko — des pages liberees, donc des lignes
                # ecrites puis supprimees.
                try:
                    local_study = optuna.load_study(study_name=study_name,
                                                    storage=from_storage)
                except KeyError:
                    # Absente de la SOURCE : il n'y a rien a importer. Ce
                    # n'est pas un echec — mais ce n'est plus une raison de
                    # supprimer la destination, qui reste intacte.
                    print(f"⏭️  {study_name}: absente de la source, "
                          f"destination laissee intacte.")
                    continue

                try:
                    optuna.delete_study(study_name=study_name, storage=to_storage)

                    if to_storage == neon_url:
                        print(f"🗑️  Existing study '{study_name}' deleted in Neon.")
                    elif to_storage == local_url:
                        print(f"🗑️  Existing local study '{study_name}' deleted.")
                except KeyError:
                    # L'étude n'existait pas encore, c'est parfait
                    pass

                optuna.copy_study(
                    from_study_name=study_name,
                    from_storage=from_storage,
                    to_storage=to_storage,
                    to_study_name=study_name,
                )
                
                print(f"🚀 Successfully uploaded {study_name} to Neon: {len(local_study.trials)} trials synchronized.")

        except Exception as e:
            print(f"❌ Error with {study_name}: {e}")
            failed.append(study_name)

    if failed:
        print(f"❌ {len(failed)}/{len(studies)} etudes en echec : "
              f"{', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()