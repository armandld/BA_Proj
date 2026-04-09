import optuna
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Upload Local SQLite to Neon")
    parser.add_argument("--train-dir", default="../Train_results", help="Directory where local DBs are")
    parser.add_argument("--in-url", default="", help="Neon URL")
    parser.add_argument("--LocalToNeon", action="store_true", help="Upload from Local to Neon instead of Neon to Local")
    parser.add_argument("--ResetNeon", action="store_true", help="Reset Neon data instead of importing")
    parser.add_argument("--ResetLocal", action="store_true", help="Reset local data instead of importing")
    args = parser.parse_args()

    # Destination : Ton PostgreSQL Neon
    if args.in_url == "":
        neon_url = "postgresql://neondb_owner:npg_osTe7ENJpZz5@ep-patient-hall-abitnl4g-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    else:
        neon_url = args.in_url

    studies = ["q_has_v2_phase1", "q_has_v2_phase1_agr", "q_has_v2_phase1b", "q_has_v2_phase1b_agr",
               "q_has_v2_phase2", "q_has_v2_phase2_agr", "q_has_v2_phase3",
               "classical_v2_phase1", "classical_v2_phase2","classical_v2_phase3"]

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
                else:
                    to_storage = local_url
                try:
                    optuna.delete_study(study_name=study_name, storage=to_storage)

                    if to_storage == neon_url:
                        print(f"🗑️  Existing study '{study_name}' deleted in Neon.")
                    elif to_storage == local_url:
                        print(f"🗑️  Existing local study '{study_name}' deleted.")
                except KeyError:
                    # L'étude n'existait pas encore, c'est parfait
                    pass

                if to_storage == local_url:
                    from_storage = neon_url
                elif to_storage == neon_url:
                    from_storage = local_url
                
                local_study = optuna.load_study(study_name=study_name, storage=from_storage)
                
                optuna.copy_study(
                    from_study_name=study_name,
                    from_storage=from_storage,
                    to_storage=to_storage,
                    to_study_name=study_name,
                )
                
                print(f"🚀 Successfully uploaded {study_name} to Neon: {len(local_study.trials)} trials synchronized.")

        except Exception as e:
            print(f"❌ Error with {study_name}: {e}")

if __name__ == "__main__":
    main()