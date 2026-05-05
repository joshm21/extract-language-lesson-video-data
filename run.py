print("starting...")


def run():
    from core.pipeline import Runner
    Runner().run()


def cleanup_runs(base_data_dir="./data"):
    import shutil
    from pathlib import Path
    data_path = Path(base_data_dir)
    # Glob finds all run_ folders inside any artifacts folder
    for run_folder in data_path.glob("*/artifacts/run_*"):
        if run_folder.is_dir():
            print(f"Deleting: {run_folder}")
            shutil.rmtree(run_folder)


if __name__ == "__main__":
    # cleanup_runs()  # use this to delete all */artifacts/run_* folders
    run()
