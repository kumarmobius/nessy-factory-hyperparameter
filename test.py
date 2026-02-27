
from pathlib import Path
import sys
import traceback
# from nesy_factory.language_model.

from nesy_factory.language_model.train import GemmaTrainer
from nesy_factory.language_model.tokenize import ByteLevelBPETokenizer
from nesy_factory.language_model.text_exporter import TextExporter
# from nesy_factory.language_model.gptneox import GPTNeoXBuilder
from nesy_factory.language_model.gemma import Gemma3Builder
# from nesy_factory.language_model.gemma3_modified import Gemma3Builder
# from gemma3_mo import Gemma3Builder 

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def exists_or_die(p: Path, why: str = ""):
    if not p.exists():
        msg = f"Expected file not found: {p}"
        if why:
            msg += f" — {why}"
        raise FileNotFoundError(msg)

def main():
    try:
        # base directories (edit if you want)
        base = Path("test_files").expanduser()
        artifacts = base / "artifacts"
        ensure_dir(base)
        ensure_dir(artifacts)

        # canonical paths
        train_txt = base / "train.txt"
        tokenizer_json = base / "tokenizer.json"
        model_weights = artifacts / "gemma3.pt"
        model_config = artifacts / "gemma3_config.json"
        model_py = artifacts / "gemma3_model.py"
        model_out = artifacts / "gemma3_model_trained.py"   # can be same as model_py
        best_pt = artifacts / "best.pt"
        final_pt = artifacts / "final.pt"
        training_report = artifacts / "training_report.json"
        loss_csv = artifacts / "loss_curve.csv"

        print("Paths being used:")
        print(" - train_txt      :", train_txt)
        print(" - tokenizer_json :", tokenizer_json)
        print(" - artifacts dir  :", artifacts)
        print()

        # 1) Export dataset -> train.txt
        print("1) Exporting dataset to train.txt (max_rows=10)...")
        meta = TextExporter().run(
            dataset="roneneldan/TinyStories",
            split="train",
            output_file=str(train_txt),
            max_rows=1000,
            streaming=True,
        )
        print("OK → wrote", meta.get("exported_rows", "??"), "lines to", meta.get("output_file", train_txt))

        # Ensure train file exists
        exists_or_die(train_txt, "Dataset export failed or path wrong")

        # 2) Train tokenizer and save JSON
        print("\n2) Training tokenizer on train.txt...")
        tok = ByteLevelBPETokenizer()
        report = tok.run(
            text_file=str(train_txt),
            vocab_size=32000,
            min_frequency=2,
            special_tokens="[PAD],[UNK],[BOS],[EOS]",
            add_bos_eos=True,
        )
        print("Tokenizer training report:", report)

        print("Saving tokenizer to:", tokenizer_json)
        tok.save(str(tokenizer_json))

        # verify tokenizer saved
        exists_or_die(tokenizer_json, "tok.save() did not create the tokenizer file")

        # quick load test
        tok2 = ByteLevelBPETokenizer()
        tok2.load(str(tokenizer_json))
        print("Tokenizer loaded OK; sample encode:", tok2.encode("Hello world"))

        # 3) Build model skeleton with GPTNeoXBuilder (now tokenizer exists)
        print("\n3) Running GPTNeoXBuilder (uses tokenizer_json)...")
        summary = Gemma3Builder().run(
            tokenizer_json=str(tokenizer_json),
            n_layers=6,
            layer_pattern="S*3,F*1,S*2",
            model_weights_out=str(model_weights),
            model_config_out=str(model_config),
            model_py_out=str(model_py),
        )
        print("GPTNeoXBuilder result:", summary)

        # verify builder produced model_py and config (weights may or may not be populated)
        exists_or_die(model_py, "GPTNeoXBuilder should have written the model .py file")
        exists_or_die(model_config, "GPTNeoXBuilder should have written the model config JSON")

        # 4) Trainer: use the generated model_py as model_py_in
        print("\n4) Starting GemmaTrainer.run() using the generated model_py file...")
        trainer = GemmaTrainer()

        # final safety printout
        print("Pre-train existence checks:")
        print(" - tokenizer_json exists:", tokenizer_json.exists())
        print(" - train_txt exists:", train_txt.exists())
        print(" - model_py exists:", model_py.exists())
        print(" - model_config exists:", model_config.exists())

        # run trainer
        trainer.run(
            tokenizer_json=str(tokenizer_json),
            train_corpus=str(train_txt),
            model_config=str(model_config),
            model_weights=str(model_weights),
            model_py_in=str(model_py),        # <-- use generated model .py as input
            model_py_out=str(model_out),       # overwrite or keep same as desired
            learning_rate=1e-4, min_lr=1e-5, warmup_steps=1,
            max_iters=5, batch_size=8, block_size=64, grad_accum=1,
            eval_interval=5, eval_iters=1, weight_decay=0.1, beta2=0.95,
            clip_grad_norm=0.5, val_fraction=0.1, num_proc=5,
            best_weights=str(best_pt),
            final_weights=str(final_pt),
            training_report=str(training_report),
            loss_curve_csv=str(loss_csv),
        )

        print("\nTraining completed. Check artifacts directory for outputs.")
        print("Artifacts:", list(artifacts.iterdir()))
        import csv,json,os
        loss_csv_path = str(loss_csv)
        schema_json_path = os.path.join(os.path.dirname(loss_csv_path), "schema_json.json")

        schema_data = []

        with open(loss_csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                schema_data.append({
                    "epoch": int(row["update"]),
                    "loss": float(row["train_loss"]),
                    "validation_loss": float(row["val_loss"])
                })

        with open(schema_json_path, "w") as jsonfile:
            json.dump(schema_data, jsonfile, indent=2)

        print(f"✅ schema_json saved to {schema_json_path} ({len(schema_data)} entries)")

    except Exception as e:
        print("\nERROR — caught exception:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
