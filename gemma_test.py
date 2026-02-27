from nesy_factory.language_model.train import GemmaTrainer
from nesy_factory.language_model.tokenize import ByteLevelBPETokenizer
from nesy_factory.language_model.text_exporter import TextExporter
from nesy_factory.language_model.gemma import Gemma3Builder


meta = TextExporter().run(
    dataset="roneneldan/TinyStories",
    split="train",
    output_file="/home/gaian/Desktop/git_lm/train.txt",
    max_rows=100,
    streaming=True,
)
print("OK → wrote", meta["exported_rows"], "lines to", meta["output_file"])


summary = Gemma3Builder().run(
    tokenizer_json="/home/gaian/Desktop/git_lm/tokenizer.json",
    n_layers=6,
    layer_pattern="S*3,F*1,S*2",          
    model_weights_out="/home/gaian/Desktop/git_lm/artifacts/gemma3.pt",
    model_config_out="/home/gaian/Desktop/git_lm/artifacts/gemma3_config.json",
    model_py_out="/home/gaian/Desktop/git_lm/artifacts/gemma3_model.py",
)
print(summary)

tok = ByteLevelBPETokenizer()

report = tok.run(
    text_file="/home/gaian/Desktop/git_lm/train.txt",  
    vocab_size=32000,
    min_frequency=2,
    special_tokens="[PAD],[UNK],[BOS],[EOS]",
    add_bos_eos=True,
)
print("Training report:", report)

# Save tokenizer JSON
tok.save("/home/gaian/Desktop/git_lm/tokenizer.json")
tok = ByteLevelBPETokenizer()
tok.load("/home/gaian/Desktop/git_lm/tokenizer.json") 
ids = tok.encode("hello world")
text = tok.decode(ids)
print(ids, "\n", text)

# Encode/decode quick test
ids = tok.encode("Hello world!")
print("IDs:", ids)
print("Decoded:", tok.decode(ids))


trainer = GemmaTrainer()

trainer.run(
    tokenizer_json="/home/gaian/Desktop/git_lm/tokenizer.json",
    train_corpus="/home/gaian/Desktop/git_lm/train.txt",
    model_config="/home/gaian/Desktop/git_lm/artifacts/gemma3_config.json",
    model_weights="/home/gaian/Desktop/git_lm/artifacts/gemma3.pt",
    model_py_in="/home/gaian/Desktop/git_lm/artifacts/gemma3_model.py",
    model_py_out=".../artifacts/gemma3_model.py",
    learning_rate=1e-4, min_lr=1e-5, warmup_steps=1,
    max_iters=6, batch_size=8, block_size=64, grad_accum=1,
    eval_interval=1, eval_iters=5, weight_decay=0.1, beta2=0.95,
    clip_grad_norm=0.5, val_fraction=0.1, num_proc=6,
    best_weights="/home/gaian/Desktop/git_lm/artifacts/best.pt",
    final_weights="/home/gaian/Desktop/git_lm/artifacts/final.pt",
    training_report="/home/gaian/Desktop/git_lm/artifacts/training_report.json",
    loss_curve_csv="/home/gaian/Desktop/git_lm/artifacts/loss_curve.csv",
)