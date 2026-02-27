import json
import nesy_factory
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from nesy_factory.VAE.vae import VAE
from nesy_factory.VAE.denoise_AE import DenoisingAutoencoder
from nesy_factory.VAE.maskaware_ae import MaskAwareAutoencoder
from nesy_factory.RNNs.gru import GRU
from nesy_factory.RNNs.simple_rnn import SimpleRNN
from nesy_factory.utils.helper import (
    parse_units,
    compute_vae_loss,
    compute_dae_loss,
    get_optimizer,
    load_tensor_generic,
    get_loss_fn,
    compute_rnn_loss
)


# --------------------------------------------------
# SEARCH SPACE
# --------------------------------------------------

def generate_search_space(trial, config):

    model_name = config["model_name"]

    hp = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }

    # -------- Autoencoders --------
    if model_name in ["vae", "denoise_ae", "masked_autoencoder"]:

        num_enc = trial.suggest_int("num_encoder_layers", 1, 100)
        enc_units = [
            trial.suggest_categorical(f"enc_l{i}", [32, 64, 128, 256])
            for i in range(num_enc)
        ]

        num_dec = trial.suggest_int("num_decoder_layers", 1, 100)
        dec_units = [
            trial.suggest_categorical(f"dec_l{i}", [32, 64, 128, 256])
            for i in range(num_dec)
        ]

        hp.update({
            "latent_dim": trial.suggest_int("latent_dim", 8, 128),
            "encoder_layers_units": ",".join(map(str, enc_units)),
            "decoder_layers_units": ",".join(map(str, dec_units)),
            "encoder_dropout": trial.suggest_float("encoder_dropout", 0.0, 0.25),
            "decoder_dropout": trial.suggest_float("decoder_dropout", 0.0, 0.25),
            "weight_init": trial.suggest_categorical(
                "weight_init",
                ["xavier_uniform", "kaiming_uniform", "orthogonal"]
            ),
        })

        if model_name == "masked_autoencoder":
            hp["mask_loss_weight"] = trial.suggest_float("mask_loss_weight", 0.5, 2.0)
    if model_name == "gru":

        num_layers = trial.suggest_int("num_layers", 1, 100)

        hidden_dims = [
            trial.suggest_categorical(f"hidden_l{i}", [32, 64, 128, 256])
            for i in range(num_layers)
        ]

        hp.update({
            "hidden_dims": ",".join(map(str, hidden_dims)),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
            "pooling": trial.suggest_categorical("pooling", ["last", "mean"]),
            "learn_init_hidden": trial.suggest_categorical("learn_init_hidden", [False, True]),
            "weight_init": trial.suggest_categorical(
            "weight_init", ["xavier_uniform", "kaiming_uniform", "orthogonal"]
            ),
            "loss_function": trial.suggest_categorical(
                "loss_function", ["mseloss", "maeloss", "huberloss"]
            ),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "tanh", "leaky_relu"]
            ),
            })
    if model_name == "simple_rnn":

        num_layers = trial.suggest_int("num_layers", 1, 10)

        hidden_dims = [
            trial.suggest_categorical(f"hidden_l{i}", [32, 64, 128, 256])
            for i in range(num_layers)
        ]

        hp.update({
            "hidden_dims": ",".join(map(str, hidden_dims)),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
            "pooling": trial.suggest_categorical("pooling", ["last", "mean"]),
            "learn_init_hidden": trial.suggest_categorical("learn_init_hidden", [False, True]),
            "nonlinearity": trial.suggest_categorical("nonlinearity", ["tanh", "relu"]),
            "weight_init": trial.suggest_categorical(
                "weight_init", ["xavier_uniform", "kaiming_uniform", "orthogonal"]
            ),
            "loss_function": trial.suggest_categorical(
                "loss_function", ["mseloss", "maeloss", "huberloss"]
            ),
        })
    


    return hp


# --------------------------------------------------
# OBJECTIVE
# --------------------------------------------------

def create_objective(train_tensor, val_tensor, train_target, val_target, config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model_name"]

    if model_name in ["gru", "simple_rnn"]:
        input_dim = train_tensor.shape[-1]
    else:
        input_dim = train_tensor.shape[1]

    def objective(trial):

        hp = generate_search_space(trial, config)
        trial_config = {**config, **hp}

        # -------- DataLoader --------

        if model_name in ["gru","simple_rnn", "masked_autoencoder"]:
            train_loader = DataLoader(
                TensorDataset(train_tensor, train_target),
                batch_size=trial_config["batch_size"],
                shuffle=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                TensorDataset(val_tensor, val_target),
                batch_size=trial_config["batch_size"],
                shuffle=False,
            )
        else:
            train_loader = DataLoader(
                TensorDataset(train_tensor),
                batch_size=trial_config["batch_size"],
                shuffle=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                TensorDataset(val_tensor),
                batch_size=trial_config["batch_size"],
                shuffle=False,
            )

        # -------- Model Creation --------

        if model_name == "vae":

            model = VAE(
                input_dim=input_dim,
                latent_dim=trial_config["latent_dim"],
                encoder_layers=parse_units(trial_config["encoder_layers_units"]),
                decoder_layers=parse_units(trial_config["decoder_layers_units"]),
                encoder_dropout=trial_config["encoder_dropout"],
                decoder_dropout=trial_config["decoder_dropout"],
                weight_init=trial_config["weight_init"],
            )

        elif model_name == "denoise_ae":

            model = DenoisingAutoencoder(
                input_dim=input_dim,
                hidden_dims=parse_units(trial_config["encoder_layers_units"]),
                latent_dim=trial_config["latent_dim"],
                dropout=trial_config["encoder_dropout"],
                weight_init=trial_config["weight_init"],
            )

        elif model_name == "masked_autoencoder":

            model = MaskAwareAutoencoder(
                feature_dim=input_dim,
                latent_dim=trial_config["latent_dim"],
                encoder_layers=parse_units(trial_config["encoder_layers_units"]),
                decoder_layers=parse_units(trial_config["decoder_layers_units"]),
                encoder_dropout=trial_config["encoder_dropout"],
                decoder_dropout=trial_config["decoder_dropout"],
                weight_init=trial_config["weight_init"],
            )

        elif model_name == "gru":
            hidden_dims = [int(x) for x in trial_config["hidden_dims"].split(",")]
            model = GRU(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=config["output_dim"],
                dropout=trial_config["dropout"],
                bidirectional=trial_config["bidirectional"],
                pooling=trial_config["pooling"],
                learn_init_hidden=trial_config["learn_init_hidden"],
                weight_init=trial_config["weight_init"],
                loss_function=trial_config["loss_function"],
                learning_method=trial_config["learning_method"],
            )
        elif model_name == "simple_rnn":

            hidden_dims = [int(x) for x in trial_config["hidden_dims"].split(",")]

            model = SimpleRNN(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=config["output_dim"],
                dropout=trial_config["dropout"],
                bidirectional=trial_config["bidirectional"],
                pooling=trial_config["pooling"],
                learn_init_hidden=trial_config["learn_init_hidden"],
                nonlinearity=trial_config["nonlinearity"],
                weight_init=trial_config["weight_init"],
                learning_method=trial_config["learning_method"]
            )


        else:
            raise ValueError("Unsupported model")

        model = model.to(device)

        optimizer = get_optimizer(
            trial_config["optimizer"],
            model,
            trial_config["learning_rate"],
            trial_config["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_val_loss = float("inf")
        patience = config.get("early_stop_patience", 5)
        patience_counter = 0

        # -------- Training Loop --------

        for epoch in range(config.get("epochs", 30)):

            model.train()

            for batch in train_loader:

                optimizer.zero_grad()

                if model_name in ["gru", "simple_rnn"]:
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    preds = model(x)
                    loss_fn = get_loss_fn(trial_config["loss_function"])
                    loss = compute_rnn_loss(preds, y, loss_fn, learning_method=trial_config.get("learning_method", "backprop"))

                elif model_name == "masked_autoencoder":
                    x, mask = batch
                    x = x.to(device)
                    mask = mask.to(device)
                    masked_input = x * (1 - mask)
                    recon = model(masked_input, mask)
                    loss = nn.MSELoss()(recon, x)

                elif model_name == "vae":
                    x = batch[0].to(device)
                    loss_val = compute_vae_loss(model, x)
                    if isinstance(loss_val, dict):
                        loss = loss_val["loss"]
                    else:
                        loss = loss_val

                else:
                    x = batch[0].to(device)
                    loss = compute_dae_loss(model, x)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            # -------- Validation --------

            model.eval()
            val_loss = 0
            count = 0

            with torch.no_grad():
                for batch in val_loader:

                    if model_name in ["gru", "simple_rnn"]:
                        x, y = batch
                        x = x.to(device)
                        y = y.to(device)
                        preds = model(x)
                        loss_fn = get_loss_fn(trial_config["loss_function"])
                        loss = compute_rnn_loss(preds, y, loss_fn, learning_method=trial_config["learning_method"])
                    elif model_name == "masked_autoencoder":
                        x, mask = batch
                        x = x.to(device)
                        mask = mask.to(device)
                        masked_input = x * (1 - mask)
                        recon = model(masked_input, mask)
                        loss = nn.MSELoss()(recon, x)

                    elif model_name == "vae":
                        x = batch[0].to(device)
                        loss_val = compute_vae_loss(model, x)

                        if isinstance(loss_val, dict):
                            loss = loss_val["loss"]
                        else:
                            loss = loss_val

                    else:
                        x = batch[0].to(device)
                        loss = compute_dae_loss(model, x)

                    val_loss += loss.item()
                    count += 1

            val_loss /= max(1, count)
            scheduler.step(val_loss)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return best_val_loss

    return objective


def consolidate_best_params(best_params, model_name):
    """Merge individual layer params into single comma-separated strings."""
    consolidated = {}

    for key, value in best_params.items():
        # Skip individual layer keys — we'll rebuild them below
        if key.startswith("enc_l") or key.startswith("dec_l") or key.startswith("hidden_l"):
            continue
        consolidated[key] = value

    # ---- Rebuild encoder/decoder/hidden layer strings ----
    if model_name in ["vae", "denoise_ae", "masked_autoencoder"]:
        num_enc = best_params.get("num_encoder_layers", 0)
        enc_units = [str(best_params[f"enc_l{i}"]) for i in range(num_enc)]
        consolidated["encoder_layers_units"] = ",".join(enc_units)

        num_dec = best_params.get("num_decoder_layers", 0)
        dec_units = [str(best_params[f"dec_l{i}"]) for i in range(num_dec)]
        consolidated["decoder_layers_units"] = ",".join(dec_units)

    elif model_name in ["gru", "simple_rnn"]:
        num_layers = best_params.get("num_layers", 0)
        hidden_units = [str(best_params[f"hidden_l{i}"]) for i in range(num_layers)]
        consolidated["hidden_dims"] = ",".join(hidden_units)

    return consolidated

if __name__ == "__main__":

    # -------- Load Config --------
    with open(r"gru\config.json") as f:
        config = json.load(f)

    model_name = config["model_name"]

    if model_name == "vae":

        train_tensor = load_tensor_generic("train_set.csv")
        val_tensor = load_tensor_generic("val_set.csv")

        train_target = None
        val_target = None

    elif model_name == "gru":
        train_tensor = load_tensor_generic("gru/X_train.pt")
        val_tensor = load_tensor_generic("gru/X_val.pt")
        train_target = load_tensor_generic("gru/y_train.pt")
        val_target = load_tensor_generic("gru/y_val.pt")
    
    elif model_name == "simple_rnn":
        train_tensor = load_tensor_generic("simple_rnn/X_train.pt")
        val_tensor = load_tensor_generic("simple_rnn/X_val.pt")
        train_target = load_tensor_generic("simple_rnn/y_train.pt")
        val_target = load_tensor_generic("simple_rnn/y_val.pt")

    else:
        raise ValueError("Unsupported model")


    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5),
    )

    objective = create_objective(
        train_tensor,
        val_tensor,
        train_target,
        val_target,
        config,
    )

    study.optimize(objective, n_trials=1, timeout=3600)

    # Consolidate layer params into single attributes
    consolidated_params = consolidate_best_params(study.best_params, model_name)

    output_path = f"{model_name}/best_hyperparameters.json"
    with open(output_path, "w") as f:
        json.dump(consolidated_params, f, indent=4)

    print("Best Value:", study.best_value)
    print("Best Params:", consolidated_params)