"""
Optimizacija hiperparametara Bajesovom pretragom (KerasTuner).

Pretraga se sprovodi nad najuspesnijim modelom iz osnovnog poredjenja
(podrazumevano DenseNet121). Pretrazivanje nad svim modelima bi visestruko
povecalo utrosak racunarskih resursa, a ne bi promenilo zakljucke rada.

Prostor pretrage obuhvata cetiri hiperparametra:
  dropout           - jacina regularizacije klasifikacione glave
  lr_glava          - stopa ucenja u fazi 1
  lr_finog          - stopa ucenja u fazi 2
  udeo_odmrznutih   - koliki deo osnove ulazi u fino podesavanje

Kriterijum optimizacije je macro F1 na validacionom skupu, isti onaj po kom
se bira najbolja epoha - ne tacnost, iz vec objasnjenih razloga.

Rezultat je model sacuvan kao 'densenet121_opt', koji skripta za evaluaciju
automatski ukljucuje u uporednu tabelu.

Pokretanje:
    python -u -m scripts.optimizacija
    python -u -m scripts.optimizacija --pokusaja 8 --model resnet50
"""
import argparse
import copy
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import keras_tuner as kt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import pripremi_sve, tezine_klasa
from src.models import PUNA_IMENA, napravi_model, odmrzni_osnovu
from src.paths import direktorijum_rezultata

EPOHE_PRETRAGE_FAZA1 = 10   # skraceno u odnosu na finalni trening
EPOHE_PRETRAGE_FAZA2 = 8

# Prostor pretrage
STOPE_GLAVE = [3e-3, 1e-3, 5e-4, 3e-4, 1e-4]
STOPE_FINOG = [1e-4, 5e-5, 3e-5, 1e-5, 3e-6]
UDEO_ODMRZNUTIH = [0.15, 0.3, 0.5, 0.7]


class MakroF1(keras.callbacks.Callback):
    """Macro F1 na validacionom skupu posle svake epohe."""

    def __init__(self, validacioni):
        super().__init__()
        self.validacioni = validacioni

    def on_epoch_end(self, epoha, dnevnik=None):
        dnevnik = dnevnik if dnevnik is not None else {}
        y = np.concatenate([o.numpy() for _, o in self.validacioni])
        p = np.argmax(self.model.predict(self.validacioni, verbose=0), axis=1)
        dnevnik["val_makro_f1"] = f1_score(y, p, average="macro")


def _optimizator(hp, ime_stope, vrednosti):
    """
    Bira optimizator i stopu ucenja.

    AdamW razdvaja opadanje tezina od gradijenta, pa je kod finog podesavanja
    velikih pretreniranih mreza cesto stabilniji od klasicnog Adam-a.
    """
    stopa = hp.Choice(ime_stope, vrednosti)
    if hp.Choice("optimizator", ["adam", "adamw"]) == "adamw":
        return keras.optimizers.AdamW(
            stopa, weight_decay=hp.Choice("opadanje_tezina", [1e-4, 1e-2]))
    return keras.optimizers.Adam(stopa)


class DvofazniModel(kt.HyperModel):
    """
    Hipermodel koji unutar jednog pokusaja izvodi obe faze obucavanja.

    Bez ovoga bi se optimizovali samo hiperparametri faze 1, pa bi izbor
    stope ucenja za fino podesavanje ostao proizvoljan.
    """

    def __init__(self, ime, klase, cfg):
        self.ime = ime
        self.klase = klase
        self.cfg = cfg

    def build(self, hp):
        cfg = copy.deepcopy(self.cfg)
        cfg["trening"]["dropout"] = hp.Choice("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])
        cfg["trening"]["glava_neurona"] = hp.Choice("glava_neurona", [0, 128, 256])

        # Hiperparametri faze 2 se prijavljuju vec ovde, iako se koriste u fit().
        # Bez toga ih prvi pokusaj ne bi imao u prostoru pretrage.
        hp.Choice("udeo_odmrznutih", UDEO_ODMRZNUTIH)
        hp.Choice("lr_finog", STOPE_FINOG)

        model = napravi_model(self.ime, len(self.klase), cfg)
        model.compile(
            optimizer=_optimizator(hp, "lr_glava", STOPE_GLAVE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
        return model

    def fit(self, hp, model, *args, **kwargs):
        validacioni = kwargs.get("validation_data")
        pozivi = list(kwargs.pop("callbacks", [])) + [MakroF1(validacioni)]

        # ---- faza 1
        h1 = model.fit(*args, **kwargs, epochs=EPOHE_PRETRAGE_FAZA1,
                       callbacks=pozivi, verbose=2)

        # ---- faza 2
        odmrzni_osnovu(model, hp.Choice("udeo_odmrznutih", UDEO_ODMRZNUTIH))
        model.compile(
            optimizer=_optimizator(hp, "lr_finog", STOPE_FINOG),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

        h2 = model.fit(*args, **kwargs, epochs=EPOHE_PRETRAGE_FAZA2,
                       callbacks=pozivi, verbose=2)

        svi = list(h1.history["val_makro_f1"]) + list(h2.history["val_makro_f1"])
        return {"val_makro_f1": float(max(svi))}


def _finalni_optimizator(vrednosti, stopa):
    """Rekonstruise izabrani optimizator za finalni trening."""
    if vrednosti.get("optimizator") == "adamw":
        return keras.optimizers.AdamW(
            stopa, weight_decay=vrednosti.get("opadanje_tezina", 1e-4))
    return keras.optimizers.Adam(stopa)


def tabela_pokusaja(tjuner, putanja):
    """Pravi tabelu svih isprobanih konfiguracija - ide direktno u rad."""
    redovi = []
    for i, pokusaj in enumerate(tjuner.oracle.get_best_trials(
            num_trials=len(tjuner.oracle.trials)), start=1):
        v = pokusaj.hyperparameters.values
        redovi.append({
            "Rang": i,
            "dropout": v.get("dropout"),
            "neurona u glavi": v.get("glava_neurona"),
            "optimizator": v.get("optimizator"),
            "lr (faza 1)": v.get("lr_glava"),
            "lr (faza 2)": v.get("lr_finog"),
            "udeo odmrznutih": v.get("udeo_odmrznutih"),
            "val macro F1": round(pokusaj.score, 4) if pokusaj.score else None,
        })
    tab = pd.DataFrame(redovi)
    tab.to_csv(putanja, index=False)
    return tab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="densenet121")
    p.add_argument("--pokusaja", type=int, default=20)
    p.add_argument("--epohe_finog", type=int, default=20,
                   help="epohe faze 2 u finalnom treningu")
    p.add_argument("--strpljenje", type=int, default=8)
    args = p.parse_args()

    skupovi, tabele, klase, cfg = pripremi_sve()
    tezine = tezine_klasa(tabele["trening"], klase) \
        if cfg["trening"]["koristi_tezine_klasa"] else None
    izlaz = direktorijum_rezultata()

    print("=" * 70)
    print(f"OPTIMIZACIJA HIPERPARAMETARA: {PUNA_IMENA[args.model]}")
    print(f"Broj pokusaja: {args.pokusaja}")
    print("=" * 70)

    tjuner = kt.BayesianOptimization(
        hypermodel=DvofazniModel(args.model, klase, cfg),
        objective=kt.Objective("val_makro_f1", direction="max"),
        max_trials=args.pokusaja,
        seed=cfg["seed"],
        directory=str(izlaz / "pretraga"),
        project_name=args.model,
        overwrite=True,
    )

    pocetak = time.time()
    tjuner.search(skupovi["trening"],
                  validation_data=skupovi["validacioni"],
                  class_weight=tezine)
    trajanje_pretrage = time.time() - pocetak

    tab = tabela_pokusaja(tjuner, izlaz / "tabela_5_pretraga_hiperparametara.csv")
    print("\n" + "=" * 70)
    print("REZULTATI PRETRAGE")
    print("=" * 70)
    print(tab.to_string(index=False))

    najbolji = tjuner.get_best_hyperparameters(1)[0]
    print("\nNajbolja konfiguracija:")
    for k, v in najbolji.values.items():
        print(f"  {k:<18} {v}")

    # ------------------------------------------------ finalni trening
    print("\n" + "=" * 70)
    print("FINALNI TRENING SA NAJBOLJOM KONFIGURACIJOM")
    print("=" * 70)

    cfg_opt = copy.deepcopy(cfg)
    cfg_opt["trening"].update({
        "dropout": najbolji.values["dropout"],
        "glava_neurona": najbolji.values.get("glava_neurona", 0),
        "lr_glava": najbolji.values["lr_glava"],
        "lr_finog": najbolji.values["lr_finog"],
        "udeo_odmrznutih": najbolji.values["udeo_odmrznutih"],
        # Finalni trening ide duze: ranije je uoceno da modeli u fazi 2
        # jos napreduju kada se dostigne zadati broj epoha.
        "epohe_finog": args.epohe_finog,
        "strpljenje": args.strpljenje,
    })

    ime_opt = f"{args.model}_opt"
    direktorijum = izlaz / ime_opt
    direktorijum.mkdir(parents=True, exist_ok=True)

    model = napravi_model(args.model, len(klase), cfg_opt)
    t = cfg_opt["trening"]

    def pozivi(faza):
        return [
            MakroF1(skupovi["validacioni"]),
            keras.callbacks.EarlyStopping(
                monitor="val_makro_f1", mode="max", patience=t["strpljenje"],
                restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_makro_f1", mode="max", factor=0.5,
                patience=max(2, t["strpljenje"] // 2), min_lr=1e-7, verbose=1),
            keras.callbacks.CSVLogger(direktorijum / f"dnevnik_faza{faza}.csv"),
        ]

    pocetak = time.time()
    model.compile(optimizer=_finalni_optimizator(najbolji.values, t["lr_glava"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(skupovi["trening"], validation_data=skupovi["validacioni"],
                   epochs=t["epohe_glava"], class_weight=tezine,
                   callbacks=pozivi(1), verbose=1)
    najbolji_f1 = max(h1.history["val_makro_f1"])
    najbolje_tezine = model.get_weights()

    odmrzni_osnovu(model, t["udeo_odmrznutih"])
    model.compile(optimizer=_finalni_optimizator(najbolji.values, t["lr_finog"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(skupovi["trening"], validation_data=skupovi["validacioni"],
                   epochs=t["epohe_finog"], class_weight=tezine,
                   callbacks=pozivi(2), verbose=1)
    if max(h2.history["val_makro_f1"]) <= najbolji_f1:
        print("\nFaza 2 nije nadmasila fazu 1; vracaju se tezine faze 1.")
        model.set_weights(najbolje_tezine)
    else:
        najbolji_f1 = max(h2.history["val_makro_f1"])
    trajanje = time.time() - pocetak

    model.save(direktorijum / "model.keras")
    with open(direktorijum / "sazetak.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": ime_opt,
            "puno_ime": f"{PUNA_IMENA[args.model]} (optimizovan)",
            "osnovni_model": args.model,
            "najbolji_hiperparametri": najbolji.values,
            "konfiguracija_treninga": cfg_opt["trening"],
            "broj_pokusaja": args.pokusaja,
            "najbolji_val_makro_f1": round(float(najbolji_f1), 4),
            "trajanje_pretrage_sekundi": round(trajanje_pretrage, 1),
            "trajanje_sekundi": round(trajanje, 1),
            "trajanje_citljivo": f"{int(trajanje // 60)} min {int(trajanje % 60)} s",
            "parametri": {"ukupno": model.count_params()},
            "istorija": {
                "faza1": {k: [float(x) for x in v] for k, v in h1.history.items()},
                "faza2": {k: [float(x) for x in v] for k, v in h2.history.items()},
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"\nPretraga: {trajanje_pretrage / 60:.1f} min | "
          f"Finalni trening: {trajanje / 60:.1f} min")
    print(f"Sacuvano u {direktorijum}")
    print("\nPokreni potom: python -u -m scripts.evaluacija")


if __name__ == "__main__":
    main()