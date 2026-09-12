"""
Visestruko pokretanje i analiza stabilnosti rezultata.

Svaki model se obucava vise puta, sa razlicitim semenom slucajnih brojeva.
Razlog je sto rezultat jednog pokretanja nije potpuno ponovljiv: operacije
na grafickom procesoru nisu deterministicne, pa se izmedju dva pokretanja
sa istim semenom dobijaju vidljivo razlicite vrednosti metrika.

Umesto jedne vrednosti, u radu se tada prijavljuje srednja vrednost i
standardna devijacija. Time se razlike izmedju modela mogu tumaciti kao
stvarne, a ne kao posledica jednog povoljnog pokretanja.

Pokretanje:
    python -u -m scripts.visestruko
    python -u -m scripts.visestruko --semena 42 7 123 --modeli densenet121
"""
import argparse
import copy
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.data import pripremi_sve, tezine_klasa, ucitaj_konfiguraciju
from src.models import PUNA_IMENA, napravi_model, odmrzni_osnovu
from src.paths import direktorijum_rezultata

plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

BOJE = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#e67e22"]


class MakroF1(keras.callbacks.Callback):
    def __init__(self, validacioni):
        super().__init__()
        self.validacioni = validacioni

    def on_epoch_end(self, epoha, dnevnik=None):
        dnevnik = dnevnik if dnevnik is not None else {}
        y = np.concatenate([o.numpy() for _, o in self.validacioni])
        p = np.argmax(self.model.predict(self.validacioni, verbose=0), axis=1)
        dnevnik["val_makro_f1"] = f1_score(y, p, average="macro")


def _optimizator(cfg, stopa):
    if cfg["trening"].get("optimizator") == "adamw":
        return keras.optimizers.AdamW(
            stopa, weight_decay=cfg["trening"].get("opadanje_tezina", 1e-4))
    return keras.optimizers.Adam(stopa)


def _pozivi(validacioni, strpljenje):
    return [
        MakroF1(validacioni),
        keras.callbacks.EarlyStopping(
            monitor="val_makro_f1", mode="max", patience=strpljenje,
            restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_makro_f1", mode="max", factor=0.5,
            patience=max(2, strpljenje // 2), min_lr=1e-7, verbose=0),
    ]


def jedno_pokretanje(ime, arhitektura, skupovi, tabele, klase, cfg):
    """Obucava jedan model i vraca metrike na test skupu."""
    t = cfg["trening"]
    tezine = tezine_klasa(tabele["trening"], klase) if t["koristi_tezine_klasa"] else None

    model = napravi_model(arhitektura, len(klase), cfg)
    pocetak = time.time()

    epohe1 = (t["epohe_glava"] + t["epohe_finog"]
              if arhitektura == "cnn_od_nule" else t["epohe_glava"])
    model.compile(optimizer=_optimizator(cfg, t["lr_glava"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(skupovi["trening"], validation_data=skupovi["validacioni"],
                   epochs=epohe1, class_weight=tezine,
                   callbacks=_pozivi(skupovi["validacioni"], t["strpljenje"]),
                   verbose=2)
    najbolji_f1 = max(h1.history["val_makro_f1"])
    najbolje_tezine = model.get_weights()

    if arhitektura != "cnn_od_nule":
        odmrzni_osnovu(model, t.get("udeo_odmrznutih", 0.3))
        model.compile(optimizer=_optimizator(cfg, t["lr_finog"]),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h2 = model.fit(skupovi["trening"], validation_data=skupovi["validacioni"],
                       epochs=t["epohe_finog"], class_weight=tezine,
                       callbacks=_pozivi(skupovi["validacioni"], t["strpljenje"]),
                       verbose=2)
        # Isto kao u scripts/trening.py: ako fino podesavanje nije pomoglo,
        # vracaju se tezine faze 1, da se ne prijavi losiji model od najboljeg.
        if max(h2.history["val_makro_f1"]) <= najbolji_f1:
            model.set_weights(najbolje_tezine)
        else:
            najbolji_f1 = max(h2.history["val_makro_f1"])

    trajanje = time.time() - pocetak

    y = np.concatenate([o.numpy() for _, o in skupovi["test"]])
    p = np.argmax(model.predict(skupovi["test"], verbose=0), axis=1)

    _, odziv_po_klasi, _, _ = precision_recall_fscore_support(
        y, p, average=None, labels=range(len(klase)), zero_division=0)
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        y, p, average="macro", zero_division=0)

    keras.backend.clear_session()

    return {
        "Model": ime,
        "seme": cfg["seed"],
        "Accuracy": accuracy_score(y, p),
        "Precision (macro)": p_ma,
        "Recall (macro)": r_ma,
        "F1 (macro)": f_ma,
        "Recall COVID19": float(odziv_po_klasi[klase.index("COVID19")])
        if "COVID19" in klase else np.nan,
        "val macro F1": round(float(najbolji_f1), 4),
        "Trajanje (min)": round(trajanje / 60, 1),
    }


def ucitaj_optimizovanu_konfiguraciju(cfg, arhitektura="densenet121"):
    """Preuzima najbolje hiperparametre iz rezultata pretrage, ako postoje."""
    putanja = direktorijum_rezultata() / f"{arhitektura}_opt" / "sazetak.json"
    if not putanja.exists():
        return None
    sazetak = json.loads(putanja.read_text(encoding="utf-8"))
    naj = sazetak.get("najbolji_hiperparametri")
    if not naj:
        return None

    novi = copy.deepcopy(cfg)
    # Ako je sacuvana cela konfiguracija finalnog treninga, koristi se ona -
    # inace bi se optimizovani model ovde obucavao krace (drugi broj epoha i
    # strpljenja) nego prilikom optimizacije, pa poredjenje ne bi bilo posteno.
    if sazetak.get("konfiguracija_treninga"):
        novi["trening"].update(sazetak["konfiguracija_treninga"])
        return novi
    novi["trening"].update({
        "dropout": naj.get("dropout", cfg["trening"]["dropout"]),
        "glava_neurona": naj.get("glava_neurona", 0),
        "lr_glava": naj.get("lr_glava", cfg["trening"]["lr_glava"]),
        "lr_finog": naj.get("lr_finog", cfg["trening"]["lr_finog"]),
        "udeo_odmrznutih": naj.get("udeo_odmrznutih", 0.3),
        "optimizator": naj.get("optimizator", "adam"),
        "opadanje_tezina": naj.get("opadanje_tezina", 1e-4),
    })
    return novi


def grafikon_stabilnosti(sazetak, putanja):
    metrike = ["Accuracy", "F1 (macro)", "Recall COVID19"]
    x = np.arange(len(metrike))
    sirina = 0.8 / len(sazetak)

    fig, osa = plt.subplots(figsize=(10, 5))
    for i, (model, red) in enumerate(sazetak.iterrows()):
        sredine = [red[(m, "mean")] for m in metrike]
        greske = [red[(m, "std")] for m in metrike]
        osa.bar(x + i * sirina - 0.4 + sirina / 2, sredine, sirina,
                yerr=greske, capsize=4, label=model,
                color=BOJE[i % len(BOJE)], error_kw={"lw": 1.2})

    osa.set_xticks(x, metrike)
    osa.set_ylim(0.8, 1.02)
    osa.set_ylabel("Vrednost metrike")
    osa.set_title("Stabilnost rezultata kroz vise pokretanja\n"
                  "(srednja vrednost, uspravne linije: standardna devijacija)")
    osa.legend(fontsize=9, ncols=2)
    plt.savefig(putanja)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--semena", type=int, nargs="+", default=[42, 7, 123])
    p.add_argument("--modeli", nargs="+", default=None)
    p.add_argument("--bez_optimizovanog", action="store_true")
    args = p.parse_args()

    izlaz = direktorijum_rezultata() / "visestruko"
    izlaz.mkdir(parents=True, exist_ok=True)

    osnovni_cfg = ucitaj_konfiguraciju()
    arhitekture = args.modeli or osnovni_cfg["modeli"]

    # Lista poslova: (prikazno ime, arhitektura, konfiguracija)
    poslovi = [(PUNA_IMENA[a], a, osnovni_cfg) for a in arhitekture]

    cfg_opt = None if args.bez_optimizovanog else ucitaj_optimizovanu_konfiguraciju(osnovni_cfg)
    if cfg_opt:
        poslovi.append(("DenseNet121 (optimizovan)", "densenet121", cfg_opt))
        print("Ukljucena je i optimizovana konfiguracija DenseNet121.")

    print("=" * 78)
    print(f"VISESTRUKO POKRETANJE: {len(poslovi)} modela x {len(args.semena)} semena")
    print(f"Semena: {args.semena}")
    print("=" * 78)

    redovi = []
    csv_putanja = izlaz / "pojedinacna_pokretanja.csv"

    for seme in args.semena:
        # Podela na trening i validacioni skup zavisi od semena, pa se
        # skupovi prave iznova za svako pokretanje.
        cfg_seme = copy.deepcopy(osnovni_cfg)
        cfg_seme["seed"] = seme
        skupovi, tabele, klase, _ = pripremi_sve(cfg=cfg_seme)

        for ime, arhitektura, cfg_modela in poslovi:
            cfg = copy.deepcopy(cfg_modela)
            cfg["seed"] = seme
            print(f"\n--- seme {seme} | {ime} ---")
            red = jedno_pokretanje(ime, arhitektura, skupovi, tabele, klase, cfg)
            redovi.append(red)
            print(f"    Accuracy {red['Accuracy']:.4f} | "
                  f"F1 (macro) {red['F1 (macro)']:.4f} | "
                  f"Recall COVID19 {red['Recall COVID19']:.4f}")
            # Upis posle svakog pokretanja, da se nista ne izgubi pri prekidu
            pd.DataFrame(redovi).to_csv(csv_putanja, index=False)

    tab = pd.DataFrame(redovi)
    metrike = ["Accuracy", "Precision (macro)", "Recall (macro)",
               "F1 (macro)", "Recall COVID19", "Trajanje (min)"]
    sazetak = tab.groupby("Model")[metrike].agg(["mean", "std"])
    sazetak = sazetak.sort_values(("F1 (macro)", "mean"), ascending=False)
    sazetak.round(4).to_csv(direktorijum_rezultata() / "tabela_6_stabilnost.csv")

    grafikon_stabilnosti(sazetak, direktorijum_rezultata() / "grafikon_stabilnost.png")

    print("\n" + "=" * 78)
    print(f"REZULTATI KROZ {len(args.semena)} POKRETANJA (srednja vrednost +/- st. devijacija)")
    print("=" * 78)
    for model, red in sazetak.iterrows():
        print(f"\n{model}")
        for m in ["Accuracy", "F1 (macro)", "Recall COVID19"]:
            print(f"  {m:<18} {red[(m, 'mean')]:.4f} +/- {red[(m, 'std')]:.4f}")
    print("\n" + "=" * 78)
    print("Pojedinacna pokretanja:")
    print(tab.round(4).to_string(index=False))
    print("=" * 78)
    print(f"\nSacuvano u {direktorijum_rezultata()}")


if __name__ == "__main__":
    main()