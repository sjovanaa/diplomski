"""
Ansambl obucenih modela.

Verovatnoce vise modela se usrednjavaju, pa se klasa bira po najvecoj
prosecnoj verovatnoci. Greske pojedinacnih modela su delimicno nezavisne,
pa se usrednjavanjem medjusobno ponistavaju.

Kljucna metodoloska odluka: sastav ansambla bira se ISKLJUCIVO na
validacionom skupu. Test skup se koristi samo jednom, za merenje konacno
izabranog ansambla. Kada bi se sastav birao po rezultatu na testu, dobijena
vrednost vise ne bi bila nepristrasna procena performansi.

Pokretanje:
    python -u -m scripts.ansambl
    python -u -m scripts.ansambl --max_clanova 3
"""
import argparse
import itertools
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

import src.models  # registruje sloj Normalizacija
from src.data import pripremi_sve
from src.models import PUNA_IMENA
from src.paths import direktorijum_rezultata

plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def verovatnoce_svih(izlaz, skupovi):
    """Za svaki sacuvan model racuna verovatnoce na validacionom i test skupu."""
    rezultat = {}
    for direktorijum in sorted(p for p in izlaz.iterdir() if p.is_dir()):
        putanja = direktorijum / "model.keras"
        if not putanja.exists():
            continue
        ime = direktorijum.name
        print(f"Ucitavam: {PUNA_IMENA.get(ime, ime)}")
        model = keras.models.load_model(putanja)
        rezultat[ime] = {
            "validacioni": model.predict(skupovi["validacioni"], verbose=0),
            "test": model.predict(skupovi["test"], verbose=0),
        }
        keras.backend.clear_session()
    return rezultat


def makro_f1(y, verovatnoce):
    p = np.argmax(verovatnoce, axis=1)
    return precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_clanova", type=int, default=4)
    args = p.parse_args()

    skupovi, tabele, klase, cfg = pripremi_sve()
    izlaz = direktorijum_rezultata()

    y_val = np.concatenate([o.numpy() for _, o in skupovi["validacioni"]])
    y_test = np.concatenate([o.numpy() for _, o in skupovi["test"]])

    predikcije = verovatnoce_svih(izlaz, skupovi)
    if len(predikcije) < 2:
        print("Potrebna su najmanje dva sacuvana modela.")
        return

    imena = list(predikcije)
    print("\nPojedinacni modeli (validacioni macro F1):")
    for ime in imena:
        print(f"  {PUNA_IMENA.get(ime, ime):<28} "
              f"{makro_f1(y_val, predikcije[ime]['validacioni']):.4f}")

    # ---- pretraga sastava, iskljucivo na validacionom skupu
    redovi = []
    for k in range(2, min(args.max_clanova, len(imena)) + 1):
        for sastav in itertools.combinations(imena, k):
            prosek = np.mean([predikcije[i]["validacioni"] for i in sastav], axis=0)
            redovi.append({"sastav": sastav, "broj": k,
                           "val_makro_f1": makro_f1(y_val, prosek)})

    tab = pd.DataFrame(redovi).sort_values("val_makro_f1", ascending=False)
    print("\nNajboljih pet sastava (po validacionom skupu):")
    for _, red in tab.head(5).iterrows():
        clanovi = ", ".join(PUNA_IMENA.get(i, i) for i in red["sastav"])
        print(f"  {red['val_makro_f1']:.4f}  |  {clanovi}")

    najbolji = tab.iloc[0]["sastav"]

    # ---- test skup se dodiruje tek sada, za konacno izabrani ansambl
    prosek_test = np.mean([predikcije[i]["test"] for i in najbolji], axis=0)
    y_pred = np.argmax(prosek_test, axis=1)

    tacnost = accuracy_score(y_test, y_pred)
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    _, odziv, _, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(klase)), zero_division=0)

    print("\n" + "=" * 70)
    print("ANSAMBL - REZULTAT NA TEST SKUPU")
    print("=" * 70)
    print("Sastav:", ", ".join(PUNA_IMENA.get(i, i) for i in najbolji))
    print(f"Accuracy          {tacnost:.4f}")
    print(f"Precision (macro) {p_ma:.4f}")
    print(f"Recall (macro)    {r_ma:.4f}")
    print(f"F1 (macro)        {f_ma:.4f}")
    if "COVID19" in klase:
        print(f"Recall COVID19    {odziv[klase.index('COVID19')]:.4f}")
    print("\nMatrica konfuzije:")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                       index=klase, columns=klase).to_string())
    print("=" * 70)

    with open(izlaz / "ansambl.json", "w", encoding="utf-8") as f:
        json.dump({
            "sastav": list(najbolji),
            "val_makro_f1": float(tab.iloc[0]["val_makro_f1"]),
            "test": {"accuracy": float(tacnost), "precision_macro": float(p_ma),
                     "recall_macro": float(r_ma), "f1_macro": float(f_ma),
                     "recall_po_klasama": dict(zip(klase, map(float, odziv)))},
        }, f, indent=2, ensure_ascii=False)

    tab["sastav"] = tab["sastav"].map(lambda s: " + ".join(s))
    tab.round(4).to_csv(izlaz / "tabela_7_ansambli.csv", index=False)
    print(f"\nSacuvano u {izlaz}")


if __name__ == "__main__":
    main()