"""
Evaluacija obucenih modela na test skupu i njihovo medjusobno poredjenje.

Generise sve tabele i grafikone za poglavlja 'Rezultati' i 'Analiza':
  - uporednu tabelu metrika za sve modele
  - metrike po klasama za svaki model
  - matrice konfuzije (apsolutne i normalizovane)
  - ROC krive sa AUC vrednostima (jedan-protiv-ostalih)
  - krive ucenja sa oznacenom granicom izmedju faza

Pokretanje:
    python -m scripts.evaluacija
    python -m scripts.evaluacija --model resnet50
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_curve)

import src.models  # neophodno: registruje sloj Normalizacija pri ucitavanju
from src.data import pripremi_sve
from src.models import PUNA_IMENA
from src.paths import direktorijum_rezultata

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

BOJE_MODELA = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad"]


# ---------------------------------------------------------------- predikcija

def predvidi(model, test_ds):
    """Vraca stvarne oznake, predvidjene oznake i verovatnoce."""
    y_stvarno = np.concatenate([o.numpy() for _, o in test_ds])
    verovatnoce = model.predict(test_ds, verbose=0)
    return y_stvarno, np.argmax(verovatnoce, axis=1), verovatnoce


# ---------------------------------------------------------------- grafikoni

def nacrtaj_matricu(y_stvarno, y_pred, klase, naslov, putanja):
    mk = confusion_matrix(y_stvarno, y_pred)
    mk_norm = mk.astype(float) / mk.sum(axis=1, keepdims=True)

    fig, ose = plt.subplots(1, 2, figsize=(11, 4.6))
    for osa, podaci, oznaka, fmt in [
        (ose[0], mk, "Apsolutne vrednosti", "d"),
        (ose[1], mk_norm, "Normalizovano po redovima", ".2f"),
    ]:
        slika = osa.imshow(podaci, cmap="Blues", vmin=0,
                           vmax=podaci.max() if fmt == "d" else 1.0)
        osa.set_xticks(range(len(klase)), klase, rotation=20)
        osa.set_yticks(range(len(klase)), klase)
        osa.set_xlabel("Predvidjena klasa"); osa.set_ylabel("Stvarna klasa")
        osa.set_title(oznaka, fontsize=11)
        osa.grid(False)
        prag = podaci.max() * 0.6
        for i in range(len(klase)):
            for j in range(len(klase)):
                osa.text(j, i, format(podaci[i, j], fmt), ha="center", va="center",
                         color="white" if podaci[i, j] > prag else "black", fontsize=10)
        fig.colorbar(slika, ax=osa, fraction=0.046)

    fig.suptitle(naslov, fontsize=13)
    plt.tight_layout(); plt.savefig(putanja); plt.close()


def nacrtaj_roc(y_stvarno, verovatnoce, klase, naslov, putanja):
    """ROC krive po pristupu jedan-protiv-ostalih."""
    fig, osa = plt.subplots(figsize=(6, 5))
    povrsine = {}

    for i, ime in enumerate(klase):
        fpr, tpr, _ = roc_curve((y_stvarno == i).astype(int), verovatnoce[:, i])
        povrsine[ime] = auc(fpr, tpr)
        osa.plot(fpr, tpr, lw=2, label=f"{ime} (AUC = {povrsine[ime]:.3f})")

    osa.plot([0, 1], [0, 1], "k--", lw=1, label="nasumicno pogadjanje")
    osa.set_xlabel("Stopa laznih pozitiva"); osa.set_ylabel("Stopa stvarnih pozitiva")
    osa.set_title(naslov); osa.legend(loc="lower right", fontsize=9)
    plt.savefig(putanja); plt.close()
    return povrsine


def nacrtaj_krive_ucenja(sazetak, putanja):
    """Krive ucenja obe faze, sa vertikalnom linijom na granici."""
    faza1 = sazetak["istorija"].get("faza1", {})
    faza2 = sazetak["istorija"].get("faza2", {})
    granica = len(faza1.get("loss", []))

    def spoji(kljuc):
        return list(faza1.get(kljuc, [])) + list(faza2.get(kljuc, []))

    fig, ose = plt.subplots(1, 3, figsize=(14, 4))
    parovi = [("loss", "val_loss", "Funkcija gubitka"),
              ("accuracy", "val_accuracy", "Tacnost"),
              (None, "val_makro_f1", "Macro F1 (validacioni)")]

    for osa, (tren, val, naslov) in zip(ose, parovi):
        if tren:
            osa.plot(spoji(tren), label="trening", lw=2)
        osa.plot(spoji(val), label="validacioni", lw=2)
        if faza2:
            osa.axvline(granica - 0.5, color="gray", ls="--", lw=1.2)
            osa.text(granica - 0.4, osa.get_ylim()[1] * 0.97, " faza 2",
                     fontsize=9, color="gray", va="top")
        osa.set_xlabel("Epoha"); osa.set_title(naslov); osa.legend(fontsize=9)

    fig.suptitle(f"Krive ucenja - {sazetak['puno_ime']}", fontsize=13)
    plt.tight_layout(); plt.savefig(putanja); plt.close()


# ---------------------------------------------------------------- evaluacija

def evaluiraj(ime, test_ds, klase, izlaz_korena):
    direktorijum = izlaz_korena / ime
    putanja_modela = direktorijum / "model.keras"
    if not putanja_modela.exists():
        print(f"  preskacem {ime}: model nije pronadjen")
        return None

    puno_ime = PUNA_IMENA.get(ime, ime)
    print(f"\nEvaluacija: {puno_ime}")
    model = keras.models.load_model(putanja_modela)
    y_stvarno, y_pred, verovatnoce = predvidi(model, test_ds)

    tacnost = accuracy_score(y_stvarno, y_pred)
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        y_stvarno, y_pred, average="macro", zero_division=0)
    p_pt, r_pt, f_pt, _ = precision_recall_fscore_support(
        y_stvarno, y_pred, average="weighted", zero_division=0)

    nacrtaj_matricu(y_stvarno, y_pred, klase,
                    f"Matrica konfuzije - {puno_ime}",
                    direktorijum / "matrica_konfuzije.png")
    povrsine = nacrtaj_roc(y_stvarno, verovatnoce, klase,
                           f"ROC krive - {puno_ime}",
                           direktorijum / "roc_krive.png")

    sazetak_putanja = direktorijum / "sazetak.json"
    sazetak = json.loads(sazetak_putanja.read_text(encoding="utf-8")) \
        if sazetak_putanja.exists() else {}
    if sazetak.get("istorija"):
        nacrtaj_krive_ucenja(sazetak, direktorijum / "krive_ucenja.png")

    # metrike po klasama
    izvestaj = classification_report(y_stvarno, y_pred, target_names=klase,
                                     output_dict=True, zero_division=0)
    po_klasama = pd.DataFrame(izvestaj).T.loc[klase]
    po_klasama["AUC"] = [povrsine[k] for k in klase]
    po_klasama.round(4).to_csv(direktorijum / "metrike_po_klasama.csv")

    print(po_klasama.round(3).to_string())

    return {
        "Model": sazetak.get("puno_ime", puno_ime),
        "Accuracy": tacnost,
        "Precision (macro)": p_ma, "Recall (macro)": r_ma, "F1 (macro)": f_ma,
        "Precision (weighted)": p_pt, "Recall (weighted)": r_pt,
        "F1 (weighted)": f_pt,
        "AUC (prosek)": float(np.mean(list(povrsine.values()))),
        "Recall COVID19": float(po_klasama.loc["COVID19", "recall"])
        if "COVID19" in klase else np.nan,
        "Trajanje (min)": round(sazetak.get("trajanje_sekundi", 0) / 60, 1),
        "Parametara": sazetak.get("parametri", {}).get("ukupno"),
    }


def uporedni_grafikon(tab, putanja):
    metrike = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    x = np.arange(len(metrike))
    sirina = 0.8 / len(tab)

    fig, osa = plt.subplots(figsize=(9, 4.8))
    for i, (_, red) in enumerate(tab.iterrows()):
        vrednosti = [red[m] for m in metrike]
        stubici = osa.bar(x + i * sirina - 0.4 + sirina / 2, vrednosti, sirina,
                          label=red["Model"],
                          color=BOJE_MODELA[i % len(BOJE_MODELA)])
        osa.bar_label(stubici, fmt="%.3f", fontsize=7, rotation=90, padding=2)

    osa.axhline(0.664, color="black", ls=":", lw=1.2)
    osa.text(len(metrike) - 0.5, 0.672, "tacnost trivijalnog klasifikatora",
             fontsize=8, ha="right")
    osa.set_xticks(x, metrike)
    osa.set_ylim(0, 1.12); osa.set_ylabel("Vrednost metrike")
    osa.set_title("Uporedni prikaz performansi modela na test skupu")
    osa.legend(fontsize=9, ncols=2)
    plt.savefig(putanja); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="evaluiraj samo jedan model")
    args = p.parse_args()

    skupovi, tabele, klase, cfg = pripremi_sve()
    izlaz = direktorijum_rezultata()
    if args.model:
        modeli = [args.model]
    else:
        # ukljucuju se i modeli nastali optimizacijom, kojih nema u config.yaml
        dodatni = sorted(p.name for p in izlaz.iterdir()
                         if p.is_dir() and p.name not in cfg["modeli"]
                         and (p / "model.keras").exists())
        modeli = list(cfg["modeli"]) + dodatni

    redovi = [r for r in (evaluiraj(ime, skupovi["test"], klase, izlaz)
                          for ime in modeli) if r]
    if not redovi:
        print("Nijedan model nije evaluiran.")
        return

    tab = pd.DataFrame(redovi).sort_values("F1 (macro)", ascending=False)
    tab.round(4).to_csv(izlaz / "tabela_4_poredjenje_modela.csv", index=False)
    uporedni_grafikon(tab, izlaz / "grafikon_poredjenje_modela.png")

    print("\n" + "=" * 100)
    print("UPOREDNI PREGLED NA TEST SKUPU")
    print("=" * 100)
    prikaz = tab[["Model", "Accuracy", "Precision (macro)", "Recall (macro)",
                  "F1 (macro)", "AUC (prosek)", "Recall COVID19",
                  "Trajanje (min)"]]
    print(prikaz.round(4).to_string(index=False))
    print("=" * 100)
    print(f"\nSve sacuvano u: {izlaz}")


if __name__ == "__main__":
    main()