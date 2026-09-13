"""
Grad-CAM vizualizacija - interpretacija odluka modela.

Metoda racuna gradijent skora predvidjene klase u odnosu na mape obelezja
poslednjeg konvolucionog sloja. Prosecan gradijent po kanalu sluzi kao tezina
tog kanala, pa se ponderisana suma mapa prikazuje kao toplotna mapa preko
originalnog snimka.

Svrha u ovom radu je provera da li modeli donose odluku na osnovu plucnog
parenhima ili na osnovu artefakata snimanja (ivice, natpisi, oznake aparata).
To je kljucno pitanje kod skupova koji su nastali spajanjem vise izvora.

Podrzani su modeli sa pretreniranom osnovom. Model 'cnn_od_nule' se preskace.

Pokretanje:
    python -m scripts.gradcam
    python -m scripts.gradcam --model densenet121 --po_klasi 4
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import src.models  # registruje sloj Normalizacija
from src.data import (_iseci, indeksiraj_snimke, imena_klasa, podeli_podatke,
                      ucitaj_konfiguraciju)
from src.models import PUNA_IMENA, pronadji_osnovu
from src.paths import direktorijum_rezultata

plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 10})


def ucitaj_snimak(putanja, visina, sirina, isecanje=None):
    """Priprema snimak isto kao u treningu, ukljucujuci isecanje ako je bilo."""
    bajtovi = tf.io.read_file(putanja)
    slika = tf.io.decode_image(bajtovi, channels=3, expand_animations=False)
    if isecanje and isecanje.get("aktivno"):
        slika = _iseci(slika, isecanje)
    slika = tf.image.resize(slika, [visina, sirina], method="bilinear")
    return tf.cast(slika, tf.float32)


def toplotna_mapa(model, slika, klasa=None):
    """
    Racuna Grad-CAM mapu za jedan snimak.

    Sloj augmentacije se namerno preskace - u rezimu zakljucivanja on je
    neaktivan, pa se racuna direktno normalizacija pa osnova.
    """
    osnova = pronadji_osnovu(model)
    if osnova is None:
        return None, None

    # Slojevi glave se ne navode po imenu, nego se uzimaju svi slojevi posle
    # osnove. Time metoda radi i za modele sa dodatnim skrivenim slojem u
    # glavi, kakav nastaje optimizacijom hiperparametara.
    indeks_osnove = model.layers.index(osnova)
    glava = model.layers[indeks_osnove + 1:]

    ulaz = tf.expand_dims(slika, 0)

    with tf.GradientTape() as traka:
        x = model.get_layer("normalizacija")(ulaz)
        obelezja = osnova(x, training=False)      # poslednja konvoluciona mapa
        traka.watch(obelezja)

        h = obelezja
        for sloj in glava:
            h = sloj(h, training=False)
        predikcija = h

        if klasa is None:
            klasa = int(tf.argmax(predikcija[0]))
        skor = predikcija[:, klasa]

    gradijenti = traka.gradient(skor, obelezja)
    tezine = tf.reduce_mean(gradijenti, axis=(1, 2))          # prosek po prostoru

    mapa = tf.reduce_sum(obelezja[0] * tezine[0], axis=-1)
    mapa = tf.nn.relu(mapa)                                   # samo pozitivan doprinos
    mapa = mapa / (tf.reduce_max(mapa) + keras.backend.epsilon())

    mapa = tf.image.resize(mapa[..., None], slika.shape[:2]).numpy().squeeze()
    return mapa, (klasa, float(predikcija[0, klasa]))


def nacrtaj(ime, model, primeri, klase, visina, sirina, putanja, isecanje=None):
    fig, ose = plt.subplots(2, len(primeri), figsize=(3 * len(primeri), 6.4))

    for j, (put, stvarna) in enumerate(primeri):
        slika = ucitaj_snimak(put, visina, sirina, isecanje)
        mapa, (pred, poverenje) = toplotna_mapa(model, slika)
        sivo = slika.numpy().astype("uint8")

        ose[0, j].imshow(sivo)
        ose[0, j].set_title(f"stvarno: {stvarna}", fontsize=10)

        ose[1, j].imshow(sivo)
        ose[1, j].imshow(mapa, cmap="jet", alpha=0.42)
        tacno = klase[pred] == stvarna
        ose[1, j].set_title(f"{klase[pred]}  ({poverenje:.2f})", fontsize=10,
                            color="#27ae60" if tacno else "#c0392b")

        for osa in (ose[0, j], ose[1, j]):
            osa.axis("off")

    fig.suptitle(f"Grad-CAM - {PUNA_IMENA.get(ime, ime)}", fontsize=13)
    plt.tight_layout()
    plt.savefig(putanja)
    plt.close()
    print(f"  sacuvano: {putanja}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="samo jedan model")
    p.add_argument("--po_klasi", type=int, default=2, help="broj primera po klasi")
    args = p.parse_args()

    cfg = ucitaj_konfiguraciju()
    v, s = cfg["slika"]["visina"], cfg["slika"]["sirina"]

    df = indeksiraj_snimke()
    klase = imena_klasa(df)
    _, _, test = podeli_podatke(df, cfg["podaci"]["udeo_validacionog"], cfg["seed"])

    primeri = []
    for k in klase:
        uzorak = test[test.klasa == k].sample(args.po_klasi, random_state=cfg["seed"])
        primeri += [(r.putanja, k) for _, r in uzorak.iterrows()]

    izlaz = direktorijum_rezultata()
    if args.model:
        modeli = [args.model]
    else:
        dodatni = sorted(p.name for p in izlaz.iterdir()
                         if p.is_dir() and p.name not in cfg["modeli"]
                         and (p / "model.keras").exists())
        modeli = list(cfg["modeli"]) + dodatni

    for ime in modeli:
        putanja_modela = izlaz / ime / "model.keras"
        if not putanja_modela.exists():
            continue
        model = keras.models.load_model(putanja_modela)
        if pronadji_osnovu(model) is None:
            print(f"Preskacem {ime}: nema ugnezdenu osnovu.")
            continue
        print(f"Grad-CAM: {PUNA_IMENA.get(ime, ime)}")
        # Model obucen na isecenim snimcima mora se i ovde gledati na isecenim
        isec = dict(cfg["isecanje"], aktivno=ime.endswith("_isecen"))
        nacrtaj(ime, model, primeri, klase, v, s,
                izlaz / ime / "gradcam.png", isec)


if __name__ == "__main__":
    main()