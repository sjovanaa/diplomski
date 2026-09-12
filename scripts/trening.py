"""
Obucavanje modela u dve faze.

  Faza 1 - osnova je zamrznuta, obucava se samo klasifikaciona glava
  Faza 2 - odmrzava se gornji deo osnove i fino podesava malom stopom ucenja

Model 'cnn_od_nule' nema pretrenirane tezine, pa se obucava u jednoj fazi
sa ukupnim brojem epoha obe faze.

Pokretanje:
    python -m scripts.trening --model resnet50
    python -m scripts.trening --svi
    python -m scripts.trening --svi --brzo        # provera na malom uzorku
"""
import argparse
import json
import os
import time
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from src.data import pripremi_sve, tezine_klasa
from src.models import PUNA_IMENA, broj_parametara, napravi_model, odmrzni_osnovu
from src.paths import direktorijum_rezultata


class MakroF1(keras.callbacks.Callback):
    """
    Racuna macro F1 na validacionom skupu posle svake epohe.

    Keras ne nudi ovu metriku za celobrojne oznake, a upravo ona je merodavna
    kod neuravnotezenog skupa: tacnost bi bila visoka i kada model potpuno
    zanemari najmalobrojniju klasu.
    """

    def __init__(self, validacioni):
        super().__init__()
        self.validacioni = validacioni
        self.y_stvarno = None      # racuna se jednom, ne u svakoj epohi

    def on_epoch_end(self, epoha, dnevnik=None):
        # dnevnik je prazan recnik kada ga Keras prosledi, pa provera mora
        # biti na None - "or {}" bi napravio novi recnik i metrika bi se
        # izgubila pre nego sto je ostali povratni pozivi procitaju.
        dnevnik = dnevnik if dnevnik is not None else {}
        if self.y_stvarno is None:
            self.y_stvarno = np.concatenate([o.numpy() for _, o in self.validacioni])
        y_pred = np.argmax(self.model.predict(self.validacioni, verbose=0), axis=1)
        dnevnik["val_makro_f1"] = f1_score(self.y_stvarno, y_pred, average="macro")
        print(f"   val_makro_f1: {dnevnik['val_makro_f1']:.4f}")


def _povratni_pozivi(validacioni, izlaz, strpljenje, faza):
    # ModelCheckpoint se namerno ne koristi: cuvao bi najbolji model unutar
    # jedne faze, pa bi faza 2 prepisala bolji rezultat faze 1. Model se cuva
    # eksplicitno, tek posle poredjenja obe faze.
    return [
        MakroF1(validacioni),
        keras.callbacks.EarlyStopping(
            monitor="val_makro_f1", mode="max", patience=strpljenje,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_makro_f1", mode="max", factor=0.5,
            patience=max(2, strpljenje // 2), min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(izlaz / f"dnevnik_faza{faza}.csv"),
    ]


def obuci(ime, skupovi, tabele, klase, cfg):
    izlaz = direktorijum_rezultata() / ime
    izlaz.mkdir(parents=True, exist_ok=True)

    t = cfg["trening"]
    tezine = tezine_klasa(tabele["trening"], klase) if t["koristi_tezine_klasa"] else None

    print("\n" + "=" * 70)
    print(f"MODEL: {PUNA_IMENA[ime]}")
    print("=" * 70)

    model = napravi_model(ime, len(klase), cfg)
    param = broj_parametara(model)
    print(f"Parametara ukupno: {param['ukupno']:,} | obucivih: {param['obucivi']:,}")

    pocetak = time.time()
    istorija = {}

    # ------------------------------------------------ faza 1
    if ime == "cnn_od_nule":
        epohe = t["epohe_glava"] + t["epohe_finog"]
        stopa = t["lr_glava"]
        print(f"\nJednofazno obucavanje ({epohe} epoha, lr={stopa})")
    else:
        epohe = t["epohe_glava"]
        stopa = t["lr_glava"]
        print(f"\nFAZA 1 - zamrznuta osnova ({epohe} epoha, lr={stopa})")

    model.compile(optimizer=keras.optimizers.Adam(stopa),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    h1 = model.fit(skupovi["trening"], validation_data=skupovi["validacioni"],
                   epochs=epohe, class_weight=tezine,
                   callbacks=_povratni_pozivi(skupovi["validacioni"], izlaz,
                                              t["strpljenje"], 1),
                   verbose=1)
    istorija["faza1"] = {k: [float(v) for v in vr] for k, vr in h1.history.items()}
    najbolji_f1 = max(h1.history["val_makro_f1"])
    najbolje_tezine = model.get_weights()      # snimak najboljih tezina faze 1
    najbolja_faza = 1

    # ------------------------------------------------ faza 2
    if ime != "cnn_od_nule":
        odmrznuto = odmrzni_osnovu(model, t.get("udeo_odmrznutih", 0.3))
        print(f"\nFAZA 2 - fino podesavanje: odmrznuto {odmrznuto} slojeva, "
              f"lr={t['lr_finog']}")

        # Obavezno ponovno kompajliranje: bez njega Keras ne registruje
        # promenu statusa 'trainable' na slojevima osnove.
        model.compile(optimizer=keras.optimizers.Adam(t["lr_finog"]),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        h2 = model.fit(skupovi["trening"], validation_data=skupovi["validacioni"],
                       epochs=t["epohe_finog"], class_weight=tezine,
                       callbacks=_povratni_pozivi(skupovi["validacioni"], izlaz,
                                                  t["strpljenje"], 2),
                       verbose=1)
        istorija["faza2"] = {k: [float(v) for v in vr] for k, vr in h2.history.items()}

        f1_faza2 = max(h2.history["val_makro_f1"])
        if f1_faza2 > najbolji_f1:
            najbolji_f1, najbolja_faza = f1_faza2, 2
        else:
            # Fino podesavanje nije donelo poboljsanje - vracaju se tezine faze 1.
            print(f"\nFaza 2 ({f1_faza2:.4f}) nije nadmasila fazu 1 "
                  f"({najbolji_f1:.4f}); vracaju se tezine faze 1.")
            model.set_weights(najbolje_tezine)

    print(f"\nNajbolji val_makro_f1: {najbolji_f1:.4f} (faza {najbolja_faza})")
    trajanje = time.time() - pocetak

    # ------------------------------------------------ cuvanje
    model.save(izlaz / "model.keras")
    sazetak = {
        "model": ime,
        "puno_ime": PUNA_IMENA[ime],
        "najbolji_val_makro_f1": round(float(najbolji_f1), 4),
        "najbolja_faza": najbolja_faza,
        "parametri": param,
        "trajanje_sekundi": round(trajanje, 1),
        "trajanje_citljivo": f"{int(trajanje // 60)} min {int(trajanje % 60)} s",
        "vreme": datetime.now().isoformat(timespec="seconds"),
        "konfiguracija": cfg["trening"],
        "istorija": istorija,
    }
    with open(izlaz / "sazetak.json", "w", encoding="utf-8") as f:
        json.dump(sazetak, f, indent=2, ensure_ascii=False)

    print(f"\nZavrseno za {sazetak['trajanje_citljivo']}. Sacuvano u {izlaz}")
    return sazetak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="ime jednog modela")
    p.add_argument("--svi", action="store_true", help="svi modeli iz config.yaml")
    p.add_argument("--brzo", action="store_true",
                   help="mali uzorak i 2 epohe - samo provera ispravnosti")
    args = p.parse_args()

    ogranici = 60 if args.brzo else None
    skupovi, tabele, klase, cfg = pripremi_sve(ogranici=ogranici)

    if args.brzo:
        cfg["trening"]["epohe_glava"] = 2
        cfg["trening"]["epohe_finog"] = 1
        cfg["trening"]["strpljenje"] = 99

    print("GPU:", tf.config.list_physical_devices("GPU") or "nije dostupan (CPU)")
    print(f"Trening: {len(tabele['trening'])} | Validacioni: {len(tabele['validacioni'])}"
          f" | Test: {len(tabele['test'])}")

    modeli = cfg["modeli"] if args.svi else [args.model]
    if not modeli or modeli == [None]:
        p.error("navedi --model IME ili --svi")

    sazeci = [obuci(ime, skupovi, tabele, klase, cfg) for ime in modeli]

    print("\n" + "=" * 70)
    print("SVI MODELI ZAVRSENI")
    print("=" * 70)
    for s in sazeci:
        print(f"  {s['puno_ime']:<20} {s['trajanje_citljivo']}")


if __name__ == "__main__":
    main()