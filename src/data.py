"""
Ucitavanje, podela i pretprocesiranje skupa rendgenskih snimaka.

Kljucna projektna odluka: ovaj modul vraca SIROVE snimke skalirane na
zadatu dimenziju, sa vrednostima piksela u opsegu [0, 255]. Normalizacija
specificna za arhitekturu (keras.applications.*.preprocess_input) primenjuje
se kao prvi sloj UNUTAR modela. Zahvaljujuci tome jedan isti skup podataka
koristi sva cetiri modela, cime je poredjenje metodoloski ispravno.
"""
import random
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from keras import layers
from sklearn.model_selection import train_test_split

from .paths import KOREN_PROJEKTA, pronadji_skup_podataka

EKSTENZIJE = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".tif", ".tiff"}
AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------- konfiguracija

def ucitaj_konfiguraciju(putanja: Path = None) -> dict:
    putanja = putanja or KOREN_PROJEKTA / "config.yaml"
    with open(putanja, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def postavi_seme(seed: int) -> None:
    """Fiksira sve izvore slucajnosti radi ponovljivosti eksperimenta."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


# ---------------------------------------------------------------- indeksiranje

def indeksiraj_snimke(koren: Path = None) -> pd.DataFrame:
    """Pravi tabelu svih snimaka sa kolonama: podskup, klasa, putanja."""
    koren = Path(koren) if koren else pronadji_skup_podataka()

    zapisi = []
    for p in koren.rglob("*"):
        if p.is_file() and p.suffix.lower() in EKSTENZIJE:
            delovi = p.relative_to(koren).parts
            if len(delovi) < 2:
                continue
            zapisi.append({
                "podskup": delovi[0].lower(),
                "klasa": delovi[-2].upper(),
                "putanja": str(p),
            })

    df = pd.DataFrame(zapisi)
    if df.empty:
        raise RuntimeError(f"U direktorijumu {koren} nije pronadjen nijedan snimak.")
    return df.sort_values(["podskup", "klasa", "putanja"]).reset_index(drop=True)


def imena_klasa(df: pd.DataFrame) -> list:
    """Abecedno sortirana imena klasa - redosled mora biti deterministican,
    jer od njega zavisi tumacenje matrice konfuzije."""
    return sorted(df["klasa"].unique())


# ---------------------------------------------------------------- podela

def podeli_podatke(df: pd.DataFrame, udeo_validacionog: float, seed: int):
    """
    Deli podatke na trening, validacioni i test skup.

    Test skup je unapred definisan strukturom foldera i ne dira se.
    Validacioni skup se STRATIFIKOVANO izdvaja iz trening skupa, cime se
    ocuvava odnos klasa uprkos izrazenoj neuravnotezenosti skupa.
    """
    trening_ceo = df[df["podskup"] == "train"].reset_index(drop=True)
    test = df[df["podskup"] == "test"].reset_index(drop=True)

    trening, validacioni = train_test_split(
        trening_ceo,
        test_size=udeo_validacionog,
        stratify=trening_ceo["klasa"],
        random_state=seed,
        shuffle=True,
    )
    return (trening.reset_index(drop=True),
            validacioni.reset_index(drop=True),
            test)


def tezine_klasa(trening: pd.DataFrame, klase: list) -> dict:
    """
    Racuna tezine obrnuto proporcionalne ucestalosti klase.

    Bez ovoga model minimizuje gubitak tako sto zanemaruje najmalobrojniju
    klasu (COVID19), sto je za medicinsku primenu neprihvatljivo.
    """
    broj = trening["klasa"].value_counts()
    ukupno, k = len(trening), len(klase)
    return {i: ukupno / (k * broj[ime]) for i, ime in enumerate(klase)}


# ---------------------------------------------------------------- tf.data

def _iseci(slika, isecanje):
    """
    Uklanja pojaseve van grudnog kosa: vrat i ramena odozgo, trbuh odozdo,
    uske trake sa strana.

    Svrha je da se modelu oduzme mogucnost da odluku donosi na osnovu
    oblasti koje ne sadrze plucno tkivo. Isecanje se radi PRE skaliranja,
    da bi preostali deo zadrzao punu rezoluciju.
    """
    oblik = tf.shape(slika)
    v, s = oblik[0], oblik[1]

    gore = tf.cast(tf.cast(v, tf.float32) * isecanje["gore"], tf.int32)
    dole = tf.cast(tf.cast(v, tf.float32) * isecanje["dole"], tf.int32)
    strana = tf.cast(tf.cast(s, tf.float32) * isecanje["strane"], tf.int32)

    return tf.image.crop_to_bounding_box(
        slika, gore, strana, v - gore - dole, s - 2 * strana)


def _ucitaj_snimak(putanja, oznaka, visina, sirina, isecanje=None):
    bajtovi = tf.io.read_file(putanja)
    # channels=3 automatski prevodi jednokanalne (grayscale) snimke u tri kanala
    slika = tf.io.decode_image(bajtovi, channels=3, expand_animations=False)
    if isecanje and isecanje.get("aktivno"):
        slika = _iseci(slika, isecanje)
    slika = tf.image.resize(slika, [visina, sirina], method="bilinear")
    # Cuva se kao uint8: kes tada trosi cetiri puta manje memorije
    slika = tf.cast(slika, tf.uint8)
    slika.set_shape([visina, sirina, 3])
    return slika, oznaka


def napravi_dataset(df: pd.DataFrame, klase: list, cfg: dict,
                    mesaj: bool = False, kesiraj: bool = True) -> tf.data.Dataset:
    """Gradi tf.data.Dataset iz tabele snimaka."""
    indeks = {ime: i for i, ime in enumerate(klase)}
    putanje = df["putanja"].to_numpy()
    oznake = df["klasa"].map(indeks).to_numpy().astype("int32")

    v, s = cfg["slika"]["visina"], cfg["slika"]["sirina"]
    isecanje = cfg.get("isecanje")
    ds = tf.data.Dataset.from_tensor_slices((putanje, oznake))
    ds = ds.map(lambda p, o: _ucitaj_snimak(p, o, v, s, isecanje),
                num_parallel_calls=AUTOTUNE)

    # Redosled operacija je bitan. Kesira se posle dekodovanja, a PRE mesanja
    # i grupisanja. Kada bi kes bio posle batch(), u prvoj epohi bi se zapamtili
    # gotovi batch-evi, pa se mesanje u narednim epohama ne bi ni izvrsavalo -
    # model bi svaku epohu video identicne batch-eve istim redosledom.
    if kesiraj:
        ds = ds.cache()

    if mesaj:
        # Bafer obuhvata ceo skup: delimican bafer bi pri sortiranom ulazu
        # ostavio uzastopne primere iste klase u istom batch-u.
        bafer = cfg["podaci"].get("mesaj_bafer") or 0
        ds = ds.shuffle(max(bafer, len(df)),
                        seed=cfg["seed"], reshuffle_each_iteration=True)

    ds = ds.batch(cfg["podaci"]["batch_size"])
    # Konverzija u float32 tek na kraju, opseg ostaje [0, 255]
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y),
                num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def slojevi_augmentacije(cfg: dict) -> keras.Sequential:
    """
    Slojevi vestackog prosirivanja skupa, aktivni samo tokom treninga.

    Horizontalno preslikavanje je namerno iskljuceno: rendgenski snimak
    grudnog kosa ima fiksnu anatomsku orijentaciju (srce levo), pa bi
    preslikavanje unelo anatomski nemoguce primere.
    """
    a = cfg["augmentacija"]
    slojevi = []
    if a.get("horizontalno_ogledalo"):
        slojevi.append(layers.RandomFlip("horizontal"))
    slojevi += [
        layers.RandomRotation(a["rotacija"], fill_mode="constant"),
        layers.RandomZoom(a["zum"], fill_mode="constant"),
        layers.RandomTranslation(a["pomeraj"], a["pomeraj"],
                                          fill_mode="constant"),
        layers.RandomContrast(a["kontrast"]),
    ]
    return keras.Sequential(slojevi, name="augmentacija")


def pripremi_sve(cfg: dict = None, ogranici: int = None):
    """
    Prakticna funkcija: od konfiguracije do gotovih skupova.

    Parametar 'ogranici' sluzi za brzu lokalnu proveru koda na malom
    uzorku, bez cekanja na obradu celog skupa.
    """
    cfg = cfg or ucitaj_konfiguraciju()
    postavi_seme(cfg["seed"])

    df = indeksiraj_snimke()
    klase = imena_klasa(df)

    if ogranici:
        delovi = [g.sample(min(len(g), ogranici), random_state=cfg["seed"])
                  for _, g in df.groupby(["podskup", "klasa"])]
        df = pd.concat(delovi, ignore_index=True)

    trening, validacioni, test = podeli_podatke(
        df, cfg["podaci"]["udeo_validacionog"], cfg["seed"])

    skupovi = {
        "trening": napravi_dataset(trening, klase, cfg, mesaj=True),
        "validacioni": napravi_dataset(validacioni, klase, cfg),
        "test": napravi_dataset(test, klase, cfg),
    }
    tabele = {"trening": trening, "validacioni": validacioni, "test": test}
    return skupovi, tabele, klase, cfg