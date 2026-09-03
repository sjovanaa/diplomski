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

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
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
    tf.keras.utils.set_random_seed(seed)


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

def _ucitaj_snimak(putanja, oznaka, visina, sirina):
    bajtovi = tf.io.read_file(putanja)
    # channels=3 automatski prevodi jednokanalne (grayscale) snimke u tri kanala
    slika = tf.io.decode_image(bajtovi, channels=3, expand_animations=False)
    slika = tf.image.resize(slika, [visina, sirina], method="bilinear")
    slika = tf.cast(slika, tf.float32)          # opseg ostaje [0, 255]
    slika.set_shape([visina, sirina, 3])
    return slika, oznaka


def napravi_dataset(df: pd.DataFrame, klase: list, cfg: dict,
                    mesaj: bool = False, kesiraj: bool = True) -> tf.data.Dataset:
    """Gradi tf.data.Dataset iz tabele snimaka."""
    indeks = {ime: i for i, ime in enumerate(klase)}
    putanje = df["putanja"].to_numpy()
    oznake = df["klasa"].map(indeks).to_numpy().astype("int32")

    v, s = cfg["slika"]["visina"], cfg["slika"]["sirina"]
    ds = tf.data.Dataset.from_tensor_slices((putanje, oznake))

    if mesaj:
        ds = ds.shuffle(min(len(df), cfg["podaci"]["mesaj_bafer"]),
                        seed=cfg["seed"], reshuffle_each_iteration=True)

    ds = ds.map(lambda p, o: _ucitaj_snimak(p, o, v, s), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(cfg["podaci"]["batch_size"])
    if kesiraj:
        ds = ds.cache()
    return ds.prefetch(AUTOTUNE)


def slojevi_augmentacije(cfg: dict) -> tf.keras.Sequential:
    """
    Slojevi vestackog prosirivanja skupa, aktivni samo tokom treninga.

    Horizontalno preslikavanje je namerno iskljuceno: rendgenski snimak
    grudnog kosa ima fiksnu anatomsku orijentaciju (srce levo), pa bi
    preslikavanje unelo anatomski nemoguce primere.
    """
    a = cfg["augmentacija"]
    slojevi = []
    if a.get("horizontalno_ogledalo"):
        slojevi.append(tf.keras.layers.RandomFlip("horizontal"))
    slojevi += [
        tf.keras.layers.RandomRotation(a["rotacija"], fill_mode="constant"),
        tf.keras.layers.RandomZoom(a["zum"], fill_mode="constant"),
        tf.keras.layers.RandomTranslation(a["pomeraj"], a["pomeraj"],
                                          fill_mode="constant"),
        tf.keras.layers.RandomContrast(a["kontrast"]),
    ]
    return tf.keras.Sequential(slojevi, name="augmentacija")


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
        df = (df.groupby(["podskup", "klasa"], group_keys=False)
                .apply(lambda g: g.sample(min(len(g), ogranici),
                                          random_state=cfg["seed"])))

    trening, validacioni, test = podeli_podatke(
        df, cfg["podaci"]["udeo_validacionog"], cfg["seed"])

    skupovi = {
        "trening": napravi_dataset(trening, klase, cfg, mesaj=True),
        "validacioni": napravi_dataset(validacioni, klase, cfg),
        "test": napravi_dataset(test, klase, cfg),
    }
    tabele = {"trening": trening, "validacioni": validacioni, "test": test}
    return skupovi, tabele, klase, cfg
