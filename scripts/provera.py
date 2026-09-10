"""
Brza lokalna provera ispravnosti pripreme podataka.

Pokrece se na CPU-u za nekoliko sekundi i sluzi da se greske u kodu otkriju
PRE nego sto se pokrene visecasovni trening na GPU-u.

Pokretanje iz korena projekta:
    python -m scripts.provera
    python -m scripts.provera --ogranici 40
"""
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np

from src.data import pripremi_sve, tezine_klasa
from src.paths import direktorijum_rezultata, pronadji_skup_podataka


def main(ogranici: int):
    print("=" * 60)
    print("PROVERA PRIPREME PODATAKA")
    print("=" * 60)
    print("Skup podataka:", pronadji_skup_podataka())

    skupovi, tabele, klase, cfg = pripremi_sve(ogranici=ogranici)

    print("\nKlase:", klase)
    print("\nVelicine podskupova:")
    for ime, tab in tabele.items():
        raspodela = tab["klasa"].value_counts().reindex(klase).to_dict()
        print(f"  {ime:<12} {len(tab):>6} snimaka   {raspodela}")

    tez = tezine_klasa(tabele["trening"], klase)
    print("\nTezine klasa:")
    for i, ime in enumerate(klase):
        print(f"  {ime:<12} {tez[i]:.3f}")

    # Provera jednog batch-a: oblik, tip, opseg vrednosti
    slike, oznake = next(iter(skupovi["trening"]))
    print("\nJedan batch:")
    print(f"  oblik slika:  {tuple(slike.shape)}")
    print(f"  tip:          {slike.dtype}")
    print(f"  opseg:        [{float(np.min(slike)):.1f}, {float(np.max(slike)):.1f}]")
    print(f"  oblik oznaka: {tuple(oznake.shape)}  primer: {oznake[:8].numpy()}")

    assert slike.shape[1:] == (cfg["slika"]["visina"], cfg["slika"]["sirina"], 3), \
        "Neocekivan oblik slike."
    assert float(np.max(slike)) > 1.5, \
        "Vrednosti su vec normalizovane - normalizacija pripada modelu, ne pipeline-u."

    # Vizuelna kontrola: sacuvaj mrezu snimaka iz batch-a
    izlaz = direktorijum_rezultata() / "provera_batch.png"
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for i, osa in enumerate(ax.ravel()):
        osa.imshow(slike[i].numpy().astype("uint8"))
        osa.set_title(klase[int(oznake[i])], fontsize=10)
        osa.axis("off")
    plt.tight_layout()
    plt.savefig(izlaz, dpi=120)
    print(f"\nMreza snimaka sacuvana: {izlaz}")

    print("\n" + "=" * 60)
    print("SVE PROVERE SU PROSLE.")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ogranici", type=int, default=40,
                   help="broj snimaka po klasi i podskupu (0 = ceo skup)")
    args = p.parse_args()
    sys.exit(main(args.ogranici or None))