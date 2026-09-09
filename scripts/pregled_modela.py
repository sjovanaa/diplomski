"""
Pregled arhitektura - generise tabelu za poglavlje o predlozenom resenju.

Ne zahteva podatke ni GPU. Gradi sva cetiri modela, prebrojava parametre
i ispisuje uporednu tabelu. Prvo pokretanje preuzima ImageNet tezine
(~100 MB ukupno).

Pokretanje iz korena projekta:
    python -m scripts.pregled_modela
"""
import pandas as pd

from src.data import ucitaj_konfiguraciju
from src.models import (PUNA_IMENA, broj_parametara, dubina_osnove,
                        napravi_model, odmrzni_osnovu)
from src.paths import direktorijum_rezultata

BROJ_KLASA = 3


def main():
    cfg = ucitaj_konfiguraciju()
    udeo = cfg["trening"].get("udeo_odmrznutih", 0.3)
    redovi = []

    for ime in cfg["modeli"]:
        print(f"\nGradim model: {PUNA_IMENA[ime]} ...")
        model = napravi_model(ime, BROJ_KLASA, cfg)

        faza1 = broj_parametara(model)
        odmrznuto = odmrzni_osnovu(model, udeo)
        faza2 = broj_parametara(model)

        redovi.append({
            "Model": PUNA_IMENA[ime],
            "Ukupno parametara": faza1["ukupno"],
            "Obucivi (faza 1)": faza1["obucivi"],
            "Obucivi (faza 2)": faza2["obucivi"],
            "Odmrznuto slojeva": odmrznuto,
            "Dubina osnove": dubina_osnove(model),
        })

    tab = pd.DataFrame(redovi)

    formatirana = tab.copy()
    for kol in ["Ukupno parametara", "Obucivi (faza 1)", "Obucivi (faza 2)"]:
        formatirana[kol] = formatirana[kol].map(lambda v: f"{v:,}".replace(",", "."))

    print("\n" + "=" * 78)
    print("UPOREDNI PREGLED ARHITEKTURA")
    print("=" * 78)
    print(formatirana.to_string(index=False))
    print("=" * 78)

    putanja = direktorijum_rezultata() / "tabela_3_arhitekture.csv"
    tab.to_csv(putanja, index=False)
    print(f"\nSacuvano: {putanja}")


if __name__ == "__main__":
    main()