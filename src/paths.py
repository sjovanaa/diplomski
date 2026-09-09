"""
Lociranje skupa podataka i izlaznih direktorijuma.

Isti kod se izvrsava lokalno (Windows), na Kaggle-u i na Colab-u,
pa se putanje nikada ne upisuju rucno.
"""
from pathlib import Path

KOREN_PROJEKTA = Path(__file__).resolve().parent.parent

KAGGLE_SKUP = "prashant268/chest-xray-covid19-pneumonia"

def _trazi_koren(baza: Path):
    for nivo1 in [baza] + [d for d in baza.iterdir() if d.is_dir()]:
        if (nivo1 / "train").is_dir() and (nivo1 / "test").is_dir():
            return nivo1
        for nivo2 in [d for d in nivo1.iterdir() if d.is_dir()]:
            if (nivo2 / "train").is_dir() and (nivo2 / "test").is_dir():
                return nivo2
            for nivo3 in [d for d in nivo2.iterdir() if d.is_dir()]:
                if (nivo3 / "train").is_dir() and (nivo3 / "test").is_dir():
                    return nivo3
    return None


def pronadji_skup_podataka(dozvoli_preuzimanje: bool = True) -> Path:
    """
    Vraca direktorijum koji sadrzi podfoldere 'train' i 'test'.

    Redosled provere:
      1. /kaggle/input   - skup je vec montiran, nista se ne preuzima
      2. lokalni folder 'podaci/' ili '/content/data'
      3. kagglehub kes   - ranije preuzet skup
      4. preuzimanje preko kagglehub (~1,15 GB, samo prvi put)
    """
    for baza in [Path("/kaggle/input"), KOREN_PROJEKTA / "podaci",
                 Path("/content/data"), Path("data"),
                 Path.home() / ".cache" / "kagglehub"]:
        koren = _trazi_koren(baza)
        if koren:
            return koren

    if dozvoli_preuzimanje:
        try:
            import kagglehub
            print("Skup nije pronadjen lokalno. Preuzimanje preko kagglehub "
                  "(~1,15 GB, samo prvi put)...")
            putanja = Path(kagglehub.dataset_download(KAGGLE_SKUP))
            koren = _trazi_koren(putanja)
            if koren:
                print("Preuzeto u:", koren)
                return koren
        except ImportError:
            pass

    raise FileNotFoundError(
        "Skup podataka nije pronadjen.\n"
        "Varijanta 1: pip install kagglehub, pa ponovo pokreni "
        "(trazice Kaggle korisnicko ime i API kljuc).\n"
        "Varijanta 2: rucno raspakuj arhivu u folder 'podaci/'.\n"
        "Varijanta 3: na Kaggle-u dodaj skup preko dugmeta 'Add Input'."
    )


def direktorijum_rezultata() -> Path:
    """Vraca direktorijum u koji se upisuju modeli, metrike i grafikoni."""
    if Path("/kaggle/working").exists():
        izlaz = Path("/kaggle/working/rezultati")
    else:
        izlaz = KOREN_PROJEKTA / "rezultati"
    izlaz.mkdir(parents=True, exist_ok=True)
    return izlaz


if __name__ == "__main__":
    print("Koren projekta:  ", KOREN_PROJEKTA)
    print("Skup podataka:   ", pronadji_skup_podataka())
    print("Rezultati:       ", direktorijum_rezultata())
