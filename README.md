# Primena dubokog ucenja za klasifikaciju plucnih oboljenja

Diplomski rad — Elektrotehnicki fakultet Univerziteta u Beogradu

Poredjenje cetiri modela dubokog ucenja (sopstvena CNN arhitektura, ResNet50,
DenseNet121, EfficientNet-B0) na skupu rendgenskih snimaka grudnog kosa
podeljenom u tri klase: COVID19, NORMAL, PNEUMONIA.

---

## 1. Podesavanje na Windows-u (PyCharm)

**Kreiranje projekta**

1. Raspakuj ovaj folder na disk
2. PyCharm → `File` → `Open` → izaberi folder `diplomski`
3. `File` → `Settings` → `Project` → `Python Interpreter` → zupcanik → `Add`
   → `Virtualenv Environment` → `New` → OK

**Instalacija biblioteka** (u PyCharm terminalu, dole levo):

```
pip install -r requirements.txt
```

Instalira se `tensorflow-cpu`. To je namerno — bez NVIDIA kartice GPU verzija
nema efekta, a pravi probleme pri instalaciji na Windows-u.

**Preuzimanje podataka**

Otvori https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia
i klikni `Download` (~1,15 GB). Raspakuj arhivu u folder `podaci/` tako da
postoje putanje `podaci/.../train/` i `podaci/.../test/`. Tacna dubina nije
bitna — kod sam pronalazi koren skupa.

**Provera da sve radi**

```
python -m scripts.provera
```

Ocekivani ishod: ispis velicina podskupova, tezina klasa i jednog batch-a,
plus slika `rezultati/provera_batch.png`. Ako se ovo izvrsi bez greske,
kod je spreman za trening na GPU-u.

---

## 2. Trening na Kaggle-u (GPU)

Lokalna masina sluzi za pisanje koda, a treniranje se izvrsava na Kaggle-u.

1. Postavi projekat na GitHub (PyCharm: `VCS` → `Share Project on GitHub`)
2. kaggle.com → `Create` → `New Notebook`
3. Desni panel → `Add Input` → `chest-xray-covid19-pneumonia`
4. Desni panel → `Accelerator` → `GPU T4 x2` ili `GPU P100`
5. U prvoj celiji:

```python
!git clone https://github.com/KORISNIK/diplomski.git
%cd diplomski
!python -m scripts.provera --ogranici 40
```

Skup podataka se ne preuzima — vec je montiran pod `/kaggle/input/`.

---

## 3. Struktura projekta

```
diplomski/
├── config.yaml           svi hiperparametri na jednom mestu
├── requirements.txt
├── src/
│   ├── paths.py          lociranje podataka i izlaza u bilo kom okruzenju
│   └── data.py           indeksiranje, podela, tf.data pipeline, tezine klasa
├── scripts/
│   └── provera.py        brza provera ispravnosti pripreme podataka
├── podaci/               skup podataka (nije u git-u)
└── rezultati/            modeli, metrike, grafikoni (nije u git-u)
```

## 4. Reproduktivnost

Seme slucajnih brojeva je fiksirano u `config.yaml` i primenjuje se na
`random`, `numpy` i `tensorflow`. Podela na validacioni skup je stratifikovana
i deterministicna, pa dva pokretanja daju identicnu podelu podataka.
