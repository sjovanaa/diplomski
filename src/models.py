"""
Definicije modela koji ulaze u poredjenje.

Cetiri arhitekture:
  1. cnn_od_nule      - sopstvena konvoluciona mreza, obucava se od nule
  2. resnet50         - transfer learning, ImageNet tezine
  3. densenet121      - transfer learning, ImageNet tezine
  4. efficientnetb0   - transfer learning, ImageNet tezine

Svi modeli primaju IDENTICAN ulaz: sirove snimke dimenzije 224x224x3 sa
vrednostima piksela u opsegu [0, 255]. Normalizacija specificna za
arhitekturu izvrsava se unutar modela, kao prvi sloj. Time je poredjenje
metodoloski ispravno - nijedan model nije u prednosti zbog pripreme podataka.
"""
import keras
import tensorflow as tf
from keras import layers

from .data import slojevi_augmentacije

# Funkcije normalizacije koje pripadaju pojedinacnim arhitekturama.
# ResNet50   : caffe stil - RGB u BGR, oduzimanje ImageNet srednjih vrednosti
# DenseNet121: torch stil - skaliranje na [0,1], pa standardizacija
# EfficientNet: ocekuje sirov opseg [0,255], normalizuje interno
FUNKCIJE_NORMALIZACIJE = {
    "resnet50": keras.applications.resnet50.preprocess_input,
    "densenet121": keras.applications.densenet.preprocess_input,
    "efficientnetb0": keras.applications.efficientnet.preprocess_input,
}

OSNOVE = {
    "resnet50": keras.applications.ResNet50,
    "densenet121": keras.applications.DenseNet121,
    "efficientnetb0": keras.applications.EfficientNetB0,
}

PUNA_IMENA = {
    "cnn_od_nule": "CNN (od nule)",
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "efficientnetb0": "EfficientNet-B0",
}


@keras.utils.register_keras_serializable(package="diplomski")
class Normalizacija(layers.Layer):
    """
    Omotac oko preprocess_input funkcije odgovarajuce arhitekture.

    Realizovan kao registrovani sloj (a ne kao Lambda) da bi se model mogao
    sacuvati i kasnije ucitati bez dodatnih parametara pri ucitavanju.
    """

    def __init__(self, arhitektura, **kwargs):
        super().__init__(**kwargs)
        self.arhitektura = arhitektura

    def call(self, x):
        return FUNKCIJE_NORMALIZACIJE[self.arhitektura](x)

    def get_config(self):
        cfg = super().get_config()
        cfg["arhitektura"] = self.arhitektura
        return cfg


# ---------------------------------------------------------------- CNN od nule

def _cnn_od_nule(ulaz, broj_klasa, dropout):
    """
    Sopstvena konvoluciona mreza - referentna tacka poredjenja.

    Cetiri konvoluciona bloka sa rastucim brojem filtera (32-64-128-256).
    Svaki blok: konvolucija -> batch normalizacija -> ReLU -> uzorkovanje.
    Batch normalizacija ubrzava konvergenciju i deluje regularizujuce,
    sto je vazno jer se mreza obucava bez ikakvog prethodnog znanja.
    """
    x = layers.Rescaling(1.0 / 255)(ulaz)

    for filtera in (32, 64, 128, 256):
        x = layers.Conv2D(filtera, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filtera, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)

    # Globalno usrednjavanje umesto Flatten: drasticno manje parametara
    # i manja sklonost preprilagodjavanju na skupu ove velicine.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    return layers.Dense(broj_klasa, activation="softmax", name="izlaz")(x)


# ---------------------------------------------------------------- transfer learning

def _transfer(ulaz, ime, broj_klasa, dropout, oblik):
    """
    Pretrenirana osnova sa novom klasifikacionom glavom.

    Osnova se NE pravi sa parametrom input_tensor, jer bi tada Keras njene
    slojeve ugradio u spoljni graf. Umesto toga se gradi zasebno i poziva
    kao jedan sloj, cime ostaje ugnezden model nad kojim se kasnije moze
    kontrolisano odmrznuti gornji deo.

    Argument training=False obezbedjuje da slojevi batch normalizacije rade
    u rezimu zakljucenja i u fazi finog podesavanja.
    """
    x = Normalizacija(ime, name="normalizacija")(ulaz)

    osnova = OSNOVE[ime](include_top=False, weights="imagenet", input_shape=oblik)
    osnova.trainable = False          # faza 1: osnova je zamrznuta
    x = osnova(x, training=False)

    x = layers.GlobalAveragePooling2D(name="usrednjavanje")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    return layers.Dense(broj_klasa, activation="softmax", name="izlaz")(x)


# ---------------------------------------------------------------- javni interfejs

def napravi_model(ime: str, broj_klasa: int, cfg: dict) -> keras.Model:
    """Gradi model po imenu iz konfiguracije."""
    if ime not in PUNA_IMENA:
        raise ValueError(f"Nepoznat model '{ime}'. Dostupni: {list(PUNA_IMENA)}")

    v, s, k = (cfg["slika"]["visina"], cfg["slika"]["sirina"], cfg["slika"]["kanali"])
    dropout = cfg["trening"].get("dropout", 0.3)

    ulaz = keras.Input(shape=(v, s, k), name="ulaz")
    # Augmentacija je deo modela, ali je aktivna iskljucivo u fazi obucavanja
    x = slojevi_augmentacije(cfg)(ulaz)

    if ime == "cnn_od_nule":
        izlaz = _cnn_od_nule(x, broj_klasa, dropout)
    else:
        izlaz = _transfer(x, ime, broj_klasa, dropout, (v, s, k))

    return keras.Model(ulaz, izlaz, name=ime)


def pronadji_osnovu(model: keras.Model):
    """
    Vraca ugnezdenu pretreniranu osnovu, ili None ako je model 'cnn_od_nule'.

    Sloj augmentacije se izostavlja izricito: Sequential je podklasa klase
    Model, pa bi inace bio prepoznat kao osnova.
    """
    kandidati = [sloj for sloj in model.layers
                 if isinstance(sloj, keras.Model) and sloj.name != "augmentacija"]
    return kandidati[0] if kandidati else None


def dubina_osnove(model: keras.Model) -> int:
    osnova = pronadji_osnovu(model)
    return len(osnova.layers) if osnova else len(model.layers)


def odmrzni_osnovu(model: keras.Model, udeo: float = 0.3) -> int:
    """
    Priprema modela za drugu fazu - fino podesavanje.

    Odmrzava se samo gornjih 'udeo' slojeva osnove, jer donji slojevi
    prepoznaju opsta obelezja (ivice, teksture) koja vaze i za rendgenske
    snimke, dok gornji kodiraju obelezja specificna za ImageNet.

    Slojevi batch normalizacije OSTAJU zamrznuti. Njihove pokretne
    statistike su izracunate na milionima ImageNet slika i njihovo
    azuriranje na malom skupu bi ih pokvarilo - to je najcesci uzrok
    naglog pada tacnosti pri finom podesavanju.

    Vraca broj slojeva koji se obucavaju.
    """
    osnova = pronadji_osnovu(model)
    if osnova is None:
        return 0        # cnn_od_nule nema zamrznutu osnovu

    osnova.trainable = True
    granica = int(len(osnova.layers) * (1 - udeo))

    obucava_se = 0
    for i, sloj in enumerate(osnova.layers):
        if i < granica or isinstance(sloj, layers.BatchNormalization):
            sloj.trainable = False
        else:
            sloj.trainable = True
            obucava_se += 1
    return obucava_se


def broj_parametara(model: keras.Model) -> dict:
    ukupno = model.count_params()
    obucivi = sum(int(tf.size(v)) for v in model.trainable_variables)
    return {"ukupno": ukupno, "obucivi": obucivi, "neobucivi": ukupno - obucivi}