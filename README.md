# MTG_image_recognition
Projekat za predmet "Mašinskom učenje"

Ideja i cilj projekta je pronalaženje načina da naučimo neuronsku mrežu da prepoznaje slike sa kamere pri čemu nemamo veliki broj slika sa kamere da koristimo kao trening skup. Pokušaćemo da napravimo veštački dataset metodom augmentacije slike. (Detaljnije će biti opisano u PDF-u kad ga odradim)

Dataset: Neaugmentovane slike uzete iz kartične igre Magic: The Gathering u punoj rezoluciji

# Setup 

Instalirati potrebne biblioteke u virtuelno okruženje.

```shell
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Projekat se pokreće kroz notebook `notebooks/pipeline.ipynb`

