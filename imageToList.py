from PIL import Image

def itl(path):
    # Apri l'immagine e converti in scala di grigi (L = luminosità 0-255)
    img = Image.open(path).convert('L')
    # Ridimensiona a 28x28 se non lo è già
    img = img.resize((28, 28))

    matrice = []
    for y in range(28):
        riga = []
        for x in range(28):
            valore = img.getpixel((x, y))
            riga.append(valore)
        matrice.append(riga)

    return matrice
