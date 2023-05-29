import cv2
import numpy as np

# Funzione per applicare un filtro blu alle immagini
def applica_filtro_blu(img):
    return cv2.filter2D(img, -1, np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 0]], dtype=np.float32))

# Funzione per regolare il contrasto delle immagini
def regola_contrasto(img, contrasto):
    return cv2.convertScaleAbs(img, alpha=contrasto, beta=0)

# Funzione per aggiungere rumore alle immagini
def aggiungi_rumore(img, intensita_rumore):
    rumore = np.random.normal(0, intensita_rumore, img.shape).astype(np.uint8)
    img_rumore = cv2.add(img, rumore)
    return np.clip(img_rumore, 0, 255)

# Carica l'immagine di giorno

def image_to_night(img_giorno)

# Applica il filtro blu per simulare l'illuminazione notturna
img_notte = applica_filtro_blu(img_giorno)

# Regola il contrasto dell'immagine
img_notte = regola_contrasto(img_notte, contrasto=0.7)

# Aggiungi rumore all'immagine
img_notte = aggiungi_rumore(img_notte, intensita_rumore=30)

# Visualizza l'immagine di notte
cv2.imshow('Immagine di notte', img_notte)
cv2.waitKey(0)
cv2.destroyAllWindows()

return img_notte
