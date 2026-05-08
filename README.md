# Smart Hand Mouse

Pilotez la souris Windows avec les gestes de la main, en local, sans GPU, sans entraînement.

## Démo

| Geste | Action |
|---|---|
| ✋ index seul levé | Bouge le curseur |
| 🤏 pince pouce-index | Clic gauche |
| 🖐️ 3 doigts (index + majeur + annulaire) | Clic droit |
| ✊ poing fermé | Scroll vertical (haut si main en haut, bas si main en bas) |
| ✌️ V (peace) | Pause / Reprise |

Touche `Q` pour quitter.

## Setup (Windows, 4 commandes)

```powershell
cd hand_mouse_demo
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
python hand_mouse.py
```

→ Une fenêtre s'ouvre avec votre webcam. Levez l'index, le curseur Windows bouge derrière la fenêtre.

## Comment ça marche

Le programme utilise **MediaPipe Hands** (Google) — un modèle pré-entraîné qui détecte 21 points de votre main à 30 fps sur le CPU.

### Les 21 points

```
   ____________
  |    8       |    8 = bout de l'index
  |   /|       |    4 = bout du pouce
  |  / |       |    9 = centre de la paume
  | 7  6       |    
  |/   |       |    Pour chaque doigt :
  4----5       |     - 0..3 = pouce
   \  /        |     - 4..8 = index
    \/_________|
```

### Le pipeline

```
webcam → MediaPipe → 21 points
                          ↓
                  Quels doigts levés ?
                  Pince active ?
                          ↓
              Selon le geste : pyautogui agit sur Windows
                          ↓
         Overlay (état + geste + FPS) → cv2.imshow
```

## Réglages (en haut de hand_mouse.py)

| Variable | Effet | Valeur conseillée |
|---|---|---|
| `SMOOTHING` | 0 = curseur saccadé, 1 = curseur immobile | 0.3 |
| `PINCH_THRESHOLD` | Distance pouce-index sous laquelle = clic | 0.05 (~1 cm visible) |
| `ACTIVE_ZONE` | Part centrale de la cam mappée sur l'écran | 0.7 (70%) |
| `CLICK_COOLDOWN` | Délai mini entre 2 clics (anti-spam) | 0.4 s |

## Bibliothèques utilisées

- **mediapipe** : modèle Google pour détecter les mains (21 points par main).
- **opencv-python** : webcam + affichage + dessin par-dessus l'image.
- **pyautogui** : contrôle souris/clavier Windows.

## Le code en 8 sections

```
1. RÉGLAGES               → constantes de config
2. ÉTAT                   → dictionnaire avec last_x, last_y, etc.
3. LECTURE DES DOIGTS     → fingers_up(), pinch_distance(), map_to_screen()
4. ACTIONS                → can_click(), do_click(), move_cursor()
5. RECONNAISSANCE         → handle_gesture() : if/elif sur le geste
6. OVERLAY                → draw_overlay() : texte + rectangle
7. BOUCLE PRINCIPALE      → main() : webcam → mediapipe → action
8. POINT D'ENTRÉE         → if __name__ == "__main__"
```

Lecture du code : ouvrir `hand_mouse.py`, lire de haut en bas. Chaque fonction tient en moins de 15 lignes.

## Erreurs courantes

| Symptôme | Cause | Fix |
|---|---|---|
| Le curseur tremble | `SMOOTHING` trop bas | Passer à `0.5` |
| Clic ne se déclenche pas | Seuil pince trop strict | `PINCH_THRESHOLD = 0.07` |
| Trop de clics rapprochés | Cooldown court | `CLICK_COOLDOWN = 0.6` |
| Le curseur n'atteint pas les coins | Zone active trop petite | `ACTIVE_ZONE = 0.85` |
| Webcam ne s'ouvre pas | Autre app l'utilise (Teams, Zoom) | Fermer l'autre app |
| `Failsafe triggered` | Curseur dans le coin haut-gauche | déjà désactivé via `pyautogui.FAILSAFE = False` |

## Idées d'extensions

| Niveau | Idée |
|---|---|
| Facile | Geste 4 doigts → ouvrir le menu Démarrer (`pyautogui.press("win")`) |
| Moyen | Détecter un **double clic** (deux pinces rapides) |
| Moyen | **Drag & drop** : pince maintenue = bouton enfoncé, relâcher = drop |
| Difficile | Combiner **2 mains** : main droite = curseur, main gauche = boutons |
| Créatif | Shortcuts personnalisés : signe spécifique = ouvrir une app précise |
| Wow | **Sound feedback** : `winsound.Beep(800, 50)` à chaque clic |

## Sécurité / vie privée

- Tout tourne **en local sur votre machine**.
- Aucune image n'est envoyée sur Internet.
- Le modèle MediaPipe (~5 Mo) est inclus dans le paquet pip — pas de téléchargement séparé.

## Pour démo en classe

1. Lancer `python hand_mouse.py`.
2. Réduire la fenêtre OpenCV à 1/3 de l'écran (pour voir Windows derrière).
3. **Ouvrir Chrome** avec l'index et la pince.
4. Faire défiler une page Wikipedia avec le poing.
5. Faire un clic droit sur un onglet pour le fermer.
6. Mettre en **pause** avec ✌️ pour récupérer la vraie souris.

L'effet en classe : 2 minutes de wow garanti, surtout si on enchaîne avec « et tout ce code tient en 200 lignes ».

---

**Walid Ghouili — Mai 2026**
