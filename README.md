



<div align="center">
# 🎮 Jeu de Dames avec IA
![Head](https://github.com/user-attachments/assets/7ecd786f-8557-4b6e-ae35-48263f7481d6)
Un jeu de dames intelligent développé en Python, combinant apprentissage par renforcement et algorithme Minimax pour une expérience de jeu stimulante.
[Démonstration en vidéo](votre-url)
</div>

## ✨ Fonctionnalités
- 🤖 IA hybride (Q-learning + Minimax)
- 🎯 3 niveaux de difficulté
- 🎮 Interface graphique fluide
- 🔄 Apprentissage multi-processus
- 💡 Prises multiples intelligentes
- 👑 Gestion avancée des dames

## 🛠️ Technologies Utilisées
- Python
- Pygame
- NumPy
- Multiprocessing
- Pickle
- Random

## 📦 Installation
1. **Clonez le dépôt**
   ```bash
   git clone https://github.com/votre-username/jeu-dames-ia
   ```
2. **Installez les dépendances**
   ```bash
   pip install pygame numpy
   ```
3. **Lancez le jeu**
   ```bash
   python jouer_ai.py
   ```

## 📚 Structure du Projet
```
jeu-dames-ia/
│
├── jouer_ai.py        # Interface du jeu
├── entrainement.py    # Système d'apprentissage
├── jeu_dames.py       # Logique du jeu
└── modeles/           # Modèles entraînés
    ├── modele_p500.pkl
    └── modele_final.pkl
```

## 🔋 Fonctionnalités Détaillées

### 🎮 Interface de Jeu
- Interface graphique intuitive
- Animation des coups de l'IA
- Indication des coups possibles
- Marquage des prises obligatoires

### 🧠 Système d'IA
- Q-learning pour l'apprentissage par expérience
- Minimax avec élagage alpha-beta
- Système de récompenses élaboré
- Paramètres ajustables selon la difficulté

### 🚀 Entraînement
- Apprentissage multi-processus
- Sauvegarde périodique des modèles
- Statistiques d'entraînement détaillées
- Gestion des interruptions

## 💻 Utilisation

### Jouer une partie
```python
# Mode normal
lancer_jeu_contre_ia("modele_p500.pkl", BLANC)

# Mode difficile
lancer_jeu_contre_ia("modele_final.pkl", BLANC, 'difficile')
```

### Entraîner un nouveau modèle
```python
entraineur = EntraineurIA()
entraineur.entrainer(nb_parties=1000)
```

## 🔬 Aspects Techniques
- Architecture modulaire et extensible
- Gestion efficace de la mémoire partagée
- Optimisation des performances avec multiprocessing
- Système robuste de gestion des erreurs

## 🙏 Remerciements
- La communauté Python pour les bibliothèques exceptionnelles
- Les ressources en ligne sur l'apprentissage par renforcement
- Les retours constructifs des testeurs

## 📫 Contact
Samuel Alhadef - [@SAMUELALHADEF](https://x.com/SAMUELALHADEF)
Lien du projet: [https://github.com/votre-username/jeu-dames-ia](https://github.com/votre-username/jeu-dames-ia)

---
<div align="center">
Fait avec ❤️ par [Samuel Alhadef](https://github.com/Samuelalhadef)
</div>
