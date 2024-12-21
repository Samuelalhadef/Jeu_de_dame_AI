import pygame
import pickle
import time
import random
from multiprocessing import Manager
from jeu_dames import JeuDames, Plateau, BLANC, NOIR
from entrainement import JoueurIA, EtatPlateau

# Constantes du jeu
TAILLE_CASE = 80
NB_CASES = 8
LARGEUR = TAILLE_CASE * NB_CASES
HAUTEUR = TAILLE_CASE * NB_CASES + 40

# Couleurs
COULEUR_FOND = (255, 255, 255)
COULEUR_TEXTE = (0, 0, 0)
COULEUR_REFLEXION = (70, 130, 180)


class JeuContreIA(JeuDames):
    def __init__(self, chemin_ia, couleur_joueur=BLANC, difficulte='normale'):
        super().__init__()
        self.couleur_joueur = couleur_joueur
        self.couleur_ia = NOIR if couleur_joueur == BLANC else BLANC

        # Initialiser le gestionnaire de ressources partagées
        self.manager = Manager()
        self.q_table_shared = self.manager.dict()
        self.lock = self.manager.Lock()

        # Paramètres selon la difficulté
        params = self._obtenir_params_difficulte(difficulte)

        # Charger l'IA
        try:
            with open(chemin_ia, 'rb') as f:
                q_table = pickle.load(f)
                self.q_table_shared.update(q_table)
            print("IA chargée avec succès!")
        except Exception as e:
            print(f"Erreur lors du chargement de l'IA: {e}")
            print("L'IA jouera sans apprentissage préalable.")

        # Créer l'IA avec les paramètres
        self.ia = JoueurIA(
            self.couleur_ia, self.q_table_shared, self.lock, params)

        # Variables pour l'animation et le gameplay
        self.ia_est_en_train_de_reflechir = False
        self.debut_reflexion = 0
        self.duree_reflexion = 0
        self.piece_ia_selectionnee = None
        self.derniere_piece_bougee = None
        self.prise_obligatoire = False
        self.prochain_coup_ia = None
        self.message_statut = ""

    def _obtenir_params_difficulte(self, difficulte):
        """Configure les paramètres selon le niveau de difficulté"""
        if difficulte == 'facile':
            return {
                'epsilon': 0.3,  # Plus d'exploration
                'epsilon_decay': 1.0,
                'epsilon_min': 0.3,
                'alpha': 0.1,
                'alpha_decay': 1.0,
                'alpha_min': 0.1,
                'gamma': 0.8,  # Moins de prévision à long terme
                'poids_minimax': 0.5,
                'poids_qlearning': 0.5
            }
        elif difficulte == 'difficile':
            return {
                'epsilon': 0.01,  # Presque pas d'exploration
                'epsilon_decay': 1.0,
                'epsilon_min': 0.01,
                'alpha': 0.1,
                'alpha_decay': 1.0,
                'alpha_min': 0.1,
                'gamma': 0.99,  # Maximum de prévision
                'poids_minimax': 0.9,
                'poids_qlearning': 0.1
            }
        else:  # normale
            return {
                'epsilon': 0.1,
                'epsilon_decay': 1.0,
                'epsilon_min': 0.1,
                'alpha': 0.1,
                'alpha_decay': 1.0,
                'alpha_min': 0.1,
                'gamma': 0.95,
                'poids_minimax': 0.7,
                'poids_qlearning': 0.3
            }

    def verifier_prises_possibles(self, piece):
        """Vérifie si une pièce a des prises possibles"""
        return bool(self.plateau.obtenir_prises_possibles(piece))

    def simuler_reflexion_ia(self):
        """Simule une réflexion de l'IA avec des temps variables"""
        if not self.ia_est_en_train_de_reflechir:
            self.ia_est_en_train_de_reflechir = True
            self.debut_reflexion = time.time()
            self.duree_reflexion = random.uniform(0.8, 2.0)
            self.message_statut = "L'IA réfléchit..."

            # Obtenir l'action de l'IA
            etat_actuel = EtatPlateau(
                [p for p in self.plateau.pieces if p is not None])
            self.prochain_coup_ia = self.ia.obtenir_action(
                self.plateau, entrainement=False)

            if self.prochain_coup_ia:
                self.piece_ia_selectionnee = self.prochain_coup_ia[0]
                self.piece_ia_selectionnee.selectionne = True

    def faire_jouer_ia(self):
        """Gère le tour de l'IA avec des animations"""
        if not self.ia_est_en_train_de_reflechir:
            self.simuler_reflexion_ia()
            return False

        if time.time() - self.debut_reflexion < self.duree_reflexion:
            return False

        if self.prochain_coup_ia:
            piece, sequence = self.prochain_coup_ia
            pieces_avant = len(self.plateau.pieces)

            if self.piece_ia_selectionnee:
                self.piece_ia_selectionnee.selectionne = False

            # Exécuter le mouvement
            self.plateau.executer_deplacement(piece, sequence)
            pieces_apres = len(self.plateau.pieces)

            # Gérer les prises multiples
            if pieces_apres < pieces_avant:
                prises_supplementaires = self.plateau.obtenir_prises_possibles(
                    piece)
                if prises_supplementaires:
                    self.derniere_piece_bougee = piece
                    self.prise_obligatoire = True
                    self.message_statut = "Prise obligatoire !"
                    self.ia_est_en_train_de_reflechir = False
                    return True

            # Fin du tour
            self.derniere_piece_bougee = None
            self.prise_obligatoire = False
            self.plateau.tour_blanc = not self.plateau.tour_blanc
            self.plateau.verifier_victoire()
            self.message_statut = ""

        self.ia_est_en_train_de_reflechir = False
        self.piece_ia_selectionnee = None
        return True

    def dessiner_plateau(self):
        """Dessine le plateau avec les indications de jeu"""
        super().dessiner_plateau()

        # Afficher le message de statut
        if self.message_statut:
            texte = self.police.render(self.message_statut, True,
                                       COULEUR_REFLEXION if self.ia_est_en_train_de_reflechir else COULEUR_TEXTE)
            self.fenetre.blit(
                texte, (LARGEUR // 2 - texte.get_width() // 2, HAUTEUR - 30))

    def executer(self):
        """Boucle principale du jeu"""
        clock = pygame.time.Clock()

        # Si le joueur est noir, l'IA (blanche) commence
        if self.couleur_joueur == NOIR:
            self.simuler_reflexion_ia()

        while True:
            for evenement in pygame.event.get():
                if evenement.type == pygame.QUIT:
                    pygame.quit()
                    return

                if evenement.type == pygame.MOUSEBUTTONDOWN and not self.plateau.gagnant:
                    if ((self.plateau.tour_blanc and self.couleur_joueur == BLANC) or
                            (not self.plateau.tour_blanc and self.couleur_joueur == NOIR)):
                        self._gerer_clic_souris(evenement.pos)

            # Faire jouer l'IA si c'est son tour
            if not self.plateau.gagnant and (
                (self.plateau.tour_blanc and self.couleur_joueur == NOIR) or
                    (not self.plateau.tour_blanc and self.couleur_joueur == BLANC)):
                if self.ia_est_en_train_de_reflechir:
                    self.faire_jouer_ia()

            self.dessiner_plateau()
            self.dessiner_pieces()
            pygame.display.flip()
            clock.tick(60)

    def _gerer_clic_souris(self, pos):
        """Gère les clics de souris du joueur"""
        ligne, colonne = self.obtenir_case_cliquee(pos)
        if ligne >= 8:
            return

        piece_cliquee = self.plateau.obtenir_piece(ligne, colonne)

        # Gestion des prises obligatoires
        if self.derniere_piece_bougee and self.verifier_prises_possibles(self.derniere_piece_bougee):
            if piece_cliquee != self.derniere_piece_bougee:
                return

        if self.piece_selectionnee is None:
            # Sélectionner une pièce
            if piece_cliquee and piece_cliquee.couleur == self.couleur_joueur:
                if not self.derniere_piece_bougee or piece_cliquee == self.derniere_piece_bougee:
                    self._selectionner_piece(piece_cliquee)
        else:
            # Tenter un déplacement
            self._tenter_deplacement(ligne, colonne)

    def _selectionner_piece(self, piece):
        """Sélectionne une pièce et calcule ses déplacements possibles"""
        self.deplacements_possibles = self.plateau.obtenir_deplacements_possibles(
            piece)
        if self.deplacements_possibles:
            self.piece_selectionnee = piece
            piece.selectionne = True

    def _tenter_deplacement(self, ligne, colonne):
        """Tente de déplacer la pièce sélectionnée"""
        deplacement_valide = False
        pieces_avant = len(self.plateau.pieces)

        for sequence in self.deplacements_possibles:
            if (ligne, colonne) == sequence[-1]:
                deplacement_valide = self.plateau.executer_deplacement(
                    self.piece_selectionnee, sequence)
                break

        if deplacement_valide:
            pieces_apres = len(self.plateau.pieces)
            # Gérer les prises multiples
            if pieces_apres < pieces_avant:
                self.derniere_piece_bougee = self.piece_selectionnee
                prises_supplementaires = self.plateau.obtenir_prises_possibles(
                    self.piece_selectionnee)
                if prises_supplementaires:
                    self.deplacements_possibles = prises_supplementaires
                    self.prise_obligatoire = True
                    self.message_statut = "Prise obligatoire !"
                    return

            # Fin du tour
            self._finir_tour()
        else:
            # Désélectionner la pièce
            if self.piece_selectionnee:
                self.piece_selectionnee.selectionne = False
            self.piece_selectionnee = None
            self.deplacements_possibles = []

    def _finir_tour(self):
        """Termine le tour du joueur"""
        self.piece_selectionnee.selectionne = False
        self.piece_selectionnee = None
        self.deplacements_possibles = []
        self.derniere_piece_bougee = None
        self.prise_obligatoire = False
        self.message_statut = ""
        self.plateau.tour_blanc = not self.plateau.tour_blanc
        self.plateau.verifier_victoire()

        if not self.plateau.gagnant:
            self.simuler_reflexion_ia()


def lancer_jeu_contre_ia(chemin_ia, couleur_joueur=NOIR, difficulte='normale'):
    """Lance une partie contre l'IA entraînée"""
    jeu = JeuContreIA(chemin_ia, couleur_joueur, difficulte)
    jeu.executer()


if __name__ == "__main__":
    # Exemple d'utilisation avec différentes configurations

    # Pour jouer en tant que blancs (par défaut) en difficulté normale
    lancer_jeu_contre_ia("modele_p1500.pkl")

    # Pour jouer en tant que noirs en difficulté difficile
    # lancer_jeu_contre_ia("modele_final.pkl", NOIR, 'difficile')

    # Pour jouer en tant que blancs en difficulté facile
    # lancer_jeu_contre_ia("modele_final.pkl", BLANC, 'facile')
