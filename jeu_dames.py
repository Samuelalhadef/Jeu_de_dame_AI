import pygame
import sys

# Initialisation de Pygame
pygame.init()

# Constantes
TAILLE_CASE = 80
NB_CASES = 8
LARGEUR = TAILLE_CASE * NB_CASES
# Espace supplémentaire pour afficher le message de victoire
HAUTEUR = TAILLE_CASE * NB_CASES + 40

# Couleurs
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
MARRON = (139, 69, 19)
BEIGE = (245, 222, 179)
ROUGE = (255, 0, 0)
VERT = (0, 255, 0)


class Piece:
    def __init__(self, ligne, colonne, couleur):
        self.ligne = ligne
        self.colonne = colonne
        self.couleur = couleur
        self.est_dame = False
        self.selectionne = False

    def dessiner(self, fenetre):
        x = self.colonne * TAILLE_CASE + TAILLE_CASE // 2
        y = self.ligne * TAILLE_CASE + TAILLE_CASE // 2
        rayon = TAILLE_CASE // 2 - 10

        pygame.draw.circle(fenetre, self.couleur, (x, y), rayon)
        pygame.draw.circle(fenetre, NOIR, (x, y), rayon, 2)

        if self.est_dame:
            pygame.draw.circle(fenetre, ROUGE, (x, y), rayon // 2)

        if self.selectionne:
            pygame.draw.circle(fenetre, VERT, (x, y), rayon + 2, 3)


class Plateau:
    def __init__(self):
        self.pieces = []
        self.initialiser_pieces()
        self.piece_selectionnee = None
        self.tour_blanc = True
        self.gagnant = None

    def initialiser_pieces(self):
        # Placer les pions blancs
        for ligne in range(5, 8):
            for colonne in range(8):
                if (ligne + colonne) % 2 == 0:
                    self.pieces.append(Piece(ligne, colonne, BLANC))

        # Placer les pions noirs
        for ligne in range(3):
            for colonne in range(8):
                if (ligne + colonne) % 2 == 0:
                    self.pieces.append(Piece(ligne, colonne, NOIR))

    def obtenir_piece(self, ligne, colonne):
        for piece in self.pieces:
            if piece.ligne == ligne and piece.colonne == colonne:
                return piece
        return False

    def verifier_victoire(self):
        pieces_blanches = sum(
            1 for piece in self.pieces if piece.couleur == BLANC)
        pieces_noires = sum(
            1 for piece in self.pieces if piece.couleur == NOIR)

        if pieces_blanches == 0:
            self.gagnant = "NOIR"
        elif pieces_noires == 0:
            self.gagnant = "BLANC"

        # Vérifier si un joueur ne peut plus bouger
        couleur_actuelle = BLANC if self.tour_blanc else NOIR
        peut_bouger = False
        for piece in self.pieces:
            if piece.couleur == couleur_actuelle:
                if self.obtenir_deplacements_possibles(piece) or self.obtenir_prises_possibles(piece):
                    peut_bouger = True
                    break

        if not peut_bouger:
            self.gagnant = "NOIR" if self.tour_blanc else "BLANC"

    def obtenir_prises_possibles(self, piece):
        prises = []

        def explorer_prises(ligne_depart, colonne_depart, prises_actuelles, pieces_prises):
            if piece.est_dame:
                # Pour les dames: explorer toutes les directions à distance
                directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                for dir_l, dir_c in directions:
                    dist = 1
                    while True:
                        ligne_cible = ligne_depart + dist * dir_l
                        colonne_cible = colonne_depart + dist * dir_c

                        if not (0 <= ligne_cible < 8 and 0 <= colonne_cible < 8):
                            break

                        piece_cible = self.obtenir_piece(
                            ligne_cible, colonne_cible)
                        if piece_cible:
                            if piece_cible.couleur != piece.couleur and piece_cible not in pieces_prises:
                                # Chercher une case libre après la pièce
                                dist_apres = 1
                                while True:
                                    nouvelle_ligne = ligne_cible + dist_apres * dir_l
                                    nouvelle_colonne = colonne_cible + dist_apres * dir_c

                                    if not (0 <= nouvelle_ligne < 8 and 0 <= nouvelle_colonne < 8):
                                        break

                                    if not self.obtenir_piece(nouvelle_ligne, nouvelle_colonne):
                                        nouvelles_pieces_prises = pieces_prises + \
                                            [piece_cible]
                                        nouvelle_prise = prises_actuelles + \
                                            [(nouvelle_ligne, nouvelle_colonne)]
                                        prises.append(nouvelle_prise)

                                        # Explorer les prises suivantes
                                        explorer_prises(nouvelle_ligne, nouvelle_colonne,
                                                        nouvelle_prise, nouvelles_pieces_prises)
                                    else:
                                        break
                                    dist_apres += 1
                            break
                        dist += 1
            else:
                # Pour les pions: explorer les prises adjacentes
                directions = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
                for dir_l, dir_c in directions:
                    nouvelle_ligne = ligne_depart + dir_l
                    nouvelle_colonne = colonne_depart + dir_c

                    if not (0 <= nouvelle_ligne < 8 and 0 <= nouvelle_colonne < 8):
                        continue

                    piece_sautee = self.obtenir_piece(
                        ligne_depart + dir_l//2,
                        colonne_depart + dir_c//2
                    )

                    if (piece_sautee and
                        piece_sautee.couleur != piece.couleur and
                        piece_sautee not in pieces_prises and
                            not self.obtenir_piece(nouvelle_ligne, nouvelle_colonne)):

                        nouvelles_pieces_prises = pieces_prises + \
                            [piece_sautee]
                        nouvelle_prise = prises_actuelles + \
                            [(nouvelle_ligne, nouvelle_colonne)]
                        prises.append(nouvelle_prise)

                        # Explorer les prises suivantes
                        explorer_prises(nouvelle_ligne, nouvelle_colonne,
                                        nouvelle_prise, nouvelles_pieces_prises)

        # Commencer l'exploration des prises
        explorer_prises(piece.ligne, piece.colonne, [], [])
        return prises

    def obtenir_deplacements_simples(self, piece):
        if self.existe_prise_obligatoire(piece.couleur):
            return []

        deplacements = []
        if piece.est_dame:
            # La dame peut se déplacer en diagonale sur toute la longueur
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dir_l, dir_c in directions:
                dist = 1
                while True:
                    nouvelle_ligne = piece.ligne + dist * dir_l
                    nouvelle_colonne = piece.colonne + dist * dir_c

                    if not (0 <= nouvelle_ligne < 8 and 0 <= nouvelle_colonne < 8):
                        break

                    piece_destination = self.obtenir_piece(
                        nouvelle_ligne, nouvelle_colonne)
                    if piece_destination:
                        break

                    deplacements.append([(nouvelle_ligne, nouvelle_colonne)])
                    dist += 1
        else:
            # Déplacement normal d'un pion
            direction = -1 if piece.couleur == BLANC else 1
            for dc in [-1, 1]:
                nouvelle_ligne = piece.ligne + direction
                nouvelle_colonne = piece.colonne + dc

                if (0 <= nouvelle_ligne < 8 and
                    0 <= nouvelle_colonne < 8 and
                        not self.obtenir_piece(nouvelle_ligne, nouvelle_colonne)):
                    deplacements.append([(nouvelle_ligne, nouvelle_colonne)])

        return deplacements

    def obtenir_deplacements_possibles(self, piece):
        prises = self.obtenir_prises_possibles(piece)
        if prises:
            return prises
        return self.obtenir_deplacements_simples(piece)

    def existe_prise_obligatoire(self, couleur):
        for piece in self.pieces:
            if piece.couleur == couleur and self.obtenir_prises_possibles(piece):
                return True
        return False

    def executer_deplacement(self, piece, sequence_deplacement):
        # Parcourir la séquence de déplacements
        pieces_a_supprimer = []
        for i in range(len(sequence_deplacement)):
            nouvelle_ligne, nouvelle_colonne = sequence_deplacement[i]

            # Si c'est une prise (distance > 1)
            if abs(nouvelle_ligne - piece.ligne) > 1:
                # Pour une dame
                if piece.est_dame:
                    dir_l = 1 if nouvelle_ligne > piece.ligne else -1
                    dir_c = 1 if nouvelle_colonne > piece.colonne else -1
                    ligne_tmp = piece.ligne
                    colonne_tmp = piece.colonne
                    while ligne_tmp != nouvelle_ligne:
                        ligne_tmp += dir_l
                        colonne_tmp += dir_c
                        piece_sautee = self.obtenir_piece(
                            ligne_tmp, colonne_tmp)
                        if piece_sautee and piece_sautee.couleur != piece.couleur:
                            pieces_a_supprimer.append(piece_sautee)
                # Pour un pion
                else:
                    ligne_sautee = piece.ligne + \
                        (nouvelle_ligne - piece.ligne) // 2
                    colonne_sautee = piece.colonne + \
                        (nouvelle_colonne - piece.colonne) // 2
                    piece_sautee = self.obtenir_piece(
                        ligne_sautee, colonne_sautee)
                    if piece_sautee:
                        pieces_a_supprimer.append(piece_sautee)

            # Déplacer la pièce
            piece.ligne = nouvelle_ligne
            piece.colonne = nouvelle_colonne

            # Vérifier si la pièce devient une dame
            if ((piece.couleur == BLANC and nouvelle_ligne == 0) or
                    (piece.couleur == NOIR and nouvelle_ligne == 7)):
                piece.est_dame = True

        # Supprimer toutes les pièces capturées
        for piece_supprimee in pieces_a_supprimer:
            if piece_supprimee in self.pieces:
                self.pieces.remove(piece_supprimee)

        return True


class JeuDames:
    def __init__(self):
        self.fenetre = pygame.display.set_mode((LARGEUR, HAUTEUR))
        pygame.display.set_caption("Jeu de Dames")
        self.plateau = Plateau()
        self.piece_selectionnee = None
        self.deplacements_possibles = []
        self.police = pygame.font.Font(None, 36)
        self.derniere_piece_deplacee = None
        self.prise_obligatoire = False  # Nouveau flag pour éviter le blocage

    def dessiner_plateau(self):
        self.fenetre.fill(BEIGE)
        # Dessiner les cases
        for ligne in range(NB_CASES):
            for colonne in range(NB_CASES):
                if (ligne + colonne) % 2 == 0:
                    pygame.draw.rect(self.fenetre, MARRON,
                                     (colonne * TAILLE_CASE,
                                      ligne * TAILLE_CASE,
                                      TAILLE_CASE, TAILLE_CASE))

        # Afficher les déplacements possibles
        for sequence in self.deplacements_possibles:
            for ligne, colonne in sequence:
                pygame.draw.circle(self.fenetre, VERT,
                                   (colonne * TAILLE_CASE + TAILLE_CASE // 2,
                                    ligne * TAILLE_CASE + TAILLE_CASE // 2),
                                   10)

        # Afficher le tour du joueur
        joueur = "BLANC" if self.plateau.tour_blanc else "NOIR"
        texte = self.police.render(f"Tour : {joueur}", True, NOIR)
        self.fenetre.blit(texte, (10, HAUTEUR - 30))

        # Afficher le gagnant si la partie est terminée
        if self.plateau.gagnant:
            texte = self.police.render(
                f"VICTOIRE DES {self.plateau.gagnant}S !", True, ROUGE)
            self.fenetre.blit(
                texte, (LARGEUR // 2 - texte.get_width() // 2, HAUTEUR - 30))

    def dessiner_pieces(self):
        for piece in self.plateau.pieces:
            piece.dessiner(self.fenetre)

    def obtenir_case_cliquee(self, pos_souris):
        x, y = pos_souris
        return y // TAILLE_CASE, x // TAILLE_CASE

    def verifier_prises_apres_deplacement(self, piece):
        """Vérifie si la pièce qui vient d'être déplacée peut effectuer une prise"""
        prises_possibles = self.plateau.obtenir_prises_possibles(piece)
        if prises_possibles:
            self.deplacements_possibles = prises_possibles
            self.piece_selectionnee = piece
            piece.selectionne = True
            self.prise_obligatoire = True  # Marquer qu'une prise est obligatoire
            return True
        self.prise_obligatoire = False  # Réinitialiser le flag si pas de prise possible
        return False

    def executer(self):
        while True:
            for evenement in pygame.event.get():
                if evenement.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if evenement.type == pygame.MOUSEBUTTONDOWN and not self.plateau.gagnant:
                    ligne, colonne = self.obtenir_case_cliquee(evenement.pos)
                    if ligne >= 8:  # Clic en dehors du plateau
                        continue

                    piece_cliquee = self.plateau.obtenir_piece(ligne, colonne)

                    # Si une prise est obligatoire, vérifier qu'on utilise la bonne pièce
                    if self.prise_obligatoire and piece_cliquee != self.derniere_piece_deplacee:
                        if self.plateau.obtenir_prises_possibles(self.derniere_piece_deplacee):
                            continue  # On doit continuer avec la pièce qui peut prendre

                    if self.piece_selectionnee is None:
                        # Sélectionner une pièce
                        if (piece_cliquee and
                            ((piece_cliquee.couleur == BLANC and self.plateau.tour_blanc) or
                             (piece_cliquee.couleur == NOIR and not self.plateau.tour_blanc))):

                            # Vérifier si on peut jouer avec cette pièce
                            if not self.prise_obligatoire or piece_cliquee == self.derniere_piece_deplacee:
                                self.deplacements_possibles = self.plateau.obtenir_deplacements_possibles(
                                    piece_cliquee)
                                if self.deplacements_possibles:
                                    self.piece_selectionnee = piece_cliquee
                                    piece_cliquee.selectionne = True
                    else:
                        # Tenter un déplacement
                        deplacement_valide = False
                        pieces_avant = len(self.plateau.pieces)

                        for sequence in self.deplacements_possibles:
                            if (ligne, colonne) == sequence[-1]:
                                deplacement_valide = self.plateau.executer_deplacement(
                                    self.piece_selectionnee, sequence)
                                break

                        # Si le déplacement est réussi
                        if deplacement_valide:
                            self.derniere_piece_deplacee = self.piece_selectionnee
                            pieces_apres = len(self.plateau.pieces)

                            # Si c'était une prise
                            if pieces_avant > pieces_apres:
                                # Vérifier s'il y a d'autres prises possibles avec cette pièce
                                if not self.verifier_prises_apres_deplacement(self.derniere_piece_deplacee):
                                    # Fin du tour si pas d'autres prises possibles
                                    self.piece_selectionnee.selectionne = False
                                    self.piece_selectionnee = None
                                    self.deplacements_possibles = []
                                    self.derniere_piece_deplacee = None
                                    self.prise_obligatoire = False
                                    self.plateau.tour_blanc = not self.plateau.tour_blanc
                                    self.plateau.verifier_victoire()
                            else:
                                # C'était un déplacement simple
                                self.piece_selectionnee.selectionne = False
                                self.piece_selectionnee = None
                                self.deplacements_possibles = []
                                self.derniere_piece_deplacee = None
                                self.prise_obligatoire = False
                                self.plateau.tour_blanc = not self.plateau.tour_blanc
                                self.plateau.verifier_victoire()
                        else:
                            # Si le déplacement n'est pas valide, désélectionner la pièce
                            if self.piece_selectionnee:
                                self.piece_selectionnee.selectionne = False
                            self.piece_selectionnee = None
                            self.deplacements_possibles = []

            # Dessiner le jeu
            self.dessiner_plateau()
            self.dessiner_pieces()
            pygame.display.flip()


if __name__ == "__main__":
    jeu = JeuDames()
    jeu.executer()
