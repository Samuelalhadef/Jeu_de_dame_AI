import numpy as np
import pickle
import random
import time
import os
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
from jeu_dames import JeuDames, Plateau, BLANC, NOIR


class EtatPlateau:
    """Représentation vectorisée de l'état du plateau"""

    def __init__(self, pieces):
        self.matrice = np.zeros((8, 8, 3), dtype=np.float32)

        for piece in pieces:
            if piece is not None:
                try:
                    if piece.couleur == BLANC:
                        self.matrice[piece.ligne, piece.colonne, 0] = 1
                    else:
                        self.matrice[piece.ligne, piece.colonne, 1] = 1
                    if piece.est_dame:
                        self.matrice[piece.ligne, piece.colonne, 2] = 1
                except AttributeError:
                    continue

        self.hash = hash(self.matrice.tobytes())

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        if not isinstance(other, EtatPlateau):
            return False
        return np.array_equal(self.matrice, other.matrice)

    def __str__(self):
        return str(self.hash)


class JoueurIA:
    """Agent hybride Q-learning et Minimax avec gestion d'erreurs"""

    def __init__(self, couleur, q_table_shared, lock, params=None):
        self.couleur = couleur
        self.q_table_shared = q_table_shared
        self.lock = lock
        self.PROFONDEUR_MINIMAX = 4

        default_params = {
            'epsilon': 0.8,
            'epsilon_decay': 0.9995,
            'epsilon_min': 0.2,
            'alpha': 0.4,
            'alpha_decay': 0.9999,
            'alpha_min': 0.1,
            'gamma': 0.95,
            'poids_minimax': 0.7,
            'poids_qlearning': 0.3
        }

        self.params = params if params is not None else default_params

    def obtenir_action(self, plateau, entrainement=True):
        try:
            pieces_valides = [p for p in plateau.pieces if p is not None]
            if not pieces_valides:
                return None

            etat_actuel = EtatPlateau(pieces_valides)
            actions_possibles = self._obtenir_actions_valides(plateau)

            if not actions_possibles:
                return None

            # Exploration
            if entrainement and random.random() < self.params['epsilon']:
                action = random.choice(actions_possibles)
                self.params['epsilon'] = max(
                    self.params['epsilon'] * self.params['epsilon_decay'],
                    self.params['epsilon_min']
                )
                return action

            # Exploitation avec minimax et Q-learning
            meilleure_action = None
            meilleur_score = float('-inf')
            alpha = float('-inf')
            beta = float('inf')

            for action in actions_possibles:
                if action[0] is None:
                    continue

                nouveau_plateau = self._simuler_action(plateau, action)
                if nouveau_plateau is None:
                    continue

                score_minimax = self._minimax(nouveau_plateau, self.PROFONDEUR_MINIMAX - 1,
                                              False, alpha, beta)

                with self.lock:
                    cle = (str(etat_actuel), str(
                        self._convertir_action_cle(action)))
                    score_q = self.q_table_shared.get(cle, 0.0)

                score = (self.params['poids_minimax'] * score_minimax +
                         self.params['poids_qlearning'] * score_q)

                if score > meilleur_score:
                    meilleur_score = score
                    meilleure_action = action

                alpha = max(alpha, score)
                if beta <= alpha:
                    break

            return meilleure_action or (actions_possibles[0] if actions_possibles else None)

        except Exception as e:
            print(f"Erreur dans obtenir_action: {e}")
            return None

    def _simuler_action(self, plateau, action):
        """Simule une action sur une copie du plateau"""
        try:
            nouveau_plateau = Plateau()
            piece, sequence = action

            # Copie des pièces
            nouveau_plateau.pieces = []
            for p in plateau.pieces:
                if p is None:
                    nouveau_plateau.pieces.append(None)
                    continue

                from jeu_dames import Piece
                nouvelle_piece = Piece(p.ligne, p.colonne, p.couleur)
                nouvelle_piece.est_dame = p.est_dame
                nouveau_plateau.pieces.append(nouvelle_piece)

            # Trouver la pièce correspondante
            piece_correspondante = None
            for p in nouveau_plateau.pieces:
                if (p is not None and p.ligne == piece.ligne and
                        p.colonne == piece.colonne and p.couleur == piece.couleur):
                    piece_correspondante = p
                    break

            if piece_correspondante is None:
                return None

            # Exécuter le mouvement
            nouveau_plateau.executer_deplacement(
                piece_correspondante, sequence)
            return nouveau_plateau

        except Exception as e:
            print(f"Erreur dans _simuler_action: {e}")
            return None

    def _minimax(self, plateau, profondeur, est_maximisant, alpha, beta):
        """Algorithme Minimax avec élagage alpha-beta et gestion d'erreurs"""
        try:
            if profondeur == 0 or plateau.gagnant:
                return self._evaluer_position(plateau)

            pieces_valides = [p for p in plateau.pieces if p is not None]
            if not pieces_valides:
                return 0

            couleur_active = self.couleur if est_maximisant else (
                BLANC if self.couleur == NOIR else NOIR)
            actions_possibles = self._obtenir_actions_valides(
                plateau, couleur_active)

            if est_maximisant:
                valeur = float('-inf')
                for action in actions_possibles:
                    nouveau_plateau = self._simuler_action(plateau, action)
                    if nouveau_plateau is not None:
                        valeur = max(valeur, self._minimax(
                            nouveau_plateau, profondeur - 1, False, alpha, beta))
                        alpha = max(alpha, valeur)
                        if beta <= alpha:
                            break
                return valeur
            else:
                valeur = float('inf')
                for action in actions_possibles:
                    nouveau_plateau = self._simuler_action(plateau, action)
                    if nouveau_plateau is not None:
                        valeur = min(valeur, self._minimax(
                            nouveau_plateau, profondeur - 1, True, alpha, beta))
                        beta = min(beta, valeur)
                        if beta <= alpha:
                            break
                return valeur

        except Exception as e:
            print(f"Erreur dans _minimax: {e}")
            return 0

    def _evaluer_position(self, plateau):
        """Évaluation heuristique de la position"""
        try:
            score = 0
            pieces_valides = [p for p in plateau.pieces if p is not None]

            for piece in pieces_valides:
                # Valeur de base des pièces
                valeur = 10 if piece.est_dame else 1
                multiplicateur = 1 if piece.couleur == self.couleur else -1
                score += valeur * multiplicateur

                if piece.couleur == self.couleur:
                    # Bonus position centrale
                    if 2 <= piece.ligne <= 5 and 2 <= piece.colonne <= 5:
                        score += 0.5

                    # Bonus progression
                    if self.couleur == BLANC:
                        score += (7 - piece.ligne) * 0.3
                    else:
                        score += piece.ligne * 0.3

                    # Bonus protection
                    if self._est_position_sure(plateau, piece.ligne, piece.colonne):
                        score += 0.8

            return score

        except Exception as e:
            print(f"Erreur dans _evaluer_position: {e}")
            return 0

    def _obtenir_actions_valides(self, plateau, couleur=None):
        """Obtient les actions valides pour une couleur donnée"""
        try:
            couleur = couleur or self.couleur
            pieces_valides = [
                p for p in plateau.pieces if p is not None and p.couleur == couleur]
            actions = []

            # Vérifier les prises
            for piece in pieces_valides:
                prises = plateau.obtenir_prises_possibles(piece)
                if prises:
                    return [(piece, sequence) for sequence in prises]

            # Déplacements simples
            for piece in pieces_valides:
                deplacements = plateau.obtenir_deplacements_simples(piece)
                actions.extend((piece, deplacement)
                               for deplacement in deplacements)

            return actions

        except Exception as e:
            print(f"Erreur dans _obtenir_actions_valides: {e}")
            return []

    def _est_position_sure(self, plateau, ligne, colonne):
        """Vérifie si une position est sûre"""
        try:
            pieces_adverses = [
                p for p in plateau.pieces if p is not None and p.couleur != self.couleur]
            for adverse in pieces_adverses:
                if abs(adverse.ligne - ligne) == 1 and abs(adverse.colonne - colonne) == 1:
                    ligne_saut = ligne + (adverse.ligne - ligne)
                    col_saut = colonne + (adverse.colonne - colonne)
                    if (0 <= ligne_saut <= 7 and 0 <= col_saut <= 7 and
                            not plateau.obtenir_piece(ligne_saut, col_saut)):
                        return False
            return True

        except Exception as e:
            print(f"Erreur dans _est_position_sure: {e}")
            return True

    def _convertir_action_cle(self, action):
        """Convertit une action en clé pour la Q-table"""
        try:
            piece, sequence = action
            if piece is None:
                return None
            return (piece.ligne, piece.colonne, tuple(map(tuple, sequence)))
        except Exception as e:
            print(f"Erreur dans _convertir_action_cle: {e}")
            return None


def entrainer_partie_process(seed, q_table_shared, recompenses, lock):
    """Processus d'entraînement d'une partie"""
    try:
        random.seed(seed)
        np.random.seed(seed)

        jeu = JeuDames()
        ia_blanc = JoueurIA(BLANC, q_table_shared, lock)
        ia_noir = JoueurIA(NOIR, q_table_shared, lock)

        stats = {
            'vainqueur': None,
            'nb_coups': 0,
            'prises_blanc': 0,
            'prises_noir': 0,
            'dames_blanc': 0,
            'dames_noir': 0
        }

        historique = []
        tour_blanc = True
        coups_sans_prise = 0

        while not jeu.plateau.gagnant and coups_sans_prise < 30:
            ia_active = ia_blanc if tour_blanc else ia_noir
            pieces_valides = [p for p in jeu.plateau.pieces if p is not None]
            etat_courant = EtatPlateau(pieces_valides)

            action = ia_active.obtenir_action(jeu.plateau)
            if not action:
                jeu.plateau.gagnant = "NOIR" if tour_blanc else "BLANC"
                break

            piece, sequence = action
            pieces_avant = len(
                [p for p in jeu.plateau.pieces if p is not None])
            etait_dame = piece.est_dame
            position_avant = piece.ligne

            jeu.plateau.executer_deplacement(piece, sequence)

            pieces_blanches = sum(
                1 for p in jeu.plateau.pieces if p is not None and p.couleur == BLANC)
            pieces_noires = sum(
                1 for p in jeu.plateau.pieces if p is not None and p.couleur == NOIR)

            if pieces_blanches == 0:
                jeu.plateau.gagnant = "NOIR"
                break
            elif pieces_noires == 0:
                jeu.plateau.gagnant = "BLANC"
                break

            # Calculer récompenses
            recompense = 0
            pieces_apres = len(
                [p for p in jeu.plateau.pieces if p is not None])
            pieces_prises = pieces_avant - pieces_apres

            if pieces_prises > 0:
                recompense += recompenses['prise'] * pieces_prises
                if tour_blanc:
                    stats['prises_blanc'] += pieces_prises
                else:
                    stats['prises_noir'] += pieces_prises

            if not etait_dame and piece.est_dame:
                recompense += recompenses['dame']
                if tour_blanc:
                    stats['dames_blanc'] += 1
                else:
                    stats['dames_noir'] += 1

            if (tour_blanc and position_avant > piece.ligne) or (not tour_blanc and position_avant < piece.ligne):
                recompense += recompenses['avance']

            dest_ligne, dest_col = sequence[-1]
            if (2 <= dest_ligne <= 5) and (2 <= dest_col <= 5):
                recompense += recompenses['position_centrale']

            difference_pieces = pieces_blanches - pieces_noires
            if tour_blanc:
                recompense += recompenses['difference_pieces'] * \
                    difference_pieces
            else:
                recompense += recompenses['difference_pieces'] * - \
                    difference_pieces

            # Sauvegarder transition
            pieces_valides_apres = [
                p for p in jeu.plateau.pieces if p is not None]
            historique.append((etat_courant, action, recompense,
                               EtatPlateau(pieces_valides_apres), tour_blanc))

            if pieces_prises == 0:
                coups_sans_prise += 1
            else:
                coups_sans_prise = 0

            tour_blanc = not tour_blanc
            stats['nb_coups'] += 1

        # Récompenses finales
        if jeu.plateau.gagnant == "BLANC":
            stats['vainqueur'] = 'BLANC'
            recompense_finale_blanc = recompenses['victoire']
            recompense_finale_noir = recompenses['defaite']
        elif jeu.plateau.gagnant == "NOIR":
            stats['vainqueur'] = 'NOIR'
            recompense_finale_blanc = recompenses['defaite']
            recompense_finale_noir = recompenses['victoire']
        else:
            stats['vainqueur'] = 'NUL'
            recompense_finale_blanc = recompenses['match_nul']
            recompense_finale_noir = recompenses['match_nul']

        # Mise à jour Q-table
        with lock:
            for etat, action, recompense, nouvel_etat, est_blanc in historique:
                if action is None:
                    continue

                ia = ia_blanc if est_blanc else ia_noir
                cle = (str(etat), str(ia._convertir_action_cle(action)))
                if cle[1] is None:
                    continue

                ancien_q = q_table_shared.get(cle, 0.0)
                recompense_finale = recompense_finale_blanc if est_blanc else recompense_finale_noir

                if nouvel_etat is None:
                    nouveau_q = ancien_q + \
                        ia.params['alpha'] * \
                        (recompense + recompense_finale - ancien_q)
                else:
                    meilleure_valeur_future = 0.0
                    actions_futures = ia._obtenir_actions_valides(jeu.plateau)
                    for action_future in actions_futures:
                        if action_future is None:
                            continue
                        cle_future = (str(nouvel_etat), str(
                            ia._convertir_action_cle(action_future)))
                        if cle_future[1] is not None:
                            valeur_future = q_table_shared.get(cle_future, 0.0)
                            meilleure_valeur_future = max(
                                meilleure_valeur_future, valeur_future)

                    nouveau_q = ancien_q + ia.params['alpha'] * (
                        recompense + ia.params['gamma'] *
                        meilleure_valeur_future - ancien_q
                    )

                q_table_shared[cle] = nouveau_q

                # Mettre à jour le taux d'apprentissage
                ia.params['alpha'] = max(
                    ia.params['alpha'] * ia.params['alpha_decay'],
                    ia.params['alpha_min']
                )

        return stats

    except Exception as e:
        print(f"Erreur dans entrainer_partie_process: {e}")
        return {
            'vainqueur': None,
            'nb_coups': 0,
            'prises_blanc': 0,
            'prises_noir': 0,
            'dames_blanc': 0,
            'dames_noir': 0
        }


class EntraineurIA:
    def __init__(self, nb_processus=None):
        self.nb_processus = nb_processus or mp.cpu_count()

        self.manager = Manager()
        self.q_table_shared = self.manager.dict()
        self.lock = self.manager.Lock()

        self.recompenses = {
            'victoire': 5.0,
            'defaite': -5.0,
            'match_nul': 0.0,
            'prise': 1.0,
            'dame': 2.0,
            'avance': 0.5,
            'position_centrale': 0.5,
            'menace': -0.8,
            'difference_pieces': 0.3
        }

        self._initialiser_dossiers()

    def _initialiser_dossiers(self):
        self.dossier_entrainement = Path('entrainement')
        self.dossier_entrainement.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.dossier_session = self.dossier_entrainement / \
            f'session_{timestamp}'
        self.dossier_session.mkdir(exist_ok=True)

        (self.dossier_session / 'modeles').mkdir()
        (self.dossier_session / 'stats').mkdir()
        (self.dossier_session / 'analyses').mkdir()

    def entrainer(self, nb_parties=1000, sauvegarde_interval=100):
        print(f"Démarrage de l'entraînement sur {
              self.nb_processus} processeurs...")

        stats_globales = {
            'victoires_blanc': 0,
            'victoires_noir': 0,
            'parties_nulles': 0,
            'prises_blanc': 0,
            'prises_noir': 0,
            'dames_blanc': 0,
            'dames_noir': 0,
            'taux_victoire_blanc': [],
            'taux_victoire_noir': [],
            'prises_moyennes_blanc': [],
            'prises_moyennes_noir': []
        }

        debut = time.time()

        try:
            with ProcessPoolExecutor(max_workers=self.nb_processus) as executor:
                futures = [
                    executor.submit(
                        entrainer_partie_process,
                        i,
                        self.q_table_shared,
                        self.recompenses,
                        self.lock
                    )
                    for i in range(nb_parties)
                ]

                for i, future in enumerate(futures):
                    try:
                        stats = future.result()

                        if stats['vainqueur'] == 'BLANC':
                            stats_globales['victoires_blanc'] += 1
                        elif stats['vainqueur'] == 'NOIR':
                            stats_globales['victoires_noir'] += 1
                        else:
                            stats_globales['parties_nulles'] += 1

                        stats_globales['prises_blanc'] += stats['prises_blanc']
                        stats_globales['prises_noir'] += stats['prises_noir']
                        stats_globales['dames_blanc'] += stats['dames_blanc']
                        stats_globales['dames_noir'] += stats['dames_noir']

                        if (i + 1) % sauvegarde_interval == 0:
                            self._calculer_statistiques(i + 1, stats_globales)
                            self._sauvegarder_progression(
                                i + 1, stats_globales)
                            self._verifier_performances(i + 1, stats_globales)

                    except Exception as e:
                        print(f"Erreur dans la partie {i}: {str(e)}")
                        continue

            duree = time.time() - debut
            print(f"\nEntraînement terminé en {duree:.1f} secondes")
            print(f"Vitesse moyenne: {nb_parties/duree:.1f} parties/seconde")

            self._sauvegarder_progression(nb_parties, stats_globales)
            return dict(self.q_table_shared)

        except Exception as e:
            print(f"Erreur pendant l'entraînement: {str(e)}")
            return dict(self.q_table_shared)

    def _calculer_statistiques(self, num_partie, stats):
        """Calcule les statistiques moyennes"""
        stats['taux_victoire_blanc'].append(
            stats['victoires_blanc'] / num_partie * 100
        )
        stats['taux_victoire_noir'].append(
            stats['victoires_noir'] / num_partie * 100
        )
        stats['prises_moyennes_blanc'].append(
            stats['prises_blanc'] / num_partie
        )
        stats['prises_moyennes_noir'].append(
            stats['prises_noir'] / num_partie
        )

    def _verifier_performances(self, num_partie, stats):
        if len(stats['taux_victoire_blanc']) >= 2:
            dernier_taux_blanc = stats['taux_victoire_blanc'][-1]
            dernier_taux_noir = stats['taux_victoire_noir'][-1]

            if dernier_taux_blanc < 5 and dernier_taux_noir < 5:
                print("\nAttention: Les IAs ne gagnent pas assez de parties!")
                print("Suggestions:")
                print("- Augmenter le taux d'exploration (epsilon)")
                print("- Augmenter les récompenses pour les actions agressives")
                print("- Réduire le nombre de coups maximum par partie")

            if stats['parties_nulles'] / num_partie > 0.8:
                print("\nAttention: Trop de parties nulles!")
                print("Suggestions:")
                print("- Réduire le seuil de coups sans prise")
                print("- Augmenter les récompenses pour les prises")

    def _sauvegarder_progression(self, num_partie, stats):
        # Sauvegarder le modèle
        fichier_modele = self.dossier_session / \
            'modeles' / f'modele_p{num_partie}.pkl'
        with open(fichier_modele, 'wb') as f:
            pickle.dump(dict(self.q_table_shared), f)

        # Afficher les statistiques
        print(f"\nStatistiques après {num_partie} parties:")
        print(f"Victoires Blanc: {stats['victoires_blanc']} ({
              stats['victoires_blanc']/num_partie*100:.1f}%)")
        print(f"Victoires Noir: {stats['victoires_noir']} ({
              stats['victoires_noir']/num_partie*100:.1f}%)")
        print(f"Parties nulles: {stats['parties_nulles']} ({
              stats['parties_nulles']/num_partie*100:.1f}%)")
        print("\nPerformances:")
        print(
            f"- Prises moyennes Blanc: {stats['prises_blanc']/num_partie:.2f}")
        print(f"- Prises moyennes Noir: {stats['prises_noir']/num_partie:.2f}")
        print(f"- Dames créées Blanc: {stats['dames_blanc']}")
        print(f"- Dames créées Noir: {stats['dames_noir']}")
        print(f"- Taille de la Q-table: {len(self.q_table_shared)}")

        # Sauvegarder les statistiques détaillées
        fichier_stats = self.dossier_session / \
            'stats' / f'stats_p{num_partie}.txt'
        with open(fichier_stats, 'w', encoding='utf-8') as f:
            f.write(f"=== Statistiques après {num_partie} parties ===\n")
            f.write(f"Victoires Blanc: {stats['victoires_blanc']}\n")
            f.write(f"Victoires Noir: {stats['victoires_noir']}\n")
            f.write(f"Parties nulles: {stats['parties_nulles']}\n")
            f.write(f"Prises Blanc: {stats['prises_blanc']}\n")
            f.write(f"Prises Noir: {stats['prises_noir']}\n")
            f.write(f"Dames Blanc: {stats['dames_blanc']}\n")
            f.write(f"Dames Noir: {stats['dames_noir']}\n")
            f.write(f"Taille Q-table: {len(self.q_table_shared)}\n")

            f.write("\nÉvolution des performances:\n")
            f.write("Taux de victoire Blanc (%): " +
                    ", ".join(f"{x:.1f}" for x in stats['taux_victoire_blanc']) + "\n")
            f.write("Taux de victoire Noir (%): " +
                    ", ".join(f"{x:.1f}" for x in stats['taux_victoire_noir']) + "\n")
            f.write("Prises moyennes Blanc: " +
                    ", ".join(f"{x:.2f}" for x in stats['prises_moyennes_blanc']) + "\n")
            f.write("Prises moyennes Noir: " +
                    ", ".join(f"{x:.2f}" for x in stats['prises_moyennes_noir']) + "\n")


def lancer_entrainement(nb_parties=1000):
    print("\n=== Configuration de l'entraînement ===")
    print(f"Nombre de parties: {nb_parties}")
    print(f"Processeurs utilisés: {mp.cpu_count()}")

    entraineur = EntraineurIA()

    try:
        print("\nDémarrage de l'entraînement...")
        q_table = entraineur.entrainer(
            nb_parties,
            sauvegarde_interval=max(1, nb_parties // 20)
        )

        # Sauvegarder le modèle final
        modele_final = entraineur.dossier_session / 'modeles' / 'modele_final.pkl'
        with open(modele_final, 'wb') as f:
            pickle.dump(q_table, f)

        print(f"\nEntraînement terminé!")
        print(f"Résultats sauvegardés dans: {entraineur.dossier_session}")

    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur")
        print(f"Résultats partiels sauvegardés dans: {
              entraineur.dossier_session}")

        modele_interrompu = entraineur.dossier_session / \
            'modeles' / 'modele_interrompu.pkl'
        with open(modele_interrompu, 'wb') as f:
            pickle.dump(dict(entraineur.q_table_shared), f)

    except Exception as e:
        print(f"\nErreur pendant l'entraînement: {str(e)}")
        raise


if __name__ == "__main__":
    lancer_entrainement(5000)
