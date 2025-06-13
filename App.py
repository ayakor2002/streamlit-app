import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import sys
import subprocess
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import pulp as plp
from itertools import product
from typing import Dict, List, Tuple, Any
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Système Intégré Prédiction-Planification Avancé",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2e86ab);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e86ab;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MultiPosteDefectPredictor:
    def __init__(self):
        self.models = {}
        self.transformers = {}
        self.best_model_names = {}
        self.feature_importances = {}
        self.original_data = None
        self.postes = []
        self.jour_col = None
        self.volume_col = None
        self.predictions_history = []
        self.poste_weights = {}

    def calculate_poste_weights_from_data(self, data=None):
        if data is None:
            data = self.original_data

        if data is None:
            self.poste_weights = {poste: 1.0/len(self.postes) for poste in self.postes}
            return

        poste_qi_di = {}
        total_qi_di = 0

        for poste in self.postes:
            if poste in data.columns and self.volume_col in data.columns:
                qi = data[self.volume_col].sum()
                di = data[poste].sum()
                qi_di = qi * di
                poste_qi_di[poste] = qi_di
                total_qi_di += qi_di

        if total_qi_di > 0:
            self.poste_weights = {poste: qi_di / total_qi_di
                                 for poste, qi_di in poste_qi_di.items()}
        else:
            self.poste_weights = {poste: 1.0/len(self.postes) for poste in self.postes}

    def set_poste_weights(self, weights_dict=None):
        if weights_dict is None:
            self.calculate_poste_weights_from_data()
        else:
            self.poste_weights = weights_dict.copy()
            total_weight = sum(self.poste_weights.values())
            if total_weight > 0:
                self.poste_weights = {poste: weight/total_weight
                                     for poste, weight in self.poste_weights.items()}

    def calculate_weighted_average(self, predictions_postes):
        if not self.poste_weights:
            self.set_poste_weights()

        weighted_sum = 0
        total_weight = 0

        for poste, prediction in predictions_postes.items():
            if poste in self.poste_weights:
                weight = self.poste_weights[poste]
                weighted_sum += prediction * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0

    def identify_postes(self, data):
        postes_cols = [col for col in data.columns if "_defauts" in col.lower() or "defaut" in col.lower()]

        if not postes_cols:
            excluded_keywords = ['jour', 'volume', 'production', 'date', 'time']
            postes_cols = []

            for col in data.columns:
                is_excluded = any(keyword in col.lower() for keyword in excluded_keywords)
                if not is_excluded:
                    postes_cols.append(col)

        self.postes = postes_cols
        return postes_cols

    def prepare_data_for_poste(self, data, poste_col):
        data_copy = data.copy()

        jour_col = next((col for col in data.columns if 'jour' in col.lower()), None)
        volume_col = next((col for col in data.columns if 'volume' in col.lower()), None)

        if not jour_col or not volume_col:
            raise ValueError("Les colonnes 'jour' et 'volume de production' sont requises")

        self.jour_col = jour_col
        self.volume_col = volume_col

        try:
            data_copy['jour_numerique'] = pd.to_datetime(data_copy[jour_col])
            data_copy['jour_numerique'] = data_copy['jour_numerique'].dt.dayofweek + 1
            jour_col_final = 'jour_numerique'
        except:
            jour_col_final = jour_col

        X = data_copy[[volume_col, jour_col_final]]
        y = data_copy[poste_col]

        return X, y, jour_col_final, volume_col

    def train_model_for_poste(self, X, y, poste_name, jour_col, volume_col, search_method='grid', n_iter=20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        numerical_features = [volume_col, jour_col]
        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), numerical_features)]
        )

        transformer = preprocessor.fit(X_train)
        self.transformers[poste_name] = transformer

        models = {
            'DecisionTree': Pipeline([
                ('preprocessor', preprocessor),
                ('model', DecisionTreeRegressor(random_state=42))
            ]),
            'RandomForest': Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(random_state=42))
            ]),
            'GradientBoosting': Pipeline([
                ('preprocessor', preprocessor),
                ('model', GradientBoostingRegressor(random_state=42))
            ]),
            'NeuralNetwork': Pipeline([
                ('preprocessor', preprocessor),
                ('model', MLPRegressor(max_iter=1000, random_state=42))
            ])
        }

        param_grids = {
            'DecisionTree': {
                'model__max_depth': [None, 5, 10, 15],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            },
            'RandomForest': {
                'model__n_estimators': [50, 100, 150],
                'model__max_depth': [None, 10, 15],
                'model__min_samples_split': [2, 5]
            },
            'GradientBoosting': {
                'model__n_estimators': [50, 100, 150],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [3, 5, 7]
            },
            'NeuralNetwork': {
                'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'model__alpha': [0.0001, 0.001, 0.01]
            }
        }

        best_score = float('inf')
        best_model_name = None
        best_model = None
        best_params = None

        for name, model in models.items():
            if search_method == 'grid':
                search = GridSearchCV(
                    model,
                    param_grids[name],
                    cv=TimeSeriesSplit(n_splits=5),
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
            else:
                search = RandomizedSearchCV(
                    model,
                    param_grids[name],
                    n_iter=n_iter,
                    cv=TimeSeriesSplit(n_splits=5),
                    scoring='neg_mean_squared_error',
                    random_state=42,
                    n_jobs=-1
                )

            search.fit(X_train, y_train)
            mse = mean_squared_error(y_test, search.predict(X_test))

            if mse < best_score:
                best_score = mse
                best_model_name = name
                best_model = search.best_estimator_
                best_params = search.best_params_

        self.models[poste_name] = best_model
        self.best_model_names[poste_name] = best_model_name

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if best_model_name in ['DecisionTree', 'RandomForest', 'GradientBoosting']:
            model_step = best_model.named_steps['model']
            self.feature_importances[poste_name] = pd.DataFrame({
                'feature': numerical_features,
                'importance': model_step.feature_importances_
            }).sort_values('importance', ascending=False)

        return {
            'model_name': best_model_name,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred,
            'X_test': X_test,
            'best_params': best_params
        }

    def train_all_postes(self, data, search_method='grid'):
        postes = self.identify_postes(data)

        if not postes:
            raise ValueError("Aucun poste identifié dans les données!")

        results = {}
        for poste in postes:
            X, y, jour_col, volume_col = self.prepare_data_for_poste(data, poste)
            results[poste] = self.train_model_for_poste(X, y, poste, jour_col, volume_col, search_method=search_method)

        self.set_poste_weights()
        return results, postes

    def predict_single_scenario(self, jour, volume):
        if not self.models:
            raise ValueError("Les modèles doivent être entraînés avant de faire des prédictions!")

        if 'jour_numerique' in self.transformers[list(self.transformers.keys())[0]].transformers_[0][2]:
            new_data = pd.DataFrame({
                self.volume_col: [volume],
                'jour_numerique': [jour]
            })
            X_new = new_data[[self.volume_col, 'jour_numerique']]
        else:
            new_data = pd.DataFrame({
                self.volume_col: [volume],
                self.jour_col: [jour]
            })
            X_new = new_data[[self.volume_col, self.jour_col]]

        predictions_postes = {}
        for poste, model in self.models.items():
            predictions_postes[poste] = model.predict(X_new)[0]

        predictions_chaine = {
            'max': max(predictions_postes.values()),
            'moyenne': np.mean(list(predictions_postes.values())),
            'moyenne_ponderee': self.calculate_weighted_average(predictions_postes),
            'somme': sum(predictions_postes.values())
        }

        taux_rework_postes = {}
        for poste, defauts in predictions_postes.items():
            taux_rework_postes[poste] = (defauts / volume) * 100

        taux_rework_chaine = {}
        for method, defauts in predictions_chaine.items():
            taux_rework_chaine[method] = (defauts / volume) * 100

        prediction_record = {
            'jour': jour,
            'volume': volume,
            'predictions_postes': predictions_postes,
            'predictions_chaine': predictions_chaine,
            'taux_rework_postes': taux_rework_postes,
            'taux_rework_chaine': taux_rework_chaine
        }
        self.predictions_history.append(prediction_record)

        return prediction_record

class StochasticPlanningModelComplete:
    """
    Modèle de planification stochastique avancé avec analyse multicritères
    Adapté pour l'intégration avec la prédiction de défauts
    """

    def __init__(self):
        """Initialisation du modèle"""
        self.model = None
        self.variables = {}
        self.results = {}
        self.parameters = {}
        self.scenario_analysis = {}
        self.multicriteria_scores = {}
        self.best_scenario_selection = {}
        self.predicted_rework_rate = None

    def set_parameters(self,
                      S: int = 5,
                      T: int = 3,
                      R: List[str] = None,
                      EDI: List = None,
                      p: List[List] = None,
                      D: List[List] = None,
                      seuil: float = 0.95,
                      mean_capacity: float = 160,
                      std_capacity: float = 10,
                      mean_defaut: float = 0.04,
                      std_defaut: float = 0.01,
                      m: int = 5,
                      alpha_rework: float = 0.8,
                      beta: float = 1.2,
                      b: int = 10,
                      penalite_penurie: float = 1000,
                      # Paramètres analyse multicritères
                      poids_cout: float = 0.25,
                      poids_satisfaction: float = 0.30,
                      poids_utilisation: float = 0.20,
                      poids_stabilite: float = 0.15,
                      poids_penuries: float = 0.10,
                      # Nouveaux paramètres pour l'intégration
                      use_predicted_rework: bool = False,
                      predicted_rework_rate: float = None):
        """
        Configuration complète du modèle avec intégration des prédictions
        """

        # 10 références par défaut
        if R is None:
            R = [f'REF_{i+1:02d}' for i in range(10)]

        # Demande client par défaut pour 10 références
        if EDI is None:
            EDI = [20, 35, 45, 25, 40, 50, 22, 38, 30, 42]

        # Si des demandes personnalisées sont fournies sous forme de liste
        if isinstance(EDI, list):
            EDI_dict = {R[i]: EDI[i] for i in range(min(len(R), len(EDI)))}
        else:
            EDI_dict = EDI

        # Matrice P (différence de MH) 10x10
        if p is None:
            p = [
                [0, 0.20, 0.30, 0.15, 0.25, 0.35, 0.12, 0.22, 0.32, 0.18],
                [0.20, 0, 0.40, 0.25, 0.35, 0.45, 0.22, 0.32, 0.42, 0.28],
                [0.30, 0.40, 0, 0.35, 0.45, 0.55, 0.32, 0.42, 0.52, 0.38],
                [0.15, 0.25, 0.35, 0, 0.30, 0.40, 0.17, 0.27, 0.37, 0.23],
                [0.25, 0.35, 0.45, 0.30, 0, 0.50, 0.27, 0.37, 0.47, 0.33],
                [0.35, 0.45, 0.55, 0.40, 0.50, 0, 0.37, 0.47, 0.57, 0.43],
                [0.12, 0.22, 0.32, 0.17, 0.27, 0.37, 0, 0.24, 0.34, 0.20],
                [0.22, 0.32, 0.42, 0.27, 0.37, 0.47, 0.24, 0, 0.44, 0.30],
                [0.32, 0.42, 0.52, 0.37, 0.47, 0.57, 0.34, 0.44, 0, 0.40],
                [0.18, 0.28, 0.38, 0.23, 0.33, 0.43, 0.20, 0.30, 0.40, 0]
            ]

        # Matrice D (similarité) 10x10
        if D is None:
            D = [
                [1.0, 0.2, 0.4, 0.3, 0.1, 0.5, 0.2, 0.3, 0.4, 0.1],
                [0.2, 1.0, 0.6, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.6],
                [0.4, 0.6, 1.0, 0.5, 0.4, 0.3, 0.6, 0.5, 0.2, 0.7],
                [0.3, 0.4, 0.5, 1.0, 0.6, 0.4, 0.3, 0.7, 0.5, 0.4],
                [0.1, 0.3, 0.4, 0.6, 1.0, 0.5, 0.4, 0.6, 0.7, 0.3],
                [0.5, 0.2, 0.3, 0.4, 0.5, 1.0, 0.3, 0.2, 0.4, 0.6],
                [0.2, 0.5, 0.6, 0.3, 0.4, 0.3, 1.0, 0.5, 0.3, 0.7],
                [0.3, 0.4, 0.5, 0.7, 0.6, 0.2, 0.5, 1.0, 0.4, 0.5],
                [0.4, 0.3, 0.2, 0.5, 0.7, 0.4, 0.3, 0.4, 1.0, 0.6],
                [0.1, 0.6, 0.7, 0.4, 0.3, 0.6, 0.7, 0.5, 0.6, 1.0]
            ]

        # Conversion en dictionnaire pour PuLP
        p_dict = {}
        for i in range(len(R)):
            for j in range(len(R)):
                p_dict[(R[i], j)] = p[i][j]

        D_array = np.array(D)

        # Génération stochastique des paramètres
        np.random.seed(42)
        
        # Taux de défaut - MODIFICATION POUR INTÉGRATION
        taux_defaut = {}
        
        if use_predicted_rework and predicted_rework_rate is not None:
            # Utiliser le taux prédit pour TOUS les scénarios (constant)
            base_rate = predicted_rework_rate / 100  # Conversion de pourcentage
            self.predicted_rework_rate = predicted_rework_rate
            
            for s in range(S):
                for i in R:
                    # Même taux pour tous les scénarios (vient de la prédiction)
                    taux_defaut[(s, i)] = base_rate
        else:
            # Génération stochastique classique
            for s in range(S):
                for i in R:
                    defaut = max(0.001, min(0.25, np.random.normal(mean_defaut, std_defaut)))
                    taux_defaut[(s, i)] = defaut

        # Capacités stochastiques selon spécifications
        CAPchaine = {}
        for s in range(S):
            for t in range(T):
                capacite = max(50, np.random.normal(mean_capacity, std_capacity))
                CAPchaine[(s, t)] = capacite

        self.parameters = {
            'S': S, 'T': T, 'R': R, 'EDI': EDI_dict, 'p': p_dict, 'D': D_array,
            'seuil': seuil, 'CAPchaine': CAPchaine, 'm': m, 'taux_defaut': taux_defaut,
            'alpha_rework': alpha_rework, 'beta': beta, 'b': b, 'penalite_penurie': penalite_penurie,
            'mean_capacity': mean_capacity, 'std_capacity': std_capacity,
            'mean_defaut': mean_defaut, 'std_defaut': std_defaut,
            'poids_cout': poids_cout, 'poids_satisfaction': poids_satisfaction,
            'poids_utilisation': poids_utilisation, 'poids_stabilite': poids_stabilite,
            'poids_penuries': poids_penuries,
            'use_predicted_rework': use_predicted_rework,
            'predicted_rework_rate': predicted_rework_rate
        }

    def create_model(self):
        """Création du modèle d'optimisation"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']

        self.model = plp.LpProblem("Planification_Stochastique_Complete", plp.LpMinimize)

        # Variables de décision
        # Variables binaires de séquencement
        self.variables['x'] = plp.LpVariable.dicts(
            "x",
            [(i, j, s, t) for i in R for j in range(len(R)) for s in range(S) for t in range(T)],
            cat='Binary'
        )

        # Variables continues de production
        self.variables['q'] = plp.LpVariable.dicts(
            "q",
            [(s, i, t) for s in range(S) for i in R for t in range(T)],
            lowBound=0,
            cat='Continuous'
        )

        # Variables de pénurie
        self.variables['penurie'] = plp.LpVariable.dicts(
            "penurie",
            [(s, i) for s in range(S) for i in R],
            lowBound=0,
            cat='Continuous'
        )

    def add_constraints(self):
        """Ajout des contraintes du modèle"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        x, q, penurie = self.variables['x'], self.variables['q'], self.variables['penurie']

        # 1. Satisfaction de la demande avec rework
        for s in range(S):
            for i in R:
                demande_satisfaite = plp.lpSum([
                    q[(s, i, t)] * (1 - params['taux_defaut'][(s, i)]) +
                    params['alpha_rework'] * q[(s, i, t)] * params['taux_defaut'][(s, i)]
                    for t in range(T)
                ])
                self.model += (
                    demande_satisfaite + penurie[(s, i)] >= params['EDI'][i],
                    f"Demande_s{s}_i{i}"
                )

        # 2. Contraintes de capacité
        for s in range(S):
            for t in range(T):
                capacite_utilisee = plp.lpSum([
                    q[(s, i, t)] * (1 + params['beta'] * params['taux_defaut'][(s, i)])
                    for i in R
                ])
                self.model += (
                    capacite_utilisee <= params['CAPchaine'][(s, t)],
                    f"Capacite_s{s}_t{t}"
                )

        # 3. Contraintes de séquencement
        for s in range(S):
            for t in range(T):
                # Une référence par position
                for j in range(len(R)):
                    self.model += (
                        plp.lpSum([x[(i, j, s, t)] for i in R]) == 1,
                        f"Position_s{s}_t{t}_j{j}"
                    )
                
                # Une position par référence (max)
                for i in R:
                    self.model += (
                        plp.lpSum([x[(i, j, s, t)] for j in range(len(R))]) <= 1,
                        f"Reference_s{s}_t{t}_i{i}"
                    )

        # 4. Production minimale
        for s in range(S):
            for i in R:
                for t in range(T):
                    taux_defaut_si = params['taux_defaut'][(s, i)]
                    if taux_defaut_si < 0.99:
                        production_requise = params['m'] / (1 - taux_defaut_si + params['alpha_rework'] * taux_defaut_si)
                    else:
                        production_requise = params['m'] * 2
                    
                    self.model += (
                        q[(s, i, t)] >= production_requise * plp.lpSum([x[(i, j, s, t)] for j in range(len(R))]),
                        f"Production_min_s{s}_i{i}_t{t}"
                    )

    def set_objective(self):
        """Définition de la fonction objectif"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        q, penurie = self.variables['q'], self.variables['penurie']

        # Coût de production
        cout_production = plp.lpSum([
            20 * q[(s, i, t)]
            for s in range(S) for i in R for t in range(T)
        ])

        # Coût des pénuries
        cout_penuries = plp.lpSum([
            params['penalite_penurie'] * penurie[(s, i)]
            for s in range(S) for i in R
        ])

        # Objectif : minimiser les coûts totaux
        self.model += cout_production + cout_penuries

    def solve_model(self, solver_name='PULP_CBC_CMD', time_limit=300):
        """Résolution du modèle"""
        try:
            if solver_name == 'PULP_CBC_CMD':
                solver = plp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
            else:
                solver = plp.getSolver(solver_name)

            self.model.solve(solver)

            if self.model.status == plp.LpStatusOptimal:
                self._extract_results()
                return True
            else:
                return False

        except Exception as e:
            st.error(f"Erreur lors de la résolution: {e}")
            return False

    def _extract_results(self):
        """Extraction des résultats de la solution"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']

        # Production par scénario
        production_results = {}
        for s in range(S):
            for i in R:
                for t in range(T):
                    key = (s, i, t)
                    value = self.variables['q'][key].value()
                    production_results[key] = value if value is not None else 0

        # Pénuries par scénario
        penuries_results = {}
        for s in range(S):
            for i in R:
                key = (s, i)
                value = self.variables['penurie'][key].value()
                penuries_results[key] = value if value is not None else 0

        # Séquencement par scénario
        sequencement_results = {}
        for s in range(S):
            for t in range(T):
                sequence = {}
                for i in R:
                    for j in range(len(R)):
                        key = (i, j, s, t)
                        value = self.variables['x'][key].value()
                        if value is not None and value > 0.5:
                            sequence[j] = i
                sequencement_results[(s, t)] = sequence

        self.results = {
            'production': production_results,
            'penuries': penuries_results,
            'sequencement': sequencement_results,
            'cout_total': self.model.objective.value()
        }

    def analyze_scenarios_detailed(self):
        """Analyse détaillée des scénarios pour Streamlit"""
        if not self.results:
            return

        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        production = self.results['production']
        penuries = self.results['penuries']
        sequencement = self.results['sequencement']

        self.scenario_analysis = {}

        for s in range(S):
            scenario_data = {
                'scenario_id': s + 1,
                'shifts_details': {},
                'production_summary': {},
                'kpis': {}
            }

            # Analyse par shift
            total_capacity_used = 0
            total_capacity_available = 0

            for t in range(T):
                shift_info = {
                    'execution_order': [],
                    'quantities': {},
                    'capacity_used': 0,
                    'capacity_available': params['CAPchaine'][(s, t)]
                }

                # Ordre d'exécution
                sequence = sequencement.get((s, t), {})
                ordered_refs = []
                for j in range(len(R)):
                    ref = sequence.get(j, 'VIDE')
                    ordered_refs.append(ref)
                shift_info['execution_order'] = ordered_refs

                # Quantités et capacité utilisée
                capacity_used = 0
                for i in R:
                    qty = production[(s, i, t)]
                    if qty > 0:
                        shift_info['quantities'][i] = qty
                        taux_def = params['taux_defaut'][(s, i)]
                        capacity_used += qty * (1 + params['beta'] * taux_def)

                shift_info['capacity_used'] = capacity_used
                if shift_info['capacity_available'] > 0:
                    shift_info['capacity_utilization'] = (capacity_used / shift_info['capacity_available']) * 100
                else:
                    shift_info['capacity_utilization'] = 0

                total_capacity_used += capacity_used
                total_capacity_available += shift_info['capacity_available']

                scenario_data['shifts_details'][t+1] = shift_info

            # Résumé production par référence
            total_production_utile = 0
            total_demande = sum(params['EDI'].values())
            total_penuries = 0

            for i in R:
                total_prod = sum(production[(s, i, t)] for t in range(T))
                
                # Production utile avec rework
                total_utile = 0
                for t in range(T):
                    qty = production[(s, i, t)]
                    taux_def = params['taux_defaut'][(s, i)]
                    pieces_bonnes = qty * (1 - taux_def)
                    pieces_rework_ok = qty * taux_def * params['alpha_rework']
                    total_utile += pieces_bonnes + pieces_rework_ok

                penurie = penuries[(s, i)]
                demande = params['EDI'][i]
                taux_couverture = (total_utile / demande) * 100 if demande > 0 else 0

                scenario_data['production_summary'][i] = {
                    'demande': demande,
                    'production_brute': total_prod,
                    'production_utile': total_utile,
                    'penurie': penurie,
                    'taux_couverture': taux_couverture
                }

                total_production_utile += total_utile
                total_penuries += penurie

            # Calcul des KPIs du scénario
            cout_production = sum([production[(s, i, t)] * 20 for i in R for t in range(T)])
            cout_penuries = total_penuries * params['penalite_penurie']
            cout_total = cout_production + cout_penuries

            # Stabilité (variance des productions par shift)
            productions_par_shift = []
            for t in range(T):
                prod_shift = sum(production[(s, i, t)] for i in R)
                productions_par_shift.append(prod_shift)
            
            if len(productions_par_shift) > 1:
                variance_production = np.var(productions_par_shift)
                stabilite = max(0, 100 - variance_production / 10)
            else:
                stabilite = 100

            scenario_kpis = {
                'satisfaction_globale': (total_production_utile / total_demande) * 100 if total_demande > 0 else 0,
                'utilisation_capacite': (total_capacity_used / total_capacity_available) * 100 if total_capacity_available > 0 else 0,
                'total_penuries': total_penuries,
                'cout_total': cout_total,
                'cout_production': cout_production,
                'cout_penuries': cout_penuries,
                'stabilite': stabilite,
                'efficacite_production': total_production_utile / max(1, total_capacity_used)
            }

            scenario_data['kpis'] = scenario_kpis
            self.scenario_analysis[s] = scenario_data

    def calculate_multicriteria_scores(self):
        """Calcul des scores multicritères"""
        if not self.scenario_analysis:
            return

        params = self.parameters
        S = params['S']

        # Extraction des valeurs des critères
        criteria_values = {
            'cout': [self.scenario_analysis[s]['kpis']['cout_total'] for s in range(S)],
            'satisfaction': [self.scenario_analysis[s]['kpis']['satisfaction_globale'] for s in range(S)],
            'utilisation': [self.scenario_analysis[s]['kpis']['utilisation_capacite'] for s in range(S)],
            'stabilite': [self.scenario_analysis[s]['kpis']['stabilite'] for s in range(S)],
            'penuries': [self.scenario_analysis[s]['kpis']['total_penuries'] for s in range(S)]
        }

        # Normalisation des critères
        def normalize_criterion(values, inverse=False):
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return [1.0] * len(values)
            
            if inverse:
                return [(max_val - v) / (max_val - min_val) for v in values]
            else:
                return [(v - min_val) / (max_val - min_val) for v in values]

        normalized_criteria = {
            'cout': normalize_criterion(criteria_values['cout'], inverse=True),
            'satisfaction': normalize_criterion(criteria_values['satisfaction'], inverse=False),
            'utilisation': normalize_criterion(criteria_values['utilisation'], inverse=False),
            'stabilite': normalize_criterion(criteria_values['stabilite'], inverse=False),
            'penuries': normalize_criterion(criteria_values['penuries'], inverse=True)
        }

        # Calcul des scores globaux pondérés
        global_scores = {}
        for s in range(S):
            score = (
                params['poids_cout'] * normalized_criteria['cout'][s] +
                params['poids_satisfaction'] * normalized_criteria['satisfaction'][s] +
                params['poids_utilisation'] * normalized_criteria['utilisation'][s] +
                params['poids_stabilite'] * normalized_criteria['stabilite'][s] +
                params['poids_penuries'] * normalized_criteria['penuries'][s]
            )
            global_scores[s] = score

        self.multicriteria_scores = {
            'normalized_criteria': normalized_criteria,
            'global_scores': global_scores,
            'criteria_values': criteria_values
        }

    def select_best_scenario_multicriteria(self):
        """Sélection du meilleur scénario"""
        if not self.multicriteria_scores:
            return None

        global_scores = self.multicriteria_scores['global_scores']
        
        # Identification du meilleur
        best_scenario_id = max(global_scores, key=global_scores.get)
        best_score = global_scores[best_scenario_id]

        # Classement complet
        sorted_scenarios = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Analyse de robustesse
        scores_values = list(global_scores.values())
        gap_with_second = sorted_scenarios[0][1] - sorted_scenarios[1][1] if len(sorted_scenarios) > 1 else 0
        robustness = "ÉLEVÉE" if gap_with_second > 0.1 else "MODÉRÉE" if gap_with_second > 0.05 else "FAIBLE"

        self.best_scenario_selection = {
            'best_scenario_id': best_scenario_id,
            'best_score': best_score,
            'ranking': sorted_scenarios,
            'robustness': robustness,
            'gap_with_second': gap_with_second
        }

        return {
            'best_scenario': best_scenario_id + 1,
            'score': best_score,
            'kpis': self.scenario_analysis[best_scenario_id]['kpis']
        }

class IntegratedPredictionPlanningSystem:
    def __init__(self):
        self.predictor = None
        self.planner = None
        self.predicted_rework_rate = None
        self.integration_results = {}

    def setup_prediction_system(self, data):
        self.predictor = MultiPosteDefectPredictor()
        self.predictor.original_data = data.copy()
        results, postes = self.predictor.train_all_postes(data, search_method='grid')
        return True

    def make_prediction_for_planning(self, jour, volume, method='moyenne_ponderee'):
        if self.predictor is None:
            raise ValueError("Le système de prédiction doit être configuré d'abord!")

        prediction_result = self.predictor.predict_single_scenario(jour, volume)
        taux_rework_chaine = prediction_result['taux_rework_chaine'][method]
        self.predicted_rework_rate = taux_rework_chaine

        return {
            'prediction_details': prediction_result,
            'rework_rate_for_planning': taux_rework_chaine,
            'method_used': method
        }

    def setup_planning_system(self, predicted_rework_rate=None, demandes_personnalisees=None, **params):
        if predicted_rework_rate is None:
            predicted_rework_rate = self.predicted_rework_rate

        if predicted_rework_rate is None:
            raise ValueError("Aucun taux de rework prédit disponible!")

        self.planner = StochasticPlanningModelComplete()
        
        # Si des demandes personnalisées sont fournies, les utiliser
        if demandes_personnalisees is not None:
            params['EDI'] = demandes_personnalisees

        self.planner.set_parameters(
            use_predicted_rework=True,
            predicted_rework_rate=predicted_rework_rate,
            **params
        )
        return True

    def run_integrated_planning(self, time_limit=300):
        if self.planner is None:
            raise ValueError("Le système de planification doit être configuré d'abord!")

        try:
            self.planner.create_model()
            self.planner.add_constraints()
            self.planner.set_objective()
            return self.planner.solve_model(time_limit=time_limit)
        except Exception as e:
            st.error(f"Erreur lors de l'exécution: {e}")
            return False

    def analyze_integrated_results(self):
        if self.planner is None or not hasattr(self.planner, 'results') or not self.planner.results:
            return None

        self.planner.analyze_scenarios_detailed()
        self.planner.calculate_multicriteria_scores()
        self.planner.select_best_scenario_multicriteria()
        
        self.integration_results = {
            'predicted_rework_rate': self.predicted_rework_rate,
            'planning_results': self.planner.results,
            'scenario_analysis': self.planner.scenario_analysis,
            'multicriteria_scores': self.planner.multicriteria_scores,
            'best_scenario_selection': self.planner.best_scenario_selection
        }

        return self.integration_results

# Fonctions d'interface Streamlit
def create_header():
    """Créer l'en-tête avec les logos et le titre"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Logo Yazaki stylisé
        st.markdown("""
        <div style='text-align: center; padding: 15px;'>
            <div style='
                background: linear-gradient(135deg, #1f4e79, #2c5aa0);
                color: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(31, 78, 121, 0.3);
                margin: 10px;
                border: 3px solid #1f4e79;
            '>
                <div style='font-size: 28px; margin-bottom: 8px;'>🏭</div>
                <div style='font-weight: bold; font-size: 20px; letter-spacing: 1px;'>YAZAKI</div>
                <div style='font-size: 12px; opacity: 0.9; margin-top: 5px;'>INDUSTRIAL SOLUTIONS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Titre central
        st.markdown("""
        <div class="main-header">
            <h1>🏭 Système Intégré Prédiction-Planification Avancé</h1>
            <p>Prédiction de défauts et planification stochastique multicritères</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Logo ENSAM stylisé
        st.markdown("""
        <div style='text-align: center; padding: 15px;'>
            <div style='
                background: linear-gradient(135deg, #2e86ab, #3a9bc1);
                color: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(46, 134, 171, 0.3);
                margin: 10px;
                border: 3px solid #2e86ab;
            '>
                <div style='font-size: 28px; margin-bottom: 8px;'>🎓</div>
                <div style='font-weight: bold; font-size: 20px; letter-spacing: 1px;'>ENSAM</div>
                <div style='font-size: 12px; opacity: 0.9; margin-top: 5px;'>ARTS ET MÉTIERS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Ligne de séparation
    st.markdown("---")
    
    # Bannière de partenariat
    st.markdown("""
    <div style='
        text-align: center; 
        padding: 15px; 
        background: linear-gradient(90deg, rgba(31, 78, 121, 0.1), rgba(46, 134, 171, 0.1)); 
        border-radius: 10px; 
        margin-bottom: 20px;
        border-left: 4px solid #1f4e79;
        border-right: 4px solid #2e86ab;
    '>
        <div style='display: flex; justify-content: center; align-items: center; gap: 20px; flex-wrap: wrap;'>
            <span style='color: #1f4e79; font-weight: bold; font-size: 16px;'>🏭 YAZAKI</span>
            <span style='color: #666; font-size: 20px;'>⚡</span>
            <span style='color: #2e86ab; font-weight: bold; font-size: 16px;'>ENSAM 🎓</span>
        </div>
        <div style='color: #666; margin-top: 8px; font-size: 14px; font-style: italic;'>
            Partenariat Industriel-Académique | Innovation & Excellence | Planification Stochastique Avancée
        </div>
    </div>
    """, unsafe_allow_html=True)

def load_data_section():
    st.header("📊 Chargement des Données")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Chargez votre fichier Excel contenant les données historiques",
            type=['xlsx', 'xls'],
            help="Le fichier doit contenir les colonnes: Jour, Volume_production, et les colonnes de défauts par poste"
        )
    
    with col2:
        if st.button("📝 Utiliser des données de démo", use_container_width=True):
            with st.spinner("Génération des données de démonstration..."):
                demo_data = create_demo_data()
                st.success(f"✅ Données de démo générées: {len(demo_data)} lignes")
                
                # Affichage des données de démonstration
                display_demo_data(demo_data)
                
                return demo_data
    
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.success(f"✅ Données chargées: {len(data)} lignes, {len(data.columns)} colonnes")
            
            with st.expander("👀 Aperçu des données"):
                st.dataframe(data.head())
                st.write("**Colonnes disponibles:**", list(data.columns))
            
            return data
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement: {e}")
            return None
    
    return None

def display_demo_data(demo_data):
    """Affiche les données de démonstration avec des statistiques et visualisations"""
    st.subheader("📊 Données de Démonstration Générées")
    
    # Statistiques générales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre de jours", len(demo_data))
    
    with col2:
        volume_moyen = demo_data['Volume_production'].mean()
        st.metric("Volume moyen", f"{volume_moyen:.0f}")
    
    with col3:
        defauts_total = demo_data[['Poste1_defauts', 'Poste2_defauts', 'Poste3_defauts']].sum().sum()
        st.metric("Total défauts", f"{defauts_total:.0f}")
    
    with col4:
        taux_defaut_moyen = (defauts_total / demo_data['Volume_production'].sum()) * 100
        st.metric("Taux défaut moyen", f"{taux_defaut_moyen:.2f}%")
    
    # Aperçu des données
    with st.expander("👀 Aperçu des Données Générées", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Premières lignes des données:**")
            st.dataframe(demo_data.head(10), use_container_width=True)
        
        with col2:
            st.write("**Statistiques descriptives:**")
            stats_df = demo_data.describe().round(2)
            st.dataframe(stats_df)

def create_demo_data(n_days=100):
    """Crée des données de démonstration avec des valeurs réalistes"""
    np.random.seed(42)
    days = range(1, n_days + 1)
    data = []

    for day in days:
        jour_semaine = ((day - 1) % 7) + 1
        
        if jour_semaine in [6, 7]:  # Weekend
            volume_base = 800
        else:  # Jours de semaine
            volume_base = 1200

        volume = volume_base + np.random.normal(0, 100)
        volume = max(volume, 100)

        # Générer des défauts réalistes pour chaque poste
        poste1_defauts = volume * 0.02 + jour_semaine * 0.5 + np.random.normal(0, 2)
        poste2_defauts = volume * 0.015 + jour_semaine * 0.3 + np.random.normal(0, 1.5)
        poste3_defauts = volume * 0.025 + jour_semaine * 0.4 + np.random.normal(0, 2.5)

        # S'assurer que les défauts sont positifs
        poste1_defauts = max(0, poste1_defauts)
        poste2_defauts = max(0, poste2_defauts)
        poste3_defauts = max(0, poste3_defauts)

        data.append({
            'Jour': jour_semaine,
            'Volume_production': volume,
            'Poste1_defauts': poste1_defauts,
            'Poste2_defauts': poste2_defauts,
            'Poste3_defauts': poste3_defauts
        })

    return pd.DataFrame(data)

def prediction_section(system, data):
    st.header("🔮 Prédiction de Défauts")
    
    if data is None:
        st.warning("⚠️ Veuillez d'abord charger des données")
        return None
    
    # Configuration et entraînement du système de prédiction
    with st.spinner("🧠 Entraînement des modèles de prédiction..."):
        success = system.setup_prediction_system(data)
    
    if success:
        st.success("✅ Modèles de prédiction entraînés avec succès!")
        
        # Afficher le taux de rework initial
        st.subheader("📊 Taux de Rework Initial de la Chaîne")
        
        # Calculer le taux moyen historique
        postes = system.predictor.postes
        taux_historiques = {}
        
        for poste in postes:
            if poste in data.columns and system.predictor.volume_col in data.columns:
                defauts_totaux = data[poste].sum()
                volume_total = data[system.predictor.volume_col].sum()
                taux_historiques[poste] = (defauts_totaux / volume_total) * 100 if volume_total > 0 else 0
        
        # Calculer le taux pondéré historique
        taux_pondere_historique = 0
        if system.predictor.poste_weights:
            for poste, poids in system.predictor.poste_weights.items():
                if poste in taux_historiques:
                    taux_pondere_historique += taux_historiques[poste] * poids
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Taux Rework Historique (Moyenne Pondérée)",
                f"{taux_pondere_historique:.2f}%",
                help="Basé sur les données historiques avec pondération Q*D"
            )
        
        with col2:
            taux_moyen_simple = np.mean(list(taux_historiques.values()))
            st.metric(
                "Taux Rework Historique (Moyenne Simple)",
                f"{taux_moyen_simple:.2f}%"
            )
        
        with col3:
            st.metric("Nombre de Postes", len(postes))
        
        return True
    else:
        st.error("❌ Échec de l'entraînement des modèles")
        return False

def new_prediction_section(system):
    st.header("🎯 Nouvelle Prédiction")
    
    if system.predictor is None:
        st.warning("⚠️ Veuillez d'abord configurer le système de prédiction")
        return None
    
    col1, col2 = st.columns(2)
    
    with col1:
        jour = st.selectbox(
            "Jour de la semaine",
            options=[1, 2, 3, 4, 5, 6, 7],
            format_func=lambda x: ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][x-1],
            index=2  # Mercredi par défaut
        )
    
    with col2:
        volume = st.number_input(
            "Volume de production prévu",
            min_value=1,
            max_value=10000,
            value=1200,
            step=50
        )
    
    if st.button("🔮 Faire la Prédiction", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            prediction_result = system.make_prediction_for_planning(jour, volume)
        
        st.subheader("📊 Résultats de Prédiction")
        
        # Métriques principales
        pred_details = prediction_result['prediction_details']
        taux_rework_nouveau = prediction_result['rework_rate_for_planning']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Nouveau Taux Rework Chaîne",
                f"{taux_rework_nouveau:.2f}%",
                help="Calculé avec moyenne pondérée"
            )
        
        with col2:
            defauts_predits = pred_details['predictions_chaine']['moyenne_ponderee']
            st.metric("Défauts Prédits", f"{defauts_predits:.1f}")
        
        with col3:
            st.metric("Volume Analysé", f"{volume:,.0f}")
        
        with col4:
            jour_name = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][jour-1]
            st.metric("Jour Analysé", jour_name)
        
        return prediction_result
    
    return None

def planning_section(system, prediction_result):
    st.header("📋 Planification Stochastique Avancée")
    
    if prediction_result is None:
        st.warning("⚠️ Veuillez d'abord faire une prédiction")
        return None
    
    st.subheader("⚙️ Configuration de la Planification")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        S = st.number_input("Nombre de scénarios", min_value=1, max_value=10, value=5)
        T = st.number_input("Nombre de shifts", min_value=1, max_value=5, value=3)
    
    with col2:
        capacity = st.number_input("Capacité par shift", min_value=50, max_value=1000, value=160)
        alpha_rework = st.slider("Taux de récupération rework", 0.0, 1.0, 0.8, 0.1)
    
    with col3:
        beta = st.slider("Facteur capacité rework", 1.0, 2.0, 1.2, 0.1)
        penalite = st.number_input("Pénalité pénurie", min_value=100, max_value=10000, value=1000)
    
    # Configuration des poids multicritères
    st.subheader("⚖️ Poids Multicritères")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        poids_cout = st.slider("Poids Coût", 0.0, 1.0, 0.25, 0.05)
    with col2:
        poids_satisfaction = st.slider("Poids Satisfaction", 0.0, 1.0, 0.30, 0.05)
    with col3:
        poids_utilisation = st.slider("Poids Utilisation", 0.0, 1.0, 0.20, 0.05)
    with col4:
        poids_stabilite = st.slider("Poids Stabilité", 0.0, 1.0, 0.15, 0.05)
    with col5:
        poids_penuries = st.slider("Poids Pénuries", 0.0, 1.0, 0.10, 0.05)
    
    # Normalisation des poids
    total_poids = poids_cout + poids_satisfaction + poids_utilisation + poids_stabilite + poids_penuries
    if total_poids > 0:
        poids_cout /= total_poids
        poids_satisfaction /= total_poids
        poids_utilisation /= total_poids
        poids_stabilite /= total_poids
        poids_penuries /= total_poids
    
    st.info(f"📊 Poids normalisés: Coût({poids_cout:.1%}), Satisfaction({poids_satisfaction:.1%}), Utilisation({poids_utilisation:.1%}), Stabilité({poids_stabilite:.1%}), Pénuries({poids_penuries:.1%})")
    
    # Configuration des demandes personnalisées
    st.subheader("📦 Demandes Personnalisées")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Modifiez les demandes pour chaque référence:")
        
        # Demandes par défaut
        references = [f'REF_{i+1:02d}' for i in range(10)]
        demandes_default = [20, 35, 45, 25, 40, 50, 22, 38, 30, 42]
        
        # Interface pour modifier les demandes
        demandes_personnalisees = []
        cols = st.columns(5)
        
        for i, (ref, demande_def) in enumerate(zip(references, demandes_default)):
            with cols[i % 5]:
                demande = st.number_input(
                    ref,
                    min_value=0,
                    max_value=200,
                    value=demande_def,
                    key=f"demande_{ref}"
                )
                demandes_personnalisees.append(demande)
    
    with col2:
        st.metric("Demande Totale", f"{sum(demandes_personnalisees):,.0f}")
        
        if st.button("🔄 Reset Demandes", use_container_width=True):
            st.rerun()
    
    # Bouton de lancement
    if st.button("🚀 Lancer la Planification Avancée", use_container_width=True):
        with st.spinner("Optimisation stochastique en cours..."):
            # Configuration du système de planification
            success_setup = system.setup_planning_system(
                S=S, T=T,
                mean_capacity=capacity,
                alpha_rework=alpha_rework,
                beta=beta,
                penalite_penurie=penalite,
                demandes_personnalisees=demandes_personnalisees,
                poids_cout=poids_cout,
                poids_satisfaction=poids_satisfaction,
                poids_utilisation=poids_utilisation,
                poids_stabilite=poids_stabilite,
                poids_penuries=poids_penuries
            )
            
            if success_setup:
                # Exécution de la planification
                success_planning = system.run_integrated_planning(time_limit=600)
                
                if success_planning:
                    # Analyse des résultats
                    results = system.analyze_integrated_results()
                    
                    if results:
                        st.success("✅ Planification stochastique réussie!")
                        
                        # Affichage immédiat des scénarios
                        display_scenario_details_advanced(system)
                        
                        return results
                    else:
                        st.error("❌ Erreur lors de l'analyse des résultats")
                else:
                    st.error("❌ Échec de la planification - Pas de solution optimale")
            else:
                st.error("❌ Erreur de configuration")
    
    return None

def display_scenario_details_advanced(system):
    """Affiche les détails avancés de tous les scénarios avec analyse multicritères"""
    st.subheader("🎯 Analyse Multicritères des Scénarios")
    
    if not hasattr(system.planner, 'scenario_analysis') or not system.planner.scenario_analysis:
        st.error("❌ Aucune analyse de scénario disponible")
        return
    
    scenario_analysis = system.planner.scenario_analysis
    params = system.planner.parameters
    
    # Affichage des scores multicritères si disponibles
    if hasattr(system.planner, 'multicriteria_scores') and system.planner.multicriteria_scores:
        st.subheader("🏆 Scores Multicritères")
        
        # Tableau comparatif
        comparison_data = []
        global_scores = system.planner.multicriteria_scores['global_scores']
        
        for s in range(len(scenario_analysis)):
            kpis = scenario_analysis[s]['kpis']
            comparison_data.append({
                'Scénario': f'S{s+1}',
                'Score Global': f"{global_scores[s]:.3f}",
                'Coût Total': f"{kpis['cout_total']:.0f}",
                'Satisfaction (%)': f"{kpis['satisfaction_globale']:.1f}",
                'Utilisation (%)': f"{kpis['utilisation_capacite']:.1f}",
                'Stabilité': f"{kpis['stabilite']:.1f}",
                'Pénuries': f"{kpis['total_penuries']:.1f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Meilleur scénario
        if hasattr(system.planner, 'best_scenario_selection') and system.planner.best_scenario_selection:
            best_selection = system.planner.best_scenario_selection
            best_scenario_id = best_selection['best_scenario_id']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🥇 Meilleur Scénario", f"S{best_scenario_id + 1}")
            with col2:
                st.metric("📊 Score Multicritères", f"{best_selection['best_score']:.3f}")
            with col3:
                st.metric("🛡️ Robustesse", best_selection['robustness'])
    
    # Tabs pour chaque scénario
    tab_names = [f"Scénario {s+1}" for s in range(len(scenario_analysis))]
    tabs = st.tabs(tab_names)
    
    for tab_idx, (s, scenario_data) in enumerate(scenario_analysis.items()):
        with tabs[tab_idx]:
            
            # KPIs du scénario
            kpis = scenario_data['kpis']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Satisfaction", f"{kpis['satisfaction_globale']:.1f}%")
            with col2:
                st.metric("Utilisation Capacité", f"{kpis['utilisation_capacite']:.1f}%")
            with col3:
                st.metric("Stabilité", f"{kpis['stabilite']:.1f}")
            with col4:
                st.metric("Coût Total", f"{kpis['cout_total']:,.0f}")
            
            # Détails par shift
            st.write("### 📋 Plan d'Exécution par Shift")
            
            for t in range(params['T']):
                shift_key = t + 1
                if shift_key not in scenario_data['shifts_details']:
                    st.error(f"Données manquantes pour le shift {shift_key} du scénario {s+1}")
                    continue
                    
                shift_info = scenario_data['shifts_details'][shift_key]
                
                st.write(f"#### 🔄 Shift {t+1} - Scénario {s+1}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Ordre d'exécution
                    ordre_execution = shift_info['execution_order']
                    ordre_clean = [ref for ref in ordre_execution if ref != 'VIDE']
                    
                    if ordre_clean:
                        st.write("**Ordre d'exécution:**")
                        ordre_display = " → ".join(ordre_clean)
                        st.markdown(f"`{ordre_display}`")
                    else:
                        st.write("**Aucune production programmée**")
                    
                    # Quantités détaillées
                    if shift_info['quantities']:
                        st.write("**Quantités à produire:**")
                        
                        # Créer un DataFrame pour l'affichage
                        quantities_data = []
                        for ref, qty in shift_info['quantities'].items():
                            if qty > 0:
                                taux_defaut = params['taux_defaut'][(s, ref)]
                                prod_utile = qty * (1 - taux_defaut)
                                prod_recuperee = qty * taux_defaut * params['alpha_rework']
                                total_utile = prod_utile + prod_recuperee
                                
                                quantities_data.append({
                                    'Référence': ref,
                                    'Quantité Brute': f"{qty:.0f}",
                                    'Production Utile': f"{total_utile:.0f}",
                                    'Taux Défaut': f"{taux_defaut*100:.1f}%"
                                })
                        
                        if quantities_data:
                            df_quantities = pd.DataFrame(quantities_data)
                            st.dataframe(df_quantities, hide_index=True, use_container_width=True)
                
                with col2:
                    # Métriques du shift
                    st.write("**Métriques du Shift:**")
                    st.metric("Capacité Utilisée", 
                             f"{shift_info['capacity_used']:.0f}/{shift_info['capacity_available']:.0f}")
                    st.metric("Taux d'Utilisation", 
                             f"{shift_info['capacity_utilization']:.1f}%")
                    
                    nb_refs_actives = len([ref for ref, qty in shift_info['quantities'].items() if qty > 0])
                    st.metric("Références Actives", nb_refs_actives)
                
                st.markdown("---")
            
            # Résumé production par référence
            st.write("### 📊 Résumé Production par Référence")
            
            production_summary = []
            for ref, info in scenario_data['production_summary'].items():
                production_summary.append({
                    'Référence': ref,
                    'Demande': f"{info['demande']:.0f}",
                    'Production Brute': f"{info['production_brute']:.0f}",
                    'Production Utile': f"{info['production_utile']:.0f}",
                    'Pénurie': f"{info['penurie']:.0f}",
                    'Taux Couverture': f"{info['taux_couverture']:.1f}%"
                })
            
            df_production = pd.DataFrame(production_summary)
            st.dataframe(df_production, hide_index=True, use_container_width=True)

def dashboard_section_advanced(system, results):
    st.header("📊 Dashboard Multicritères Avancé")
    
    if results is None:
        st.warning("⚠️ Aucun résultat de planification disponible")
        return
    
    if not system.planner or not hasattr(system.planner, 'scenario_analysis'):
        st.error("❌ Données d'analyse manquantes")
        return
    
    # Vérification du taux de rework intégré
    st.subheader("🔗 Intégration Prédiction-Planification")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Taux Rework Prédit", f"{results['predicted_rework_rate']:.2f}%")
    
    with col2:
        # Vérifier que le taux est bien utilisé dans la planification
        taux_utilise = system.planner.parameters['taux_defaut'][(0, system.planner.parameters['R'][0])] * 100
        st.metric("Taux Utilisé en Planification", f"{taux_utilise:.2f}%")
    
    with col3:
        match = abs(results['predicted_rework_rate'] - taux_utilise) < 0.01
        st.metric("Intégration", "✅ Réussie" if match else "❌ Échouée")
    
    # Analyse multicritères
    if 'multicriteria_scores' in results and results['multicriteria_scores']:
        st.subheader("🏆 Analyse Multicritères")
        
        scores = results['multicriteria_scores']
        scenario_analysis = results['scenario_analysis']
        
        # Graphique radar des critères
        col1, col2 = st.columns(2)
        
        with col1:
            # Tableau des scores
            comparison_data = []
            for s in range(len(scenario_analysis)):
                kpis = scenario_analysis[s]['kpis']
                global_score = scores['global_scores'][s]
                
                comparison_data.append({
                    'Scénario': f'S{s+1}',
                    'Score Global': f"{global_score:.3f}",
                    'Satisfaction': f"{kpis['satisfaction_globale']:.1f}%",
                    'Utilisation': f"{kpis['utilisation_capacite']:.1f}%",
                    'Coût': f"{kpis['cout_total']:,.0f}",
                    'Stabilité': f"{kpis['stabilite']:.1f}",
                    'Pénuries': f"{kpis['total_penuries']:.1f}"
                })
            
            df_scores = pd.DataFrame(comparison_data)
            st.dataframe(df_scores, use_container_width=True)
        
        with col2:
            # Graphique des scores globaux
            scenarios = [f'S{s+1}' for s in range(len(scenario_analysis))]
            global_scores_values = list(scores['global_scores'].values())
            
            fig = px.bar(
                x=scenarios, 
                y=global_scores_values,
                title="Scores Multicritères par Scénario",
                labels={'x': 'Scénarios', 'y': 'Score Global'}
            )
            
            # Colorer le meilleur en or
            colors = ['gold' if score == max(global_scores_values) else 'lightblue' 
                     for score in global_scores_values]
            fig.update_traces(marker_color=colors)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Meilleur scénario recommandé
        if 'best_scenario_selection' in results and results['best_scenario_selection']:
            best_selection = results['best_scenario_selection']
            
            st.subheader("🥇 Scénario Recommandé")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Meilleur Scénario", f"S{best_selection['best_scenario_id'] + 1}")
            
            with col2:
                st.metric("Score Multicritères", f"{best_selection['best_score']:.3f}")
            
            with col3:
                st.metric("Écart avec 2ème", f"{best_selection['gap_with_second']:.3f}")
            
            with col4:
                st.metric("Robustesse", best_selection['robustness'])
            
            # Plan d'exécution recommandé
            best_id = best_selection['best_scenario_id']
            best_scenario_data = scenario_analysis[best_id]
            
            st.write("### 📋 Plan d'Exécution Recommandé")
            
            for t in range(system.planner.parameters['T']):
                shift_key = t + 1
                if shift_key in best_scenario_data['shifts_details']:
                    shift_info = best_scenario_data['shifts_details'][shift_key]
                    ordre = ' → '.join([ref for ref in shift_info['execution_order'] if ref != 'VIDE'])
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Shift {t+1}:** {ordre}")
                    with col2:
                        st.write(f"Util: {shift_info['capacity_utilization']:.1f}%")

def export_section_advanced(system, results):
    st.header("📁 Export des Résultats Avancés")
    
    if results is None:
        st.warning("⚠️ Aucun résultat à exporter")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Exporter vers Excel Complet", use_container_width=True):
            try:
                # Créer un buffer en mémoire
                output = io.BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # 1. Résumé intégré avec prédiction
                    summary_data = [{
                        'Taux_Rework_Predit_Pct': results['predicted_rework_rate'],
                        'Cout_Total_Optimal': results['planning_results']['cout_total'],
                        'Nombre_Scenarios': len(results['scenario_analysis']),
                        'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    }]
                    
                    df_summary = pd.DataFrame(summary_data)
                    df_summary.to_excel(writer, sheet_name='Resume_Integration', index=False)
                    
                    # 2. Scores multicritères si disponibles
                    if 'multicriteria_scores' in results:
                        scores_data = []
                        for s in range(len(results['scenario_analysis'])):
                            scores_data.append({
                                'Scenario': f'S{s+1}',
                                'Score_Global': results['multicriteria_scores']['global_scores'][s],
                                'Score_Cout_Norm': results['multicriteria_scores']['normalized_criteria']['cout'][s],
                                'Score_Satisfaction_Norm': results['multicriteria_scores']['normalized_criteria']['satisfaction'][s],
                                'Score_Utilisation_Norm': results['multicriteria_scores']['normalized_criteria']['utilisation'][s],
                                'Score_Stabilite_Norm': results['multicriteria_scores']['normalized_criteria']['stabilite'][s],
                                'Score_Penuries_Norm': results['multicriteria_scores']['normalized_criteria']['penuries'][s]
                            })
                        
                        df_scores = pd.DataFrame(scores_data)
                        df_scores.to_excel(writer, sheet_name='Scores_Multicriteres', index=False)
                    
                    # 3. Comparaison des scénarios détaillée
                    planning_comparison = []
                    for s, data in results['scenario_analysis'].items():
                        kpis = data['kpis']
                        planning_comparison.append({
                            'Scenario': f'S{s+1}',
                            'Satisfaction_Pct': kpis['satisfaction_globale'],
                            'Utilisation_Capacite_Pct': kpis['utilisation_capacite'],
                            'Stabilite': kpis['stabilite'],
                            'Total_Penuries': kpis['total_penuries'],
                            'Cout_Total': kpis['cout_total'],
                            'Cout_Production': kpis['cout_production'],
                            'Cout_Penuries': kpis['cout_penuries'],
                            'Efficacite_Production': kpis['efficacite_production']
                        })
                    
                    df_planning = pd.DataFrame(planning_comparison)
                    df_planning.to_excel(writer, sheet_name='Comparaison_Scenarios', index=False)
                    
                    # 4. Production détaillée par scénario
                    if hasattr(system.planner, 'results'):
                        prod_data = []
                        for s in range(len(results['scenario_analysis'])):
                            for ref in system.planner.parameters['R']:
                                for t in range(system.planner.parameters['T']):
                                    qty = system.planner.results['production'][(s, ref, t)]
                                    if qty > 0:
                                        prod_data.append({
                                            'Scenario': f'S{s+1}',
                                            'Reference': ref,
                                            'Shift': f'T{t+1}',
                                            'Quantite': qty,
                                            'Taux_Rework_Utilise': system.planner.parameters['taux_defaut'][(s, ref)] * 100
                                        })
                        
                        if prod_data:
                            df_prod = pd.DataFrame(prod_data)
                            df_prod.to_excel(writer, sheet_name='Production_Details', index=False)
                    
                    # 5. Meilleur scénario si disponible
                    if 'best_scenario_selection' in results:
                        best_data = [{
                            'Meilleur_Scenario': f"S{results['best_scenario_selection']['best_scenario_id'] + 1}",
                            'Score_Multicriteres': results['best_scenario_selection']['best_score'],
                            'Robustesse': results['best_scenario_selection']['robustness'],
                            'Ecart_2eme': results['best_scenario_selection']['gap_with_second']
                        }]
                        
                        df_best = pd.DataFrame(best_data)
                        df_best.to_excel(writer, sheet_name='Meilleur_Scenario', index=False)
                
                # Préparer le téléchargement
                processed_data = output.getvalue()
                
                st.download_button(
                    label="⬇️ Télécharger Excel Complet",
                    data=processed_data,
                    file_name=f"resultats_integres_avances_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Fichier Excel avancé généré!")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'export: {e}")
    
    with col2:
        if st.button("📊 Export Visualisations", use_container_width=True):
            st.info("🚧 Export des graphiques en développement")

# Application principale
def main():
    create_header()
    
    # Initialisation du système dans la session
    if 'system' not in st.session_state:
        st.session_state.system = IntegratedPredictionPlanningSystem()
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'prediction_trained' not in st.session_state:
        st.session_state.prediction_trained = False
    
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    if 'planning_results' not in st.session_state:
        st.session_state.planning_results = None
    
    # Sidebar pour navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        
        step = st.radio(
            "Choisissez une étape:",
            [
                "📊 1. Chargement des Données",
                "🔮 2. Prédiction de Défauts",
                "🎯 3. Nouvelle Prédiction",
                "📋 4. Planification Avancée",
                "📈 5. Dashboard Multicritères",
                "📁 6. Export Avancé"
            ]
        )
        
        st.markdown("---")
        
        # État du processus
        st.header("📋 État du Processus")
        
        if st.session_state.data is not None:
            st.success("✅ Données chargées")
        else:
            st.error("❌ Données non chargées")
        
        if st.session_state.prediction_trained:
            st.success("✅ Modèles entraînés")
        else:
            st.error("❌ Modèles non entraînés")
        
        if st.session_state.prediction_result is not None:
            st.success("✅ Prédiction effectuée")
        else:
            st.error("❌ Pas de prédiction")
        
        if st.session_state.planning_results is not None:
            st.success("✅ Planification terminée")
        else:
            st.error("❌ Planification non effectuée")
        
        st.markdown("---")
        
        # Bouton de reset
        if st.button("🔄 Recommencer", use_container_width=True):
            for key in ['system', 'data', 'prediction_trained', 'prediction_result', 'planning_results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        # Informations système
        st.markdown("---")
        st.header("ℹ️ Informations")
        st.markdown("""
        **Système Intégré Avancé v2.0**
        
        🔮 **Prédiction:** Modèles ML pour prédire les défauts
        
        📋 **Planification:** Optimisation stochastique multicritères
        
        📊 **Dashboard:** Analyse multicritères avancée
        
        ⚖️ **Nouveautés:**
        - Poids multicritères configurables
        - Analyse de robustesse
        - Score global pondéré
        - Export complet
        """)
    
    # Contenu principal selon l'étape sélectionnée
    if step == "📊 1. Chargement des Données":
        data = load_data_section()
        if data is not None:
            st.session_state.data = data
            # Reset des étapes suivantes si nouvelles données
            st.session_state.prediction_trained = False
            st.session_state.prediction_result = None
            st.session_state.planning_results = None
    
    elif step == "🔮 2. Prédiction de Défauts":
        if st.session_state.data is not None:
            success = prediction_section(st.session_state.system, st.session_state.data)
            if success:
                st.session_state.prediction_trained = True
        else:
            st.warning("⚠️ Veuillez d'abord charger des données à l'étape 1")
    
    elif step == "🎯 3. Nouvelle Prédiction":
        if st.session_state.prediction_trained:
            prediction_result = new_prediction_section(st.session_state.system)
            if prediction_result is not None:
                st.session_state.prediction_result = prediction_result
                # Reset de la planification si nouvelle prédiction
                st.session_state.planning_results = None
        else:
            st.warning("⚠️ Veuillez d'abord entraîner les modèles à l'étape 2")
    
    elif step == "📋 4. Planification Avancée":
        if st.session_state.prediction_result is not None:
            planning_results = planning_section(st.session_state.system, st.session_state.prediction_result)
            if planning_results is not None:
                st.session_state.planning_results = planning_results
        else:
            st.warning("⚠️ Veuillez d'abord effectuer une prédiction à l'étape 3")
    
    elif step == "📈 5. Dashboard Multicritères":
        if st.session_state.planning_results is not None:
            dashboard_section_advanced(st.session_state.system, st.session_state.planning_results)
        else:
            st.warning("⚠️ Veuillez d'abord effectuer la planification à l'étape 4")
    
    elif step == "📁 6. Export Avancé":
        if st.session_state.planning_results is not None:
            export_section_advanced(st.session_state.system, st.session_state.planning_results)
        else:
            st.warning("⚠️ Aucun résultat à exporter")
    
    # Footer avec logos en bas de page
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <div style='display: flex; justify-content: center; align-items: center; gap: 40px; margin-bottom: 15px;'>
            <div style='display: flex; align-items: center; gap: 10px;'>
                <span style='font-size: 24px;'>🏭</span>
                <span style='font-weight: bold; color: #1f4e79;'>YAZAKI</span>
            </div>
            <div style='color: #ccc; font-size: 20px;'>×</div>
            <div style='display: flex; align-items: center; gap: 10px;'>
                <span style='font-size: 24px;'>🎓</span>
                <span style='font-weight: bold; color: #2e86ab;'>ENSAM</span>
            </div>
        </div>
        <div style='color: #666; margin-top: 8px; font-size: 14px; font-style: italic;'>
            🏭 Système Intégré Avancé Prédiction-Planification Stochastique | 
            Développé avec ❤️ en Streamlit | 
            Modèle Multicritères v2.0 | 
            © 2024
        </div>
    </div>
    """, unsafe_allow_html=True)

# Point d'entrée
if __name__ == "__main__":
    main()
