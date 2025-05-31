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
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Système Intégré Prédiction-Planification",
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

# Classes du système intégré (copie des classes principales du code fourni)
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

class StochasticPlanningModelEnhanced:
    def __init__(self):
        self.model = None
        self.variables = {}
        self.results = {}
        self.parameters = {}
        self.scenario_analysis = {}
        self.predicted_rework_rate = None

    def set_parameters(self,
                      S: int = 3,
                      T: int = 3,
                      R: List[str] = None,
                      EDI: List = None,
                      p: List[List] = None,
                      D: List[List] = None,
                      seuil: float = 0.95,
                      mean_capacity: float = 200,
                      std_capacity: float = 20,
                      mean_defaut: float = 0.1,
                      std_defaut: float = 0.02,
                      m: int = 5,
                      alpha_rework: float = 0.8,
                      beta: float = 1.2,
                      b: int = 10,
                      penalite_penurie: float = 1000,
                      use_predicted_rework: bool = False,
                      predicted_rework_rate: float = None):

        if R is None:
            R = [f'REF_{i+1:02d}' for i in range(10)]

        if EDI is None:
            EDI = [20, 35, 45, 25, 40, 50, 22, 38, 30, 42]

        EDI_dict = {R[i]: EDI[i] for i in range(len(R))}

        if p is None:
            p = [
                [0.10, 0.20, 0.30, 0.15, 0.25, 0.35, 0.12, 0.22, 0.32, 0.18],
                [0.20, 0.10, 0.40, 0.25, 0.35, 0.45, 0.22, 0.32, 0.42, 0.28],
                [0.30, 0.40, 0.10, 0.35, 0.45, 0.55, 0.32, 0.42, 0.52, 0.38],
                [0.15, 0.25, 0.35, 0.10, 0.30, 0.40, 0.17, 0.27, 0.37, 0.23],
                [0.25, 0.35, 0.45, 0.30, 0.10, 0.50, 0.27, 0.37, 0.47, 0.33],
                [0.35, 0.45, 0.55, 0.40, 0.50, 0.10, 0.37, 0.47, 0.57, 0.43],
                [0.12, 0.22, 0.32, 0.17, 0.27, 0.37, 0.10, 0.24, 0.34, 0.20],
                [0.22, 0.32, 0.42, 0.27, 0.37, 0.47, 0.24, 0.10, 0.44, 0.30],
                [0.32, 0.42, 0.52, 0.37, 0.47, 0.57, 0.34, 0.44, 0.10, 0.40],
                [0.18, 0.28, 0.38, 0.23, 0.33, 0.43, 0.20, 0.30, 0.40, 0.10]
            ]

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

        p_dict = {}
        for i in range(len(R)):
            for j in range(len(R)):
                p_dict[(R[i], j)] = p[i][j]

        D_array = np.array(D)

        CAPchaine = {}
        for s in range(S):
            for t in range(T):
                CAPchaine[(s, t)] = mean_capacity

        np.random.seed(42)
        taux_defaut = {}

        if use_predicted_rework and predicted_rework_rate is not None:
            for s in range(S):
                for i in R:
                    taux_defaut[(s, i)] = predicted_rework_rate / 100
            self.predicted_rework_rate = predicted_rework_rate
        else:
            for s in range(S):
                for i in R:
                    defaut = max(0.01, min(0.3, np.random.normal(mean_defaut, std_defaut)))
                    taux_defaut[(s, i)] = defaut

        self.parameters = {
            'S': S, 'T': T, 'R': R, 'EDI': EDI_dict, 'p': p_dict, 'D': D_array,
            'seuil': seuil, 'CAPchaine': CAPchaine, 'm': m, 'taux_defaut': taux_defaut,
            'alpha_rework': alpha_rework, 'beta': beta, 'b': b, 'penalite_penurie': penalite_penurie,
            'mean_capacity': mean_capacity, 'std_capacity': std_capacity,
            'mean_defaut': mean_defaut, 'std_defaut': std_defaut,
            'use_predicted_rework': use_predicted_rework,
            'predicted_rework_rate': predicted_rework_rate
        }

    def create_model(self):
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']

        self.model = plp.LpProblem("Planification_Stochastique", plp.LpMinimize)

        self.variables['x'] = plp.LpVariable.dicts(
            "x",
            [(i, j, s, t) for i in R for j in range(len(R)) for s in range(S) for t in range(T)],
            cat='Binary'
        )

        self.variables['q'] = plp.LpVariable.dicts(
            "q",
            [(s, i, t) for s in range(S) for i in R for t in range(T)],
            lowBound=0,
            cat='Continuous'
        )

        self.variables['penurie'] = plp.LpVariable.dicts(
            "penurie",
            [(s, i) for s in range(S) for i in R],
            lowBound=0,
            cat='Continuous'
        )

    def add_constraints(self):
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        x, q, penurie = self.variables['x'], self.variables['q'], self.variables['penurie']

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

        for s in range(S):
            for t in range(T):
                for j in range(len(R)):
                    self.model += (
                        plp.lpSum([x[(i, j, s, t)] for i in R]) == 1,
                        f"Une_ref_par_position_s{s}_t{t}_j{j}"
                    )

        for s in range(S):
            for t in range(T):
                for i in R:
                    self.model += (
                        plp.lpSum([x[(i, j, s, t)] for j in range(len(R))]) <= 1,
                        f"Une_position_par_ref_s{s}_t{t}_i{i}"
                    )

        for s in range(S):
            for i in R:
                for t in range(T):
                    production_requise = params['m'] / (1 - params['taux_defaut'][(s, i)])
                    self.model += (
                        q[(s, i, t)] >= production_requise * plp.lpSum([x[(i, j, s, t)] for j in range(len(R))]),
                        f"Production_min_s{s}_i{i}_t{t}"
                    )

    def set_objective(self):
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        q, penurie = self.variables['q'], self.variables['penurie']

        cout_production = plp.lpSum([
            20 * q[(s, i, t)]
            for s in range(S) for i in R for t in range(T)
        ])

        cout_penuries = plp.lpSum([
            params['penalite_penurie'] * penurie[(s, i)]
            for s in range(S) for i in R
        ])

        self.model += cout_production + cout_penuries

    def solve_model(self, solver_name='PULP_CBC_CMD', time_limit=300):
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
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']

        production_results = {}
        for s in range(S):
            for i in R:
                for t in range(T):
                    key = (s, i, t)
                    production_results[key] = self.variables['q'][key].value() or 0

        penuries_results = {}
        for s in range(S):
            for i in R:
                key = (s, i)
                penuries_results[key] = self.variables['penurie'][key].value() or 0

        sequencement_results = {}
        for s in range(S):
            for t in range(T):
                sequence = {}
                for i in R:
                    for j in range(len(R)):
                        key = (i, j, s, t)
                        if self.variables['x'][key].value() and self.variables['x'][key].value() > 0.5:
                            sequence[j] = i
                sequencement_results[(s, t)] = sequence

        self.results = {
            'production': production_results,
            'penuries': penuries_results,
            'sequencement': sequencement_results,
            'cout_total': self.model.objective.value()
        }

    def analyze_scenarios_detailed(self):
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
                'penalties': {},
                'kpis': {}
            }

            for t in range(T):
                shift_info = {
                    'execution_order': [],
                    'quantities': {},
                    'capacity_used': 0,
                    'capacity_available': params['CAPchaine'][(s, t)]
                }

                sequence = sequencement.get((s, t), {})
                ordered_refs = [sequence.get(j, 'VIDE') for j in range(len(R))]
                shift_info['execution_order'] = ordered_refs

                capacity_used = 0
                for i in R:
                    qty = production[(s, i, t)]
                    if qty > 0:
                        shift_info['quantities'][i] = qty
                        taux_def = params['taux_defaut'][(s, i)]
                        capacity_used += qty * (1 + params['beta'] * taux_def)

                shift_info['capacity_used'] = capacity_used
                shift_info['capacity_utilization'] = (capacity_used / shift_info['capacity_available']) * 100

                scenario_data['shifts_details'][t+1] = shift_info

            for i in R:
                total_prod = sum(production[(s, i, t)] for t in range(T))
                total_utile = 0
                for t in range(T):
                    qty = production[(s, i, t)]
                    taux_def = params['taux_defaut'][(s, i)]
                    total_utile += qty * (1 - taux_def) + qty * taux_def * params['alpha_rework']

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

            total_production_utile = sum([data['production_utile'] for data in scenario_data['production_summary'].values()])
            total_demande = sum([data['demande'] for data in scenario_data['production_summary'].values()])
            total_penuries = sum([data['penurie'] for data in scenario_data['production_summary'].values()])

            total_capacity_used = sum([shift['capacity_used'] for shift in scenario_data['shifts_details'].values()])
            total_capacity_available = sum([shift['capacity_available'] for shift in scenario_data['shifts_details'].values()])

            cout_scenario = sum([production[(s, i, t)] * 20 for i in R for t in range(T)]) + total_penuries * params['penalite_penurie']

            scenario_kpis = {
                'satisfaction_globale': (total_production_utile / total_demande) * 100 if total_demande > 0 else 0,
                'utilisation_capacite': (total_capacity_used / total_capacity_available) * 100 if total_capacity_available > 0 else 0,
                'total_penuries': total_penuries,
                'cout_estime': cout_scenario,
                'efficacite_production': total_production_utile / max(1, total_capacity_used)
            }

            scenario_data['kpis'] = scenario_kpis
            self.scenario_analysis[s] = scenario_data

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

        self.planner = StochasticPlanningModelEnhanced()
        
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
        
        self.integration_results = {
            'predicted_rework_rate': self.predicted_rework_rate,
            'planning_results': self.planner.results,
            'scenario_analysis': self.planner.scenario_analysis
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
            <h1>🏭 Système Intégré Prédiction-Planification</h1>
            <p>Prédiction de défauts et planification stochastique optimisée</p>
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
            Partenariat Industriel-Académique | Innovation & Excellence
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
        st.metric(
            "Nombre de jours",
            len(demo_data)
        )
    
    with col2:
        volume_moyen = demo_data['Volume_production'].mean()
        st.metric(
            "Volume moyen",
            f"{volume_moyen:.0f}"
        )
    
    with col3:
        defauts_total = demo_data[['Poste1_defauts', 'Poste2_defauts', 'Poste3_defauts']].sum().sum()
        st.metric(
            "Total défauts",
            f"{defauts_total:.0f}"
        )
    
    with col4:
        taux_defaut_moyen = (defauts_total / demo_data['Volume_production'].sum()) * 100
        st.metric(
            "Taux défaut moyen",
            f"{taux_defaut_moyen:.2f}%"
        )
    
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
    
    # Visualisations des données de démo
    st.write("### 📈 Visualisations des Données")
    
    # Graphique 1: Évolution du volume de production
    col1, col2 = st.columns(2)
    
    with col1:
        fig_volume = px.line(
            demo_data, 
            x=demo_data.index, 
            y='Volume_production',
            title="Évolution du Volume de Production",
            labels={'x': 'Jours', 'Volume_production': 'Volume'}
        )
        fig_volume.update_traces(line=dict(color='blue', width=2))
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        # Graphique des défauts par poste
        defauts_cols = ['Poste1_defauts', 'Poste2_defauts', 'Poste3_defauts']
        fig_defauts = px.line(
            demo_data, 
            x=demo_data.index,
            y=defauts_cols,
            title="Évolution des Défauts par Poste",
            labels={'x': 'Jours', 'value': 'Nombre de défauts', 'variable': 'Poste'}
        )
        st.plotly_chart(fig_defauts, use_container_width=True)
    
    # Graphiques supplémentaires
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution du volume par jour de la semaine
        volume_par_jour = demo_data.groupby('Jour')['Volume_production'].mean().reset_index()
        volume_par_jour['Jour_nom'] = volume_par_jour['Jour'].map({
            1: 'Lundi', 2: 'Mardi', 3: 'Mercredi', 4: 'Jeudi', 
            5: 'Vendredi', 6: 'Samedi', 7: 'Dimanche'
        })
        
        fig_jour = px.bar(
            volume_par_jour,
            x='Jour_nom',
            y='Volume_production',
            title="Volume Moyen par Jour de la Semaine",
            color='Volume_production',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_jour, use_container_width=True)
    
    with col2:
        # Taux de défaut par poste
        taux_defauts = []
        for poste in defauts_cols:
            taux = (demo_data[poste].sum() / demo_data['Volume_production'].sum()) * 100
            taux_defauts.append({
                'Poste': poste.replace('_defauts', ''),
                'Taux_defaut': taux
            })
        
        df_taux = pd.DataFrame(taux_defauts)
        fig_taux = px.bar(
            df_taux,
            x='Poste',
            y='Taux_defaut',
            title="Taux de Défaut par Poste (%)",
            color='Taux_defaut',
            color_continuous_scale='reds'
        )
        fig_taux.update_layout(showlegend=False)
        st.plotly_chart(fig_taux, use_container_width=True)
    
    # Corrélations
    with st.expander("🔍 Analyse de Corrélation"):
        st.write("**Matrice de corrélation entre les variables:**")
        
        # Calculer la matrice de corrélation
        corr_matrix = demo_data[['Volume_production'] + defauts_cols].corr()
        
        # Créer un heatmap
        fig_corr = px.imshow(
            corr_matrix,
            title="Matrice de Corrélation",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig_corr.update_layout(width=600, height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Afficher la matrice numériquement
        st.write("**Valeurs de corrélation:**")
        st.dataframe(corr_matrix.round(3))
    
    # Résumé des caractéristiques
    with st.expander("📋 Caractéristiques des Données de Démonstration"):
        st.markdown("""
        **🎯 Caractéristiques des données générées:**
        
        - **Période:** 100 jours simulés avec variation saisonnière
        - **Volume de production:** 
          - Jours ouvrables: ~1200 unités (± 100)
          - Week-ends: ~800 unités (± 100)
        - **Défauts par poste:**
          - Poste1: ~2% du volume + variation jour + bruit
          - Poste2: ~1.5% du volume + variation jour + bruit  
          - Poste3: ~2.5% du volume + variation jour + bruit
        - **Corrélations:** Les défauts sont corrélés au volume et au jour de la semaine
        - **Réalisme:** Données basées sur des patterns industriels typiques
        
        **📊 Utilisation:**
        Ces données permettent de tester le système de prédiction et de planification
        avec des patterns réalistes de production industrielle.
        """)
    
    st.success("✅ Données de démonstration prêtes pour l'analyse !")

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

    df = pd.DataFrame(data)
    
    # Vérification des données générées
    print(f"Données de démo générées: {len(df)} lignes")
    print(f"Colonnes: {list(df.columns)}")
    print(f"Volume moyen: {df['Volume_production'].mean():.1f}")
    
    return df

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
            st.metric(
                "Nombre de Postes",
                len(postes)
            )
        
        # Graphique des taux par poste
        if taux_historiques:
            fig_taux = px.bar(
                x=list(taux_historiques.keys()),
                y=list(taux_historiques.values()),
                title="Taux de Rework Historique par Poste",
                labels={'x': 'Postes', 'y': 'Taux de Rework (%)'}
            )
            fig_taux.update_layout(showlegend=False)
            st.plotly_chart(fig_taux, use_container_width=True)
        
        # Afficher les poids de pondération
        with st.expander("⚖️ Poids de Pondération des Postes"):
            if system.predictor.poste_weights:
                weights_df = pd.DataFrame([
                    {'Poste': poste, 'Poids': poids, 'Pourcentage': f"{poids*100:.2f}%"}
                    for poste, poids in system.predictor.poste_weights.items()
                ])
                st.dataframe(weights_df)
        
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
            st.metric(
                "Défauts Prédits",
                f"{defauts_predits:.1f}"
            )
        
        with col3:
            st.metric(
                "Volume Analysé",
                f"{volume:,.0f}"
            )
        
        with col4:
            jour_name = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][jour-1]
            st.metric(
                "Jour Analysé",
                jour_name
            )
        
        # Graphique comparatif des méthodes
        methodes = list(pred_details['taux_rework_chaine'].keys())
        taux_values = list(pred_details['taux_rework_chaine'].values())
        
        fig_methodes = px.bar(
            x=methodes,
            y=taux_values,
            title="Taux de Rework selon Différentes Méthodes",
            labels={'x': 'Méthode', 'y': 'Taux de Rework (%)'}
        )
        fig_methodes.update_layout(showlegend=False)
        st.plotly_chart(fig_methodes, use_container_width=True)
        
        # Détail par poste
        with st.expander("🏭 Détail par Poste"):
            postes_data = []
            for poste, defauts in pred_details['predictions_postes'].items():
                taux = pred_details['taux_rework_postes'][poste]
                postes_data.append({
                    'Poste': poste,
                    'Défauts Prédits': f"{defauts:.1f}",
                    'Taux Rework (%)': f"{taux:.2f}"
                })
            
            df_postes = pd.DataFrame(postes_data)
            st.dataframe(df_postes)
        
        return prediction_result
    
    return None

def planning_section(system, prediction_result):
    st.header("📋 Planification Stochastique")
    
    if prediction_result is None:
        st.warning("⚠️ Veuillez d'abord faire une prédiction")
        return None
    
    st.subheader("⚙️ Configuration de la Planification")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        S = st.number_input("Nombre de scénarios", min_value=1, max_value=10, value=3)
        T = st.number_input("Nombre de shifts", min_value=1, max_value=5, value=3)
    
    with col2:
        capacity = st.number_input("Capacité par shift", min_value=50, max_value=1000, value=200)
        alpha_rework = st.slider("Taux de récupération rework", 0.0, 1.0, 0.8, 0.1)
    
    with col3:
        beta = st.slider("Facteur capacité rework", 1.0, 2.0, 1.2, 0.1)
        penalite = st.number_input("Pénalité pénurie", min_value=100, max_value=10000, value=1000)
    
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
    if st.button("🚀 Lancer la Planification", use_container_width=True):
        with st.spinner("Optimisation en cours..."):
            # Configuration du système de planification
            success_setup = system.setup_planning_system(
                S=S, T=T,
                mean_capacity=capacity,
                alpha_rework=alpha_rework,
                beta=beta,
                penalite_penurie=penalite,
                demandes_personnalisees=demandes_personnalisees
            )
            
            if success_setup:
                # Exécution de la planification
                success_planning = system.run_integrated_planning(time_limit=300)
                
                if success_planning:
                    # Analyse des résultats
                    results = system.analyze_integrated_results()
                    
                    if results:
                        st.success("✅ Planification réussie!")
                        
                        # Affichage immédiat des scénarios
                        display_scenario_details(system)
                        
                        return results
                    else:
                        st.error("❌ Erreur lors de l'analyse des résultats")
                else:
                    st.error("❌ Échec de la planification - Pas de solution optimale")
            else:
                st.error("❌ Erreur de configuration")
    
    return None

def display_scenario_details(system):
    """Affiche les détails de tous les scénarios avec positions et quantités"""
    st.subheader("🎯 Détails des Scénarios de Planification")
    
    if not hasattr(system.planner, 'scenario_analysis') or not system.planner.scenario_analysis:
        st.error("❌ Aucune analyse de scénario disponible")
        return
    
    scenario_analysis = system.planner.scenario_analysis
    params = system.planner.parameters
    
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
                st.metric("Total Pénuries", f"{kpis['total_penuries']:.1f}")
            with col4:
                st.metric("Coût Estimé", f"{kpis['cout_estime']:,.0f}")
            
            st.markdown("---")
            
            # Détails par shift
            st.write("### 📋 Plan d'Exécution par Shift")
            
            for t in range(params['T']):
                shift_info = scenario_data['shifts_details'][t+1]
                
                st.write(f"#### 🔄 Shift {t+1}")
                
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
                                # Calculer la production utile
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
            
            # Graphique de la production pour ce scénario
            st.write("### 📈 Visualisation de la Production")
            
            # Graphique en barres des quantités par référence et shift
            plot_data = []
            for t in range(params['T']):
                shift_info = scenario_data['shifts_details'][t+1]
                for ref, qty in shift_info['quantities'].items():
                    if qty > 0:
                        plot_data.append({
                            'Shift': f'Shift {t+1}',
                            'Référence': ref,
                            'Quantité': qty
                        })
            
            if plot_data:
                df_plot = pd.DataFrame(plot_data)
                fig = px.bar(df_plot, 
                           x='Shift', 
                           y='Quantité', 
                           color='Référence',
                           title=f"Production par Shift - Scénario {s+1}",
                           text='Quantité')
                fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                fig.update_layout(showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune production programmée pour ce scénario")
    
    # Comparaison rapide des scénarios
    st.subheader("⚖️ Comparaison Rapide des Scénarios")
    
    comparison_data = []
    for s, scenario_data in scenario_analysis.items():
        kpis = scenario_data['kpis']
        comparison_data.append({
            'Scénario': f'S{s+1}',
            'Satisfaction (%)': f"{kpis['satisfaction_globale']:.1f}",
            'Utilisation (%)': f"{kpis['utilisation_capacite']:.1f}",
            'Pénuries': f"{kpis['total_penuries']:.0f}",
            'Coût': f"{kpis['cout_estime']:,.0f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, hide_index=True, use_container_width=True)

def dashboard_section(system, results):
    st.header("📊 Dashboard Comparatif")
    
    if results is None:
        st.warning("⚠️ Aucun résultat de planification disponible")
        return
    
    if not system.planner or not hasattr(system.planner, 'scenario_analysis'):
        st.error("❌ Données d'analyse manquantes")
        return
    
    scenario_analysis = system.planner.scenario_analysis
    
    # Vue d'ensemble
    st.subheader("🎯 Vue d'Ensemble")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Taux Rework Utilisé",
            f"{results['predicted_rework_rate']:.2f}%"
        )
    
    with col2:
        st.metric(
            "Coût Total Optimal",
            f"{results['planning_results']['cout_total']:,.0f}"
        )
    
    with col3:
        st.metric(
            "Scénarios Analysés",
            len(scenario_analysis)
        )
    
    with col4:
        params = system.planner.parameters
        demande_totale = sum(params['EDI'].values())
        st.metric(
            "Demande Totale",
            f"{demande_totale:,.0f}"
        )
    
    # Comparaison des scénarios
    st.subheader("📈 Comparaison des Scénarios")
    
    # Préparer les données pour le graphique
    scenarios_data = []
    for s, data in scenario_analysis.items():
        kpis = data['kpis']
        scenarios_data.append({
            'Scénario': f'S{s+1}',
            'Satisfaction (%)': kpis['satisfaction_globale'],
            'Utilisation Capacité (%)': kpis['utilisation_capacite'],
            'Total Pénuries': kpis['total_penuries'],
            'Coût Estimé': kpis['cout_estime'],
            'Efficacité': kpis['efficacite_production']
        })
    
    df_scenarios = pd.DataFrame(scenarios_data)
    
    # Graphiques comparatifs
    col1, col2 = st.columns(2)
    
    with col1:
        fig_satisfaction = px.bar(
            df_scenarios,
            x='Scénario',
            y='Satisfaction (%)',
            title="Taux de Satisfaction par Scénario",
            color='Satisfaction (%)',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_satisfaction, use_container_width=True)
    
    with col2:
        fig_capacite = px.bar(
            df_scenarios,
            x='Scénario',
            y='Utilisation Capacité (%)',
            title="Utilisation de la Capacité par Scénario",
            color='Utilisation Capacité (%)',
            color_continuous_scale='plasma'
        )
        st.plotly_chart(fig_capacite, use_container_width=True)
    
    # Graphique radar comparatif
    st.subheader("🕸️ Analyse Radar des Performances")
    
    # Normaliser les données pour le radar
    metrics = ['Satisfaction (%)', 'Utilisation Capacité (%)', 'Efficacité']
    fig_radar = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, row in df_scenarios.iterrows():
        # Normaliser l'efficacité sur 100
        efficacite_norm = min(100, row['Efficacité'] * 100)
        penuries_norm = max(0, 100 - (row['Total Pénuries'] / demande_totale * 100))
        
        values = [
            row['Satisfaction (%)'],
            row['Utilisation Capacité (%)'],
            efficacite_norm,
            penuries_norm
        ]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Fermer le polygone
            theta=['Satisfaction', 'Utilisation Capacité', 'Efficacité', 'Anti-Pénuries'] + ['Satisfaction'],
            fill='toself',
            name=row['Scénario'],
            line_color=colors[i % len(colors)]
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Comparaison Multi-Critères des Scénarios"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Sélection et détail du meilleur scénario
    st.subheader("🏆 Meilleur Scénario")
    
    # Calcul du score global
    scores_globaux = {}
    for i, row in df_scenarios.iterrows():
        satisfaction_norm = row['Satisfaction (%)'] / 100
        utilisation_norm = row['Utilisation Capacité (%)'] / 100
        penuries_norm = 1 - (row['Total Pénuries'] / demande_totale)
        cout_norm = 1 - (row['Coût Estimé'] / df_scenarios['Coût Estimé'].max())
        efficacite_norm = row['Efficacité'] / df_scenarios['Efficacité'].max()
        
        score_global = (
            0.30 * satisfaction_norm +
            0.20 * utilisation_norm +
            0.25 * penuries_norm +
            0.15 * cout_norm +
            0.10 * efficacite_norm
        )
        
        scores_globaux[row['Scénario']] = score_global
    
    best_scenario_name = max(scores_globaux, key=scores_globaux.get)
    best_score = scores_globaux[best_scenario_name]
    best_scenario_id = int(best_scenario_name.replace('S', '')) - 1
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(
            "Meilleur Scénario",
            best_scenario_name,
            f"Score: {best_score:.3f}"
        )
        
        best_data = scenario_analysis[best_scenario_id]
        best_kpis = best_data['kpis']
        
        st.write("**Performances:**")
        st.write(f"• Satisfaction: {best_kpis['satisfaction_globale']:.1f}%")
        st.write(f"• Utilisation: {best_kpis['utilisation_capacite']:.1f}%")
        st.write(f"• Pénuries: {best_kpis['total_penuries']:.1f}")
        st.write(f"• Coût: {best_kpis['cout_estime']:,.0f}")
    
    with col2:
        st.write("**Plan d'Exécution Recommandé:**")
        
        for t in range(system.planner.parameters['T']):
            shift_info = best_data['shifts_details'][t+1]
            ordre = ' → '.join([ref for ref in shift_info['execution_order'] if ref != 'VIDE'])
            
            st.write(f"**Shift {t+1}:** {ordre}")
            
            if shift_info['quantities']:
                for ref, qty in shift_info['quantities'].items():
                    if qty > 0:
                        st.write(f"  • {ref}: {int(qty)} unités")
    
    # Tableau détaillé
    st.subheader("📋 Tableau Détaillé des Scénarios")
    
    # Ajouter les scores au dataframe
    df_scenarios['Score Global'] = [scores_globaux[scenario] for scenario in df_scenarios['Scénario']]
    df_scenarios = df_scenarios.sort_values('Score Global', ascending=False)
    
    st.dataframe(
        df_scenarios,
        use_container_width=True,
        hide_index=True
    )

def export_section(system, results):
    st.header("📁 Export des Résultats")
    
    if results is None:
        st.warning("⚠️ Aucun résultat à exporter")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Exporter vers Excel", use_container_width=True):
            try:
                # Créer un buffer en mémoire
                output = io.BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Résumé intégré
                    summary_data = [{
                        'Taux_Rework_Predit_Pct': results['predicted_rework_rate'],
                        'Cout_Total_Optimal': results['planning_results']['cout_total'],
                        'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    }]
                    
                    df_summary = pd.DataFrame(summary_data)
                    df_summary.to_excel(writer, sheet_name='Resume_Integration', index=False)
                    
                    # Comparaison des scénarios
                    if hasattr(system.planner, 'scenario_analysis'):
                        planning_comparison = []
                        for s, data in system.planner.scenario_analysis.items():
                            kpis = data['kpis']
                            planning_comparison.append({
                                'Scenario': f'S{s+1}',
                                'Satisfaction_Pct': kpis['satisfaction_globale'],
                                'Utilisation_Capacite_Pct': kpis['utilisation_capacite'],
                                'Total_Penuries': kpis['total_penuries'],
                                'Cout_Estime': kpis['cout_estime'],
                                'Efficacite_Production': kpis['efficacite_production']
                            })
                        
                        df_planning = pd.DataFrame(planning_comparison)
                        df_planning.to_excel(writer, sheet_name='Comparaison_Scenarios', index=False)
                
                # Préparer le téléchargement
                processed_data = output.getvalue()
                
                st.download_button(
                    label="⬇️ Télécharger Excel",
                    data=processed_data,
                    file_name=f"resultats_integres_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Fichier Excel généré!")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'export: {e}")
    
    with col2:
        if st.button("📊 Exporter Rapport PDF", use_container_width=True):
            st.info("🚧 Fonctionnalité en développement")

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
                "📋 4. Planification",
                "📈 5. Dashboard",
                "📁 6. Export"
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
        **Système Intégré v1.0**
        
        🔮 **Prédiction:** Modèles ML pour prédire les défauts
        
        📋 **Planification:** Optimisation stochastique avec contraintes
        
        📊 **Dashboard:** Comparaison multi-critères des scénarios
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
    
    elif step == "📋 4. Planification":
        if st.session_state.prediction_result is not None:
            planning_results = planning_section(st.session_state.system, st.session_state.prediction_result)
            if planning_results is not None:
                st.session_state.planning_results = planning_results
        else:
            st.warning("⚠️ Veuillez d'abord effectuer une prédiction à l'étape 3")
    
    elif step == "📈 5. Dashboard":
        if st.session_state.planning_results is not None:
            dashboard_section(st.session_state.system, st.session_state.planning_results)
        else:
            st.warning("⚠️ Veuillez d'abord effectuer la planification à l'étape 4")
    
    elif step == "📁 6. Export":
        if st.session_state.planning_results is not None:
            export_section(st.session_state.system, st.session_state.planning_results)
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
        🏭 Système Intégré Prédiction-Planification | 
        Développé avec ❤️ en Streamlit | 
        © 2024
    </div>
    """, unsafe_allow_html=True)

# Point d'entrée
if __name__ == "__main__":
    main()
