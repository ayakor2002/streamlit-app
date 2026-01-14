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

# Streamlit page configuration
st.set_page_config(
    page_title="Advanced Integrated Prediction-Planning System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        """Identifies workstations while STRICTLY PRESERVING original column names"""
        postes_cols = []
        
        defaut_keywords = ['defauts', 'defaut', 'defect', 'rework', 'echec', 'fail']
        
        for col in data.columns:
            if any(keyword in col.lower() for keyword in defaut_keywords):
                postes_cols.append(col)
        
        if not postes_cols:
            excluded_keywords = ['jour', 'volume', 'production', 'date', 'time', 'day', 'week']
            for col in data.columns:
                is_excluded = any(keyword in col.lower() for keyword in excluded_keywords)
                if not is_excluded and pd.api.types.is_numeric_dtype(data[col]):
                    postes_cols.append(col)
        
        if not postes_cols:
            raise ValueError("No defect columns identified! Please verify your data contains numeric columns representing defects by workstation.")
        
        self.postes = postes_cols
        st.info(f"📍 Identified workstations (names preserved): {postes_cols}")
        return postes_cols

    def prepare_data_for_poste(self, data, poste_col):
        """Prepares data while STRICTLY PRESERVING original column names"""
        data_copy = data.copy()

        jour_col = None
        volume_col = None
        
        jour_keywords = ['jour', 'day', 'date', 'semaine', 'week']
        for col in data.columns:
            if any(keyword in col.lower() for keyword in jour_keywords):
                jour_col = col
                break
        
        volume_keywords = ['volume', 'production', 'quantite', 'qty']
        for col in data.columns:
            if any(keyword in col.lower() for keyword in volume_keywords):
                volume_col = col
                break

        if not jour_col or not volume_col:
            available_cols = list(data.columns)
            raise ValueError(f"Columns 'jour' and 'volume' required! Available columns: {available_cols}")

        self.jour_col = jour_col
        self.volume_col = volume_col

        try:
            if data_copy[jour_col].dtype == 'object':
                try:
                    data_copy[jour_col] = pd.to_datetime(data_copy[jour_col]).dt.dayofweek + 1
                except:
                    try:
                        data_copy[jour_col] = pd.to_numeric(data_copy[jour_col], errors='coerce')
                    except:
                        pass
        except Exception as e:
            st.warning(f"Warning: Issue with day column ({jour_col}): {e}")

        if poste_col not in data_copy.columns:
            raise ValueError(f"Workstation column '{poste_col}' not found in data!")

        X = data_copy[[volume_col, jour_col]].copy()
        y = data_copy[poste_col].copy()
        
        mask = (~X.isnull().any(axis=1)) & (~y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            raise ValueError("No valid data after cleaning!")

        return X, y, jour_col, volume_col

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
        all_results = {}

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
            y_pred_test = search.predict(X_test)
            mse = mean_squared_error(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            
            all_results[name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'best_params': search.best_params_,
                'cv_score': search.best_score_
            }

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
            'best_params': best_params,
            'all_results': all_results
        }

    def train_all_postes(self, data, search_method='grid'):
        postes = self.identify_postes(data)

        if not postes:
            raise ValueError("No workstations identified in data!")

        results = {}
        for poste in postes:
            X, y, jour_col, volume_col = self.prepare_data_for_poste(data, poste)
            results[poste] = self.train_model_for_poste(X, y, poste, jour_col, volume_col, search_method=search_method)

        self.set_poste_weights()
        return results, postes

    def predict_single_scenario(self, jour, volume):
        if not self.models:
            raise ValueError("Models must be trained before making predictions!")

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
    Advanced stochastic planning model with multi-criteria analysis
    Adapted for integration with defect prediction
    """

    def __init__(self):
        """Model initialization"""
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
                      poids_cout: float = 0.25,
                      poids_satisfaction: float = 0.30,
                      poids_utilisation: float = 0.20,
                      poids_stabilite: float = 0.15,
                      poids_penuries: float = 0.10,
                      use_predicted_rework: bool = False,
                      predicted_rework_rate: float = None):
        """
        Complete model configuration with customizable stochastic parameters
        """

        if R is None:
            R = [f'REF_{i+1:02d}' for i in range(10)]

        if EDI is None:
            EDI = [20, 35, 45, 25, 40, 50, 22, 38, 30, 42]

        if isinstance(EDI, list):
            EDI_dict = {R[i]: EDI[i] for i in range(min(len(R), len(EDI)))}
        else:
            EDI_dict = EDI

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

        np.random.seed(42)
        
        CAPchaine = {}
        for s in range(S):
            for t in range(T):
                capacite = max(50, np.random.normal(mean_capacity, std_capacity))
                CAPchaine[(s, t)] = capacite
        
        taux_defaut = {}
        
        if use_predicted_rework and predicted_rework_rate is not None:
            base_rate = predicted_rework_rate / 100
            self.predicted_rework_rate = predicted_rework_rate
            
            for s in range(S):
                for i in R:
                    defaut = max(0.001, min(0.25, np.random.normal(base_rate, std_defaut)))
                    taux_defaut[(s, i)] = defaut
        else:
            for s in range(S):
                for i in R:
                    defaut = max(0.001, min(0.25, np.random.normal(mean_defaut, std_defaut)))
                    taux_defaut[(s, i)] = defaut

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
        """Create optimization model"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']

        self.model = plp.LpProblem("Complete_Stochastic_Planning", plp.LpMinimize)

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
        """Add model constraints"""
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
                    f"Demand_s{s}_i{i}"
                )

        for s in range(S):
            for t in range(T):
                capacite_utilisee = plp.lpSum([
                    q[(s, i, t)] * (1 + params['beta'] * params['taux_defaut'][(s, i)])
                    for i in R
                ])
                self.model += (
                    capacite_utilisee <= params['CAPchaine'][(s, t)],
                    f"Capacity_s{s}_t{t}"
                )

        for s in range(S):
            for t in range(T):
                for j in range(len(R)):
                    self.model += (
                        plp.lpSum([x[(i, j, s, t)] for i in R]) == 1,
                        f"Position_s{s}_t{t}_j{j}"
                    )
                
                for i in R:
                    self.model += (
                        plp.lpSum([x[(i, j, s, t)] for j in range(len(R))]) <= 1,
                        f"Reference_s{s}_t{t}_i{i}"
                    )

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
        """Define objective function"""
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
        """Solve the model"""
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
            st.error(f"Error during resolution: {e}")
            return False

    def _extract_results(self):
        """Extract solution results"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']

        production_results = {}
        for s in range(S):
            for i in R:
                for t in range(T):
                    key = (s, i, t)
                    value = self.variables['q'][key].value()
                    production_results[key] = value if value is not None else 0

        penuries_results = {}
        for s in range(S):
            for i in R:
                key = (s, i)
                value = self.variables['penurie'][key].value()
                penuries_results[key] = value if value is not None else 0

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
        """Detailed scenario analysis for Streamlit"""
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

            total_capacity_used = 0
            total_capacity_available = 0

            for t in range(T):
                shift_info = {
                    'execution_order': [],
                    'quantities': {},
                    'capacity_used': 0,
                    'capacity_available': params['CAPchaine'][(s, t)]
                }

                sequence = sequencement.get((s, t), {})
                ordered_refs = []
                for j in range(len(R)):
                    ref = sequence.get(j, 'EMPTY')
                    ordered_refs.append(ref)
                shift_info['execution_order'] = ordered_refs

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

            total_production_utile = 0
            total_demande = sum(params['EDI'].values())
            total_penuries = 0

            for i in R:
                total_prod = sum(production[(s, i, t)] for t in range(T))
                
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

            cout_production = sum([production[(s, i, t)] * 20 for i in R for t in range(T)])
            cout_penuries = total_penuries * params['penalite_penurie']
            cout_total = cout_production + cout_penuries

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
        """Calculate multi-criteria scores"""
        if not self.scenario_analysis:
            return

        params = self.parameters
        S = params['S']

        criteria_values = {
            'cout': [self.scenario_analysis[s]['kpis']['cout_total'] for s in range(S)],
            'satisfaction': [self.scenario_analysis[s]['kpis']['satisfaction_globale'] for s in range(S)],
            'utilisation': [self.scenario_analysis[s]['kpis']['utilisation_capacite'] for s in range(S)],
            'stabilite': [self.scenario_analysis[s]['kpis']['stabilite'] for s in range(S)],
            'penuries': [self.scenario_analysis[s]['kpis']['total_penuries'] for s in range(S)]
        }

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
        """Select best scenario"""
        if not self.multicriteria_scores:
            return None

        global_scores = self.multicriteria_scores['global_scores']
        
        best_scenario_id = max(global_scores, key=global_scores.get)
        best_score = global_scores[best_scenario_id]

        sorted_scenarios = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)
        
        scores_values = list(global_scores.values())
        gap_with_second = sorted_scenarios[0][1] - sorted_scenarios[1][1] if len(sorted_scenarios) > 1 else 0
        robustness = "HIGH" if gap_with_second > 0.1 else "MODERATE" if gap_with_second > 0.05 else "LOW"

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
        self.prediction_training_results = results
        return results, postes

    def make_prediction_for_planning(self, jour, volume, method='moyenne_ponderee'):
        if self.predictor is None:
            raise ValueError("Prediction system must be configured first!")

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
            raise ValueError("No predicted rework rate available!")

        self.planner = StochasticPlanningModelComplete()
        
        if demandes_personnalisees is not None:
            params['EDI'] = demandes_personnalisees

        if params.get('use_predicted_rework', True):
            params['predicted_rework_rate'] = predicted_rework_rate

        self.planner.set_parameters(**params)
        return True

    def run_integrated_planning(self, time_limit=300):
        if self.planner is None:
            raise ValueError("Planning system must be configured first!")

        try:
            self.planner.create_model()
            self.planner.add_constraints()
            self.planner.set_objective()
            return self.planner.solve_model(time_limit=time_limit)
        except Exception as e:
            st.error(f"Error during execution: {e}")
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

def create_header():
    """Create header"""
    
    st.markdown("""
    <div class="main-header">
        <h1>🏭 Advanced Integrated Prediction-Planning System</h1>
        <p>Defect prediction and multi-criteria stochastic planning</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

def load_data_section():
    """Load data section"""
    st.header("📊 Load Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your Excel file containing historical data",
            type=['xlsx', 'xls'],
            help="File must contain columns: Day, Volume_production, and defect columns by workstation"
        )
    
    with col2:
        if st.button("📝 Use Demo Data", use_container_width=True):
            with st.spinner("Generating demo data..."):
                demo_data = create_demo_data()
                st.success(f"✅ Demo data generated: {len(demo_data)} rows")
                
                display_demo_data(demo_data)
                
                return demo_data
    
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.success(f"✅ Data loaded: {len(data)} rows, {len(data.columns)} columns")
            
            with st.expander("👀 Preview raw data"):
                st.dataframe(data.head())
                st.write("**Available columns:**", list(data.columns))
            
            display_demo_data(data)
            
            return data
            
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            return None
    
    return None

def display_demo_data(data):
    """Display data with statistics and visualizations"""
    
    is_demo = set(['Jour', 'Volume_production', 'Poste1_defauts', 'Poste2_defauts', 'Poste3_defauts']).issubset(set(data.columns))
    
    if is_demo:
        st.subheader("📊 Demo Data Generated")
    else:
        st.subheader("📊 Analysis of Uploaded Data")
    
    jour_col = None
    volume_col = None
    defaut_cols = []
    
    jour_keywords = ['jour', 'day', 'date', 'semaine', 'week']
    for col in data.columns:
        if any(keyword in col.lower() for keyword in jour_keywords):
            jour_col = col
            break
    
    volume_keywords = ['volume', 'production', 'quantite', 'qty']
    for col in data.columns:
        if any(keyword in col.lower() for keyword in volume_keywords):
            volume_col = col
            break
    
    defaut_keywords = ['defauts', 'defaut', 'defect', 'rework', 'echec', 'fail']
    for col in data.columns:
        if any(keyword in col.lower() for keyword in defaut_keywords):
            defaut_cols.append(col)
    
    if not defaut_cols:
        excluded_keywords = ['jour', 'volume', 'production', 'date', 'time', 'day', 'week']
        for col in data.columns:
            is_excluded = any(keyword in col.lower() for keyword in excluded_keywords)
            if not is_excluded and pd.api.types.is_numeric_dtype(data[col]):
                defaut_cols.append(col)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of rows", len(data))
    
    with col2:
        if volume_col:
            volume_moyen = data[volume_col].mean()
            st.metric(f"Average volume\n({volume_col})", f"{volume_moyen:.0f}")
        else:
            st.metric("Average volume", "Not found", help="No volume column identified")
    
    with col3:
        if defaut_cols:
            defauts_total = data[defaut_cols].sum().sum()
            st.metric(f"Total defects\n({len(defaut_cols)} stations)", f"{defauts_total:.0f}")
        else:
            st.metric("Total defects", "Not found", help="No defect columns identified")
    
    with col4:
        if defaut_cols and volume_col:
            taux_defaut_moyen = (defauts_total / data[volume_col].sum()) * 100
            st.metric("Avg defect rate", f"{taux_defaut_moyen:.2f}%")
        else:
            st.metric("Avg defect rate", "Not calculable")
    
    st.subheader("🔍 Identified Columns")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if jour_col:
            st.success(f"✅ **Day:** {jour_col}")
        else:
            st.error("❌ **Day:** Not found")
    
    with col2:
        if volume_col:
            st.success(f"✅ **Volume:** {volume_col}")
        else:
            st.error("❌ **Volume:** Not found")
    
    with col3:
        if defaut_cols:
            st.success(f"✅ **Defects:** {len(defaut_cols)} columns")
            st.write(f"Workstations: {', '.join(defaut_cols)}")
        else:
            st.error("❌ **Defects:** Not found")
    
    with st.expander("👀 Data Preview", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**First rows of data:**")
            st.dataframe(data.head(10), use_container_width=True)
        
        with col2:
            st.write("**Descriptive statistics:**")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats_df = data[numeric_cols].describe().round(2)
                st.dataframe(stats_df)
            else:
                st.write("No numeric columns found")
    
    if defaut_cols:
        st.subheader("🏷️ Preserved Workstation Names")
        st.info(f"📋 The following names will be used exactly as provided: **{', '.join(defaut_cols)}**")

def create_demo_data(n_days=100):
    """Create demo data with realistic values"""
    np.random.seed(42)
    days = range(1, n_days + 1)
    data = []

    for day in days:
        jour_semaine = ((day - 1) % 7) + 1
        
        if jour_semaine in [6, 7]:
            volume_base = 800
        else:
            volume_base = 1200

        volume = volume_base + np.random.normal(0, 100)
        volume = max(volume, 100)

        poste1_defauts = volume * 0.02 + jour_semaine * 0.5 + np.random.normal(0, 2)
        poste2_defauts = volume * 0.015 + jour_semaine * 0.3 + np.random.normal(0, 1.5)
        poste3_defauts = volume * 0.025 + jour_semaine * 0.4 + np.random.normal(0, 2.5)

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
    """Prediction section"""
    st.header("🔮 Defect Prediction")
    
    if data is None:
        st.warning("⚠️ Please load data first")
        return None
    
    try:
        with st.spinner("🧠 Training prediction models..."):
            results, postes = system.setup_prediction_system(data)
        
        if results and postes:
            st.success("✅ Prediction models trained successfully!")
            
            st.subheader("🏆 Optimal Models Selected")
            
            model_selection_data = []
            for poste_nom_original in postes:
                if poste_nom_original in results:
                    result = results[poste_nom_original]
                    model_selection_data.append({
                        'Workstation': poste_nom_original,
                        'Optimal Model': result['model_name'],
                        'MSE': f"{result['mse']:.4f}",
                        'MAE': f"{result['mae']:.4f}",
                        'R²': f"{result['r2']:.4f}",
                        'Optimal Parameters': str(result['best_params'])
                    })
            
            df_models = pd.DataFrame(model_selection_data)
            st.dataframe(df_models, use_container_width=True)
            
            with st.expander("📊 Model Selection Details", expanded=False):
                for poste_nom_original in postes:
                    if poste_nom_original in results and 'all_results' in results[poste_nom_original]:
                        st.write(f"### 🔍 {poste_nom_original}")
                        
                        all_results = results[poste_nom_original]['all_results']
                        comparison_data = []
                        
                        for model_name, model_result in all_results.items():
                            comparison_data.append({
                                'Model': model_name,
                                'MSE': f"{model_result['mse']:.4f}",
                                'MAE': f"{model_result['mae']:.4f}",
                                'R²': f"{model_result['r2']:.4f}",
                                'CV Score': f"{model_result['cv_score']:.4f}",
                                'Best': "🏆" if model_name == results[poste_nom_original]['model_name'] else ""
                            })
                        
                        df_comparison = pd.DataFrame(comparison_data)
                        st.dataframe(df_comparison, use_container_width=True)
                        
                        fig = px.bar(
                            df_comparison, 
                            x='Model', 
                            y='R²',
                            title=f'R² Comparison for {poste_nom_original}',
                            color='R²',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Chain Rework Rate")
            
            postes_originaux = system.predictor.postes
            taux_historiques = {}
            
            for poste_nom_original in postes_originaux:
                if poste_nom_original in data.columns and system.predictor.volume_col in data.columns:
                    defauts_totaux = data[poste_nom_original].sum()
                    volume_total = data[system.predictor.volume_col].sum()
                    taux_historiques[poste_nom_original] = (defauts_totaux / volume_total) * 100 if volume_total > 0 else 0
            
            taux_pondere_historique = 0
            if system.predictor.poste_weights:
                for poste_nom_original, poids in system.predictor.poste_weights.items():
                    if poste_nom_original in taux_historiques:
                        taux_pondere_historique += taux_historiques[poste_nom_original] * poids
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Historical Rework Rate (Weighted Avg)",
                    f"{taux_pondere_historique:.2f}%",
                    help="Based on historical data with Q*D weighting"
                )
            
            with col2:
                taux_moyen_simple = np.mean(list(taux_historiques.values())) if taux_historiques else 0
                st.metric(
                    "Historical Rework Rate (Simple Avg)",
                    f"{taux_moyen_simple:.2f}%"
                )
            
            with col3:
                st.metric("Number of Workstations", len(postes_originaux))
            
            st.subheader("🏷️ Confirmation of Names")
            st.success(f"✅ **Preserved workstation names:** {', '.join(postes_originaux)}")
            
            return True
        else:
            st.error("❌ Model training failed")
            return False
            
    except Exception as e:
        st.error(f"❌ Error during training: {e}")
        st.write("**Error details:**")
        st.code(str(e))
        
        st.write("**Debug information:**")
        st.write(f"- Available columns: {list(data.columns)}")
        st.write(f"- Number of rows: {len(data)}")
        st.write(f"- Column types: {dict(data.dtypes)}")
        
        return False

def new_prediction_section(system):
    st.header("🎯 New Prediction")
    
    if system.predictor is None:
        st.warning("⚠️ Please configure prediction system first")
        return None
    
    col1, col2 = st.columns(2)
    
    with col1:
        jour = st.selectbox(
            "Day of week",
            options=[1, 2, 3, 4, 5, 6, 7],
            format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x-1],
            index=2
        )
    
    with col2:
        volume = st.number_input(
            "Expected production volume",
            min_value=1,
            max_value=10000,
            value=1200,
            step=50
        )
    
    if st.button("🔮 Make Prediction", use_container_width=True):
        with st.spinner("Computing..."):
            prediction_result = system.make_prediction_for_planning(jour, volume)
        
        st.subheader("📊 Prediction Results")
        
        pred_details = prediction_result['prediction_details']
        taux_rework_nouveau = prediction_result['rework_rate_for_planning']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "New Chain Rework Rate",
                f"{taux_rework_nouveau:.2f}%",
                help="Calculated with weighted average"
            )
        
        with col2:
            defauts_predits = pred_details['predictions_chaine']['moyenne_ponderee']
            st.metric("Predicted Defects", f"{defauts_predits:.1f}")
        
        with col3:
            st.metric("Analyzed Volume", f"{volume:,.0f}")
        
        with col4:
            jour_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][jour-1]
            st.metric("Analyzed Day", jour_name)
        
        st.subheader("🔍 Prediction Details by Workstation")
        
        prediction_details = []
        for poste_nom_original, defauts_pred in pred_details['predictions_postes'].items():
            taux_rework_poste = pred_details['taux_rework_postes'][poste_nom_original]
            
            model_name = system.predictor.best_model_names.get(poste_nom_original, "Unknown")
            
            importance_info = ""
            if poste_nom_original in system.predictor.feature_importances:
                feat_imp = system.predictor.feature_importances[poste_nom_original]
                importance_info = f"Volume: {feat_imp.iloc[0]['importance']:.3f}, Day: {feat_imp.iloc[1]['importance']:.3f}"
            
            prediction_details.append({
                'Workstation': poste_nom_original,
                'Optimal Model': model_name,
                'Predicted Defects': f"{defauts_pred:.1f}",
                'Rework Rate': f"{taux_rework_poste:.2f}%",
                'Feature Importance': importance_info if importance_info else "N/A"
            })
        
        df_predictions = pd.DataFrame(prediction_details)
        st.dataframe(df_predictions, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            postes_originaux = list(pred_details['predictions_postes'].keys())
            defauts_values = list(pred_details['predictions_postes'].values())
            
            fig_defauts = px.bar(
                x=postes_originaux,
                y=defauts_values,
                title="Predicted Defects by Workstation",
                labels={'x': 'Workstations', 'y': 'Number of Defects'},
                color=defauts_values,
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig_defauts, use_container_width=True)
        
        with col2:
            taux_values = list(pred_details['taux_rework_postes'].values())
            
            fig_taux = px.bar(
                x=postes_originaux,
                y=taux_values,
                title="Rework Rate by Workstation (%)",
                labels={'x': 'Workstations', 'y': 'Rework Rate (%)'},
                color=taux_values,
                color_continuous_scale='oranges'
            )
            st.plotly_chart(fig_taux, use_container_width=True)
        
        st.subheader("⚖️ Comparison of Aggregation Methods")
        
        aggregation_data = []
        for method, taux in pred_details['taux_rework_chaine'].items():
            defauts = pred_details['predictions_chaine'][method]
            
            method_names = {
                'max': 'Maximum',
                'moyenne': 'Simple Average',
                'moyenne_ponderee': 'Weighted Average (Recommended)',
                'somme': 'Sum'
            }
            
            aggregation_data.append({
                'Method': method_names.get(method, method),
                'Total Defects': f"{defauts:.1f}",
                'Chain Rework Rate': f"{taux:.2f}%",
                'Recommended': "✅" if method == 'moyenne_ponderee' else ""
            })
        
        df_aggregation = pd.DataFrame(aggregation_data)
        st.dataframe(df_aggregation, use_container_width=True)
        
        if system.predictor.poste_weights:
            st.subheader("⚖️ Weights Used for Weighted Average")
            
            weights_data = []
            for poste_nom_original, poids in system.predictor.poste_weights.items():
                weights_data.append({
                    'Workstation': poste_nom_original,
                    'Weight': f"{poids:.4f}",
                    'Percentage': f"{poids*100:.2f}%"
                })
            
            df_weights = pd.DataFrame(weights_data)
            st.dataframe(df_weights, use_container_width=True)
            
            fig_weights = px.pie(
                df_weights,
                values='Weight',
                names='Workstation',
                title="Weight Distribution by Workstation"
            )
            st.plotly_chart(fig_weights, use_container_width=True)
        
        return prediction_result
    
    return None

def planning_section(system, prediction_result):
    st.header("📋 Advanced Stochastic Planning")
    
    if prediction_result is None:
        st.warning("⚠️ Please make a prediction first")
        return None
    
    st.subheader("⚙️ Planning Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        S = st.number_input("Number of scenarios", min_value=1, max_value=10, value=5)
        T = st.number_input("Number of shifts", min_value=1, max_value=5, value=3)
    
    with col2:
        mean_capacity = st.number_input("Average capacity per shift", min_value=50, max_value=1000, value=160)
        alpha_rework = st.slider("Rework recovery rate", 0.0, 1.0, 0.8, 0.1)
    
    with col3:
        beta = st.slider("Rework capacity factor", 1.0, 2.0, 1.2, 0.1)
        penalite = st.number_input("Shortage penalty", min_value=100, max_value=10000, value=1000)
    
    st.subheader("🎲 Stochastic Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Capacity Variations**")
        std_capacity = st.slider(
            "Capacity standard deviation", 
            min_value=5.0, 
            max_value=50.0, 
            value=10.0, 
            step=1.0,
            help="Stochastic variation of capacity between scenarios"
        )
        
        use_predicted_rework = st.checkbox(
            "Use predicted rework rate", 
            value=True,
            help="Use predicted rate or generate stochastically"
        )
        
    with col2:
        st.write("**🔄 Rework Rate Variations**")
        
        if not use_predicted_rework:
            mean_defaut = st.slider(
                "Average defect rate", 
                min_value=0.01, 
                max_value=0.20, 
                value=0.04, 
                step=0.01,
                help="Average defect rate for stochastic generation"
            )
            
            std_defaut = st.slider(
                "Defect standard deviation", 
                min_value=0.001, 
                max_value=0.05, 
                value=0.01, 
                step=0.001,
                help="Stochastic variation of defect rate"
            )
        else:
            rework_variation = st.slider(
                "Variation around predicted rate (%)", 
                min_value=0.0, 
                max_value=50.0, 
                value=10.0, 
                step=5.0,
                help="Percentage variation around predicted rate"
            )
            
            mean_defaut = prediction_result['rework_rate_for_planning'] / 100
            std_defaut = mean_defaut * (rework_variation / 100)
    
    with st.expander("📈 Preview of Stochastic Distributions"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Capacity:**")
            st.write(f"- Average: {mean_capacity}")
            st.write(f"- Standard deviation: {std_capacity}")
            st.write(f"- Expected range: [{mean_capacity-2*std_capacity:.0f}, {mean_capacity+2*std_capacity:.0f}]")
        
        with col2:
            st.write("**Defect rate:**")
            st.write(f"- Average: {mean_defaut*100:.2f}%")
            st.write(f"- Standard deviation: {std_defaut*100:.2f}%")
            st.write(f"- Expected range: [{max(0,(mean_defaut-2*std_defaut)*100):.2f}%, {min(25,(mean_defaut+2*std_defaut)*100):.2f}%]")
    
    st.subheader("⚖️ Multi-criteria Weights")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        poids_cout = st.slider("Cost Weight", 0.0, 1.0, 0.25, 0.05)
    with col2:
        poids_satisfaction = st.slider("Satisfaction Weight", 0.0, 1.0, 0.30, 0.05)
    with col3:
        poids_utilisation = st.slider("Utilization Weight", 0.0, 1.0, 0.20, 0.05)
    with col4:
        poids_stabilite = st.slider("Stability Weight", 0.0, 1.0, 0.15, 0.05)
    with col5:
        poids_penuries = st.slider("Shortage Weight", 0.0, 1.0, 0.10, 0.05)
    
    total_poids = poids_cout + poids_satisfaction + poids_utilisation + poids_stabilite + poids_penuries
    if total_poids > 0:
        poids_cout /= total_poids
        poids_satisfaction /= total_poids
        poids_utilisation /= total_poids
        poids_stabilite /= total_poids
        poids_penuries /= total_poids
    
    st.info(f"📊 Normalized weights: Cost({poids_cout:.1%}), Satisfaction({poids_satisfaction:.1%}), Utilization({poids_utilisation:.1%}), Stability({poids_stabilite:.1%}), Shortages({poids_penuries:.1%})")
    
    st.subheader("📦 Custom Demands")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Modify demands for each reference:")
        
        references = [f'REF_{i+1:02d}' for i in range(10)]
        demandes_default = [20, 35, 45, 25, 40, 50, 22, 38, 30, 42]
        
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
        st.metric("Total Demand", f"{sum(demandes_personnalisees):,.0f}")
        
        if st.button("🔄 Reset Demands", use_container_width=True):
            st.rerun()
    
    if st.button("🚀 Launch Advanced Planning", use_container_width=True):
        with st.spinner("Stochastic optimization in progress..."):
            success_setup = system.setup_planning_system(
                S=S, T=T,
                mean_capacity=mean_capacity,
                std_capacity=std_capacity,
                mean_defaut=mean_defaut,
                std_defaut=std_defaut,
                alpha_rework=alpha_rework,
                beta=beta,
                penalite_penurie=penalite,
                demandes_personnalisees=demandes_personnalisees,
                poids_cout=poids_cout,
                poids_satisfaction=poids_satisfaction,
                poids_utilisation=poids_utilisation,
                poids_stabilite=poids_stabilite,
                poids_penuries=poids_penuries,
                use_predicted_rework=use_predicted_rework,
                predicted_rework_rate=prediction_result['rework_rate_for_planning'] if use_predicted_rework else None
            )
            
            if success_setup:
                success_planning = system.run_integrated_planning(time_limit=600)
                
                if success_planning:
                    results = system.analyze_integrated_results()
                    
                    if results:
                        st.success("✅ Stochastic planning successful!")
                        
                        st.subheader("📋 Parameters Used")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Basic parameters:**")
                            st.write(f"- Scenarios: {S}")
                            st.write(f"- Shifts: {T}")
                            st.write(f"- Average capacity: {mean_capacity}")
                        
                        with col2:
                            st.write("**Stochastic parameters:**")
                            st.write(f"- Capacity standard deviation: {std_capacity}")
                            if use_predicted_rework:
                                st.write(f"- Predicted rework rate: {prediction_result['rework_rate_for_planning']:.2f}%")
                                st.write(f"- Variation: ±{rework_variation}%")
                            else:
                                st.write(f"- Average defect rate: {mean_defaut*100:.2f}%")
                                st.write(f"- Defect standard deviation: {std_defaut*100:.2f}%")
                        
                        with col3:
                            st.write("**Rework parameters:**")
                            st.write(f"- Alpha rework: {alpha_rework}")
                            st.write(f"- Beta factor: {beta}")
                            st.write(f"- Penalty: {penalite}")
                        
                        display_scenario_details_advanced(system)
                        
                        return results
                    else:
                        st.error("❌ Error analyzing results")
                else:
                    st.error("❌ Planning failed - No optimal solution")
            else:
                st.error("❌ Configuration error")
    
    return None

def display_scenario_details_advanced(system):
    """Display advanced scenario details with multi-criteria analysis"""
    st.subheader("🎯 Multi-criteria Scenario Analysis")
    
    if not hasattr(system.planner, 'scenario_analysis') or not system.planner.scenario_analysis:
        st.error("❌ No scenario analysis available")
        return
    
    scenario_analysis = system.planner.scenario_analysis
    params = system.planner.parameters
    
    if hasattr(system.planner, 'multicriteria_scores') and system.planner.multicriteria_scores:
        st.subheader("🏆 Multi-criteria Scores")
        
        comparison_data = []
        global_scores = system.planner.multicriteria_scores['global_scores']
        
        for s in range(len(scenario_analysis)):
            kpis = scenario_analysis[s]['kpis']
            comparison_data.append({
                'Scenario': f'S{s+1}',
                'Global Score': f"{global_scores[s]:.3f}",
                'Total Cost': f"{kpis['cout_total']:.0f}",
                'Satisfaction (%)': f"{kpis['satisfaction_globale']:.1f}",
                'Utilization (%)': f"{kpis['utilisation_capacite']:.1f}",
                'Stability': f"{kpis['stabilite']:.1f}",
                'Shortages': f"{kpis['total_penuries']:.1f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        if hasattr(system.planner, 'best_scenario_selection') and system.planner.best_scenario_selection:
            best_selection = system.planner.best_scenario_selection
            best_scenario_id = best_selection['best_scenario_id']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🥇 Best Scenario", f"S{best_scenario_id + 1}")
            with col2:
                st.metric("📊 Multi-criteria Score", f"{best_selection['best_score']:.3f}")
            with col3:
                st.metric("🛡️ Robustness", best_selection['robustness'])
    
    tab_names = [f"Scenario {s+1}" for s in range(len(scenario_analysis))]
    tabs = st.tabs(tab_names)
    
    for tab_idx, (s, scenario_data) in enumerate(scenario_analysis.items()):
        with tabs[tab_idx]:
            
            kpis = scenario_data['kpis']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Satisfaction", f"{kpis['satisfaction_globale']:.1f}%")
            with col2:
                st.metric("Capacity Utilization", f"{kpis['utilisation_capacite']:.1f}%")
            with col3:
                st.metric("Stability", f"{kpis['stabilite']:.1f}")
            with col4:
                st.metric("Total Cost", f"{kpis['cout_total']:,.0f}")
            
            st.write("### 📋 Execution Plan by Shift")
            
            for t in range(params['T']):
                shift_key = t + 1
                if shift_key not in scenario_data['shifts_details']:
                    st.error(f"Missing data for shift {shift_key}")
                    continue
                    
                shift_info = scenario_data['shifts_details'][shift_key]
                
                st.write(f"#### 🔄 Shift {t+1}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    ordre_execution = shift_info['execution_order']
                    ordre_clean = [ref for ref in ordre_execution if ref != 'EMPTY']
                    
                    if ordre_clean:
                        st.write("**Execution order:**")
                        ordre_display = " → ".join(ordre_clean)
                        st.markdown(f"`{ordre_display}`")
                    else:
                        st.write("**No production scheduled**")
                    
                    if shift_info['quantities']:
                        st.write("**Quantities to produce:**")
                        
                        quantities_data = []
                        for ref, qty in shift_info['quantities'].items():
                            if qty > 0:
                                taux_defaut = params['taux_defaut'][(s, ref)]
                                prod_utile = qty * (1 - taux_defaut)
                                prod_recuperee = qty * taux_defaut * params['alpha_rework']
                                total_utile = prod_utile + prod_recuperee
                                
                                quantities_data.append({
                                    'Reference': ref,
                                    'Gross Quantity': f"{qty:.0f}",
                                    'Useful Production': f"{total_utile:.0f}",
                                    'Defect Rate': f"{taux_defaut*100:.1f}%"
                                })
                        
                        if quantities_data:
                            df_quantities = pd.DataFrame(quantities_data)
                            st.dataframe(df_quantities, hide_index=True, use_container_width=True)
                
                with col2:
                    st.write("**Shift Metrics:**")
                    st.metric("Capacity Used", 
                             f"{shift_info['capacity_used']:.0f}/{shift_info['capacity_available']:.0f}")
                    st.metric("Utilization Rate", 
                             f"{shift_info['capacity_utilization']:.1f}%")
                    
                    nb_refs_actives = len([ref for ref, qty in shift_info['quantities'].items() if qty > 0])
                    st.metric("Active References", nb_refs_actives)
                
                st.markdown("---")
            
            st.write("### 📊 Production Summary by Reference")
            
            production_summary = []
            for ref, info in scenario_data['production_summary'].items():
                production_summary.append({
                    'Reference': ref,
                    'Demand': f"{info['demande']:.0f}",
                    'Gross Production': f"{info['production_brute']:.0f}",
                    'Useful Production': f"{info['production_utile']:.0f}",
                    'Shortage': f"{info['penurie']:.0f}",
                    'Coverage Rate': f"{info['taux_couverture']:.1f}%"
                })
            
            df_production = pd.DataFrame(production_summary)
            st.dataframe(df_production, hide_index=True, use_container_width=True)

def dashboard_section_advanced(system, results):
    st.header("📊 Advanced Multi-criteria Dashboard")
    
    if results is None:
        st.warning("⚠️ No planning results available")
        return
    
    if not system.planner or not hasattr(system.planner, 'scenario_analysis'):
        st.error("❌ Analysis data missing")
        return
    
    st.subheader("🔗 Prediction-Planning Integration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Rework Rate", f"{results['predicted_rework_rate']:.2f}%")
    
    with col2:
        taux_utilise = system.planner.parameters['taux_defaut'][(0, system.planner.parameters['R'][0])] * 100
        st.metric("Rate Used in Planning", f"{taux_utilise:.2f}%")
    
    with col3:
        match = abs(results['predicted_rework_rate'] - taux_utilise) < 0.01
        st.metric("Integration", "✅ Successful" if match else "❌ Failed")
    
    if 'multicriteria_scores' in results and results['multicriteria_scores']:
        st.subheader("🏆 Multi-criteria Analysis")
        
        scores = results['multicriteria_scores']
        scenario_analysis = results['scenario_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            comparison_data = []
            for s in range(len(scenario_analysis)):
                kpis = scenario_analysis[s]['kpis']
                global_score = scores['global_scores'][s]
                
                comparison_data.append({
                    'Scenario': f'S{s+1}',
                    'Global Score': f"{global_score:.3f}",
                    'Satisfaction': f"{kpis['satisfaction_globale']:.1f}%",
                    'Utilization': f"{kpis['utilisation_capacite']:.1f}%",
                    'Cost': f"{kpis['cout_total']:,.0f}",
                    'Stability': f"{kpis['stabilite']:.1f}",
                    'Shortages': f"{kpis['total_penuries']:.1f}"
                })
            
            df_scores = pd.DataFrame(comparison_data)
            st.dataframe(df_scores, use_container_width=True)
        
        with col2:
            scenarios = [f'S{s+1}' for s in range(len(scenario_analysis))]
            global_scores_values = list(scores['global_scores'].values())
            
            fig = px.bar(
                x=scenarios, 
                y=global_scores_values,
                title="Multi-criteria Scores by Scenario",
                labels={'x': 'Scenarios', 'y': 'Global Score'}
            )
            
            colors = ['gold' if score == max(global_scores_values) else 'lightblue' 
                     for score in global_scores_values]
            fig.update_traces(marker_color=colors)
            
            st.plotly_chart(fig, use_container_width=True)
        
        if 'best_scenario_selection' in results and results['best_scenario_selection']:
            best_selection = results['best_scenario_selection']
            
            st.subheader("🥇 Recommended Scenario")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best Scenario", f"S{best_selection['best_scenario_id'] + 1}")
            
            with col2:
                st.metric("Multi-criteria Score", f"{best_selection['best_score']:.3f}")
            
            with col3:
                st.metric("Gap with 2nd", f"{best_selection['gap_with_second']:.3f}")
            
            with col4:
                st.metric("Robustness", best_selection['robustness'])
            
            best_id = best_selection['best_scenario_id']
            best_scenario_data = scenario_analysis[best_id]
            
            st.write("### 📋 Recommended Execution Plan")
            
            for t in range(system.planner.parameters['T']):
                shift_key = t + 1
                if shift_key in best_scenario_data['shifts_details']:
                    shift_info = best_scenario_data['shifts_details'][shift_key]
                    ordre = ' → '.join([ref for ref in shift_info['execution_order'] if ref != 'EMPTY'])
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Shift {t+1}:** {ordre}")
                    with col2:
                        st.write(f"Util: {shift_info['capacity_utilization']:.1f}%")

def export_section_advanced(system, results):
    st.header("📁 Export Advanced Results")
    
    if results is None:
        st.warning("⚠️ No results to export")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Complete Excel", use_container_width=True):
            try:
                output = io.BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    summary_data = [{
                        'Predicted_Rework_Rate_Pct': results['predicted_rework_rate'],
                        'Optimal_Total_Cost': results['planning_results']['cout_total'],
                        'Number_Scenarios': len(results['scenario_analysis']),
                        'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    }]
                    
                    df_summary = pd.DataFrame(summary_data)
                    df_summary.to_excel(writer, sheet_name='Integration_Summary', index=False)
                    
                    if 'multicriteria_scores' in results:
                        scores_data = []
                        for s in range(len(results['scenario_analysis'])):
                            scores_data.append({
                                'Scenario': f'S{s+1}',
                                'Global_Score': results['multicriteria_scores']['global_scores'][s],
                                'Cost_Score_Norm': results['multicriteria_scores']['normalized_criteria']['cout'][s],
                                'Satisfaction_Score_Norm': results['multicriteria_scores']['normalized_criteria']['satisfaction'][s],
                                'Utilization_Score_Norm': results['multicriteria_scores']['normalized_criteria']['utilisation'][s],
                                'Stability_Score_Norm': results['multicriteria_scores']['normalized_criteria']['stabilite'][s],
                                'Shortages_Score_Norm': results['multicriteria_scores']['normalized_criteria']['penuries'][s]
                            })
                        
                        df_scores = pd.DataFrame(scores_data)
                        df_scores.to_excel(writer, sheet_name='Multi_Criteria_Scores', index=False)
                    
                    planning_comparison = []
                    for s, data in results['scenario_analysis'].items():
                        kpis = data['kpis']
                        planning_comparison.append({
                            'Scenario': f'S{s+1}',
                            'Satisfaction_Pct': kpis['satisfaction_globale'],
                            'Capacity_Utilization_Pct': kpis['utilisation_capacite'],
                            'Stability': kpis['stabilite'],
                            'Total_Shortages': kpis['total_penuries'],
                            'Total_Cost': kpis['cout_total'],
                            'Production_Cost': kpis['cout_production'],
                            'Shortage_Cost': kpis['cout_penuries'],
                            'Production_Efficiency': kpis['efficacite_production']
                        })
                    
                    df_planning = pd.DataFrame(planning_comparison)
                    df_planning.to_excel(writer, sheet_name='Scenario_Comparison', index=False)
                    
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
                                            'Quantity': qty,
                                            'Rework_Rate_Used': system.planner.parameters['taux_defaut'][(s, ref)] * 100
                                        })
                        
                        if prod_data:
                            df_prod = pd.DataFrame(prod_data)
                            df_prod.to_excel(writer, sheet_name='Production_Details', index=False)
                    
                    if 'best_scenario_selection' in results:
                        best_data = [{
                            'Best_Scenario': f"S{results['best_scenario_selection']['best_scenario_id'] + 1}",
                            'Multi_Criteria_Score': results['best_scenario_selection']['best_score'],
                            'Robustness': results['best_scenario_selection']['robustness'],
                            'Gap_With_2nd': results['best_scenario_selection']['gap_with_second']
                        }]
                        
                        df_best = pd.DataFrame(best_data)
                        df_best.to_excel(writer, sheet_name='Best_Scenario', index=False)
                
                processed_data = output.getvalue()
                
                st.download_button(
                    label="⬇️ Download Complete Excel",
                    data=processed_data,
                    file_name=f"advanced_integrated_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Advanced Excel file generated!")
                
            except Exception as e:
                st.error(f"❌ Export error: {e}")
    
    with col2:
        if st.button("📊 Export Visualizations", use_container_width=True):
            st.info("🚧 Graph export in development")

def main():
    create_header()
    
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
    
    with st.sidebar:
        st.header("🧭 Navigation")
        
        step = st.radio(
            "Choose a step:",
            [
                "📊 1. Load Data",
                "🔮 2. Defect Prediction",
                "🎯 3. New Prediction",
                "📋 4. Advanced Planning",
                "📈 5. Multi-criteria Dashboard",
                "📁 6. Advanced Export"
            ]
        )
        
        st.markdown("---")
        
        st.header("📋 Process Status")
        
        if st.session_state.data is not None:
            st.success("✅ Data loaded")
        else:
            st.error("❌ Data not loaded")
        
        if st.session_state.prediction_trained:
            st.success("✅ Models trained")
        else:
            st.error("❌ Models not trained")
        
        if st.session_state.prediction_result is not None:
            st.success("✅ Prediction completed")
        else:
            st.error("❌ No prediction")
        
        if st.session_state.planning_results is not None:
            st.success("✅ Planning completed")
        else:
            st.error("❌ Planning not completed")
        
        st.markdown("---")
        
        if st.button("🔄 Start Over", use_container_width=True):
            for key in ['system', 'data', 'prediction_trained', 'prediction_result', 'planning_results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        st.header("ℹ️ Information")
        st.markdown("""
        **Advanced Integrated System v2.0**
        
        🔮 **Prediction:** ML models to predict defects
        
        📋 **Planning:** Multi-criteria stochastic optimization
        
        📊 **Dashboard:** Advanced multi-criteria analysis
        
        ⚖️ **Features:**
        - Customizable multi-criteria weights
        - Robustness analysis
        - Weighted global score
        - Complete export
        - **Strict preservation of original workstation names**
        - **Uniform treatment of Excel and demo data**
        - **Flexible column identification**
        """)
    
    if step == "📊 1. Load Data":
        data = load_data_section()
        if data is not None:
            st.session_state.data = data
            st.session_state.prediction_trained = False
            st.session_state.prediction_result = None
            st.session_state.planning_results = None
    
    elif step == "🔮 2. Defect Prediction":
        if st.session_state.data is not None:
            success = prediction_section(st.session_state.system, st.session_state.data)
            if success:
                st.session_state.prediction_trained = True
        else:
            st.warning("⚠️ Please load data in step 1 first")
    
    elif step == "🎯 3. New Prediction":
        if st.session_state.prediction_trained:
            prediction_result = new_prediction_section(st.session_state.system)
            if prediction_result is not None:
                st.session_state.prediction_result = prediction_result
                st.session_state.planning_results = None
        else:
            st.warning("⚠️ Please train models in step 2 first")
    
    elif step == "📋 4. Advanced Planning":
        if st.session_state.prediction_result is not None:
            planning_results = planning_section(st.session_state.system, st.session_state.prediction_result)
            if planning_results is not None:
                st.session_state.planning_results = planning_results
        else:
            st.warning("⚠️ Please make a prediction in step 3 first")
    
    elif step == "📈 5. Multi-criteria Dashboard":
        if st.session_state.planning_results is not None:
            dashboard_section_advanced(st.session_state.system, st.session_state.planning_results)
        else:
            st.warning("⚠️ Please run planning in step 4 first")
    
    elif step == "📁 6. Advanced Export":
        if st.session_state.planning_results is not None:
            export_section_advanced(st.session_state.system, st.session_state.planning_results)
        else:
            st.warning("⚠️ No results to export")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <div style='display: flex; justify-content: center; align-items: center; gap: 40px; margin-bottom: 15px;'>
            <div style='display: flex; align-items: center; gap: 10px;'>
                <span style='font-size: 24px;'>🏭</span>
                <span style='font-weight: bold; color: #1f4e79;'>INDUSTRIAL</span>
            </div>
            <div style='color: #ccc; font-size: 20px;'>×</div>
            <div style='display: flex; align-items: center; gap: 10px;'>
                <span style='font-size: 24px;'>🎓</span>
                <span style='font-weight: bold; color: #2e86ab;'>ACADEMIC</span>
            </div>
        </div>
        <div style='color: #666; margin-top: 8px; font-size: 14px; font-style: italic;'>
            🏭 Advanced Integrated Prediction-Stochastic Planning System | 
            Developed with ❤️ in Streamlit | 
            Model v2.0 - Strict Preservation of Original Names | 
            © 2024
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
       
