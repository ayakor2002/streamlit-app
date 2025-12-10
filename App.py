import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pulp as plp
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Advanced Prediction-Planning System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

class MultiStationDefectPredictor:
    def __init__(self):
        self.models = {}
        self.transformers = {}
        self.best_model_names = {}
        self.feature_importances = {}
        self.original_data = None
        self.stations = []
        self.day_col = None
        self.volume_col = None
        self.predictions_history = []
        self.station_weights = {}

    def calculate_station_weights_from_data(self, data=None):
        if data is None:
            data = self.original_data
        if data is None:
            self.station_weights = {station: 1.0/len(self.stations) for station in self.stations}
            return

        station_qi_di = {}
        total_qi_di = 0

        for station in self.stations:
            if station in data.columns and self.volume_col in data.columns:
                qi = data[self.volume_col].sum()
                di = data[station].sum()
                qi_di = qi * di
                station_qi_di[station] = qi_di
                total_qi_di += qi_di

        if total_qi_di > 0:
            self.station_weights = {station: qi_di / total_qi_di for station, qi_di in station_qi_di.items()}
        else:
            self.station_weights = {station: 1.0/len(self.stations) for station in self.stations}

    def set_station_weights(self, weights_dict=None):
        if weights_dict is None:
            self.calculate_station_weights_from_data()
        else:
            self.station_weights = weights_dict.copy()
            total_weight = sum(self.station_weights.values())
            if total_weight > 0:
                self.station_weights = {station: weight/total_weight for station, weight in self.station_weights.items()}

    def calculate_weighted_average(self, predictions_stations):
        if not self.station_weights:
            self.set_station_weights()

        weighted_sum = 0
        total_weight = 0

        for station, prediction in predictions_stations.items():
            if station in self.station_weights:
                weight = self.station_weights[station]
                weighted_sum += prediction * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0

    def identify_stations(self, data):
        """Identify stations while preserving original column names"""
        station_cols = []
        defect_keywords = ['defects', 'defect', 'rework', 'failed', 'fail', 'defauts']
        
        for col in data.columns:
            if any(keyword in col.lower() for keyword in defect_keywords):
                station_cols.append(col)
        
        if not station_cols:
            excluded_keywords = ['day', 'volume', 'production', 'date', 'time', 'week']
            for col in data.columns:
                is_excluded = any(keyword in col.lower() for keyword in excluded_keywords)
                if not is_excluded and pd.api.types.is_numeric_dtype(data[col]):
                    station_cols.append(col)
        
        if not station_cols:
            raise ValueError("No defect columns identified!")
        
        self.stations = station_cols
        st.info(f"📍 Stations identified: {station_cols}")
        return station_cols

    def prepare_data_for_station(self, data, station_col):
        """Prepare data for a station"""
        data_copy = data.copy()

        day_col = None
        volume_col = None
        
        day_keywords = ['day', 'date', 'week', 'jour', 'semaine']
        for col in data.columns:
            if any(keyword in col.lower() for keyword in day_keywords):
                day_col = col
                break
        
        volume_keywords = ['volume', 'production', 'quantity', 'qty', 'quantite']
        for col in data.columns:
            if any(keyword in col.lower() for keyword in volume_keywords):
                volume_col = col
                break

        if not day_col or not volume_col:
            raise ValueError("'Day' and 'Volume' columns required!")

        self.day_col = day_col
        self.volume_col = volume_col

        try:
            if data_copy[day_col].dtype == 'object':
                try:
                    data_copy[day_col] = pd.to_datetime(data_copy[day_col]).dt.dayofweek + 1
                except:
                    data_copy[day_col] = pd.to_numeric(data_copy[day_col], errors='coerce')
        except:
            pass

        if station_col not in data_copy.columns:
            raise ValueError(f"Station column '{station_col}' not found!")

        X = data_copy[[volume_col, day_col]].copy()
        y = data_copy[station_col].copy()
        
        mask = (~X.isnull().any(axis=1)) & (~y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            raise ValueError("No valid data after cleaning!")

        return X, y, day_col, volume_col

    def train_model_for_station(self, X, y, station_name, day_col, volume_col, search_method='grid', n_iter=20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        numerical_features = [volume_col, day_col]
        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), numerical_features)]
        )

        transformer = preprocessor.fit(X_train)
        self.transformers[station_name] = transformer

        models = {
            'DecisionTree': Pipeline([('preprocessor', preprocessor), ('model', DecisionTreeRegressor(random_state=42))]),
            'RandomForest': Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))]),
            'GradientBoosting': Pipeline([('preprocessor', preprocessor), ('model', GradientBoostingRegressor(random_state=42))]),
            'NeuralNetwork': Pipeline([('preprocessor', preprocessor), ('model', MLPRegressor(max_iter=1000, random_state=42))])
        }

        param_grids = {
            'DecisionTree': {'model__max_depth': [None, 5, 10, 15], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4]},
            'RandomForest': {'model__n_estimators': [50, 100, 150], 'model__max_depth': [None, 10, 15], 'model__min_samples_split': [2, 5]},
            'GradientBoosting': {'model__n_estimators': [50, 100, 150], 'model__learning_rate': [0.01, 0.05, 0.1], 'model__max_depth': [3, 5, 7]},
            'NeuralNetwork': {'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)], 'model__alpha': [0.0001, 0.001, 0.01]}
        }

        best_score = float('inf')
        best_model_name = None
        best_model = None
        all_results = {}

        for name, model in models.items():
            if search_method == 'grid':
                search = GridSearchCV(model, param_grids[name], cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error', n_jobs=-1)
            else:
                search = RandomizedSearchCV(model, param_grids[name], n_iter=n_iter, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
            
            search.fit(X_train, y_train)
            y_pred_test = search.predict(X_test)
            mse = mean_squared_error(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            
            all_results[name] = {'mse': mse, 'mae': mae, 'r2': r2, 'best_params': search.best_params_, 'cv_score': search.best_score_}

            if mse < best_score:
                best_score = mse
                best_model_name = name
                best_model = search.best_estimator_

        self.models[station_name] = best_model
        self.best_model_names[station_name] = best_model_name

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if best_model_name in ['DecisionTree', 'RandomForest', 'GradientBoosting']:
            model_step = best_model.named_steps['model']
            self.feature_importances[station_name] = pd.DataFrame({'feature': numerical_features, 'importance': model_step.feature_importances_}).sort_values('importance', ascending=False)

        return {'model_name': best_model_name, 'mse': mse, 'mae': mae, 'r2': r2, 'all_results': all_results}

    def train_all_stations(self, data, search_method='grid'):
        stations = self.identify_stations(data)
        if not stations:
            raise ValueError("No stations identified!")

        results = {}
        for station in stations:
            X, y, day_col, volume_col = self.prepare_data_for_station(data, station)
            results[station] = self.train_model_for_station(X, y, station, day_col, volume_col, search_method=search_method)

        self.set_station_weights()
        return results, stations

    def predict_single_scenario(self, day, volume):
        if not self.models:
            raise ValueError("Models must be trained before predictions!")

        new_data = pd.DataFrame({self.volume_col: [volume], self.day_col: [day]})
        X_new = new_data[[self.volume_col, self.day_col]]

        predictions_stations = {}
        for station, model in self.models.items():
            predictions_stations[station] = model.predict(X_new)[0]

        predictions_chain = {
            'max': max(predictions_stations.values()),
            'average': np.mean(list(predictions_stations.values())),
            'weighted_average': self.calculate_weighted_average(predictions_stations),
            'sum': sum(predictions_stations.values())
        }

        rework_rate_stations = {station: (defects / volume) * 100 for station, defects in predictions_stations.items()}
        rework_rate_chain = {method: (defects / volume) * 100 for method, defects in predictions_chain.items()}

        prediction_record = {
            'day': day,
            'volume': volume,
            'predictions_stations': predictions_stations,
            'predictions_chain': predictions_chain,
            'rework_rate_stations': rework_rate_stations,
            'rework_rate_chain': rework_rate_chain
        }
        self.predictions_history.append(prediction_record)

        return prediction_record

class StochasticPlanningModel:
    """Advanced stochastic planning model with multicriteria analysis"""

    def __init__(self):
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
                      mean_capacity: float = 160,
                      std_capacity: float = 10,
                      mean_defect: float = 0.04,
                      std_defect: float = 0.01,
                      m: int = 5,
                      alpha_rework: float = 0.8,
                      beta: float = 1.2,
                      shortage_penalty: float = 1000,
                      cost_weight: float = 0.25,
                      satisfaction_weight: float = 0.30,
                      utilization_weight: float = 0.20,
                      stability_weight: float = 0.15,
                      shortage_weight: float = 0.10,
                      use_predicted_rework: bool = False,
                      predicted_rework_rate: float = None):
        """Complete model configuration"""

        if R is None:
            R = [f'REF_{i+1:02d}' for i in range(10)]

        if EDI is None:
            EDI = [20, 35, 45, 25, 40, 50, 22, 38, 30, 42]

        if isinstance(EDI, list):
            EDI_dict = {R[i]: EDI[i] for i in range(min(len(R), len(EDI)))}
        else:
            EDI_dict = EDI

        if p is None:
            p = [[0, 0.20, 0.30, 0.15, 0.25, 0.35, 0.12, 0.22, 0.32, 0.18],
                 [0.20, 0, 0.40, 0.25, 0.35, 0.45, 0.22, 0.32, 0.42, 0.28],
                 [0.30, 0.40, 0, 0.35, 0.45, 0.55, 0.32, 0.42, 0.52, 0.38],
                 [0.15, 0.25, 0.35, 0, 0.30, 0.40, 0.17, 0.27, 0.37, 0.23],
                 [0.25, 0.35, 0.45, 0.30, 0, 0.50, 0.27, 0.37, 0.47, 0.33],
                 [0.35, 0.45, 0.55, 0.40, 0.50, 0, 0.37, 0.47, 0.57, 0.43],
                 [0.12, 0.22, 0.32, 0.17, 0.27, 0.37, 0, 0.24, 0.34, 0.20],
                 [0.22, 0.32, 0.42, 0.27, 0.37, 0.47, 0.24, 0, 0.44, 0.30],
                 [0.32, 0.42, 0.52, 0.37, 0.47, 0.57, 0.34, 0.44, 0, 0.40],
                 [0.18, 0.28, 0.38, 0.23, 0.33, 0.43, 0.20, 0.30, 0.40, 0]]

        if D is None:
            D = [[1.0, 0.2, 0.4, 0.3, 0.1, 0.5, 0.2, 0.3, 0.4, 0.1],
                 [0.2, 1.0, 0.6, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.6],
                 [0.4, 0.6, 1.0, 0.5, 0.4, 0.3, 0.6, 0.5, 0.2, 0.7],
                 [0.3, 0.4, 0.5, 1.0, 0.6, 0.4, 0.3, 0.7, 0.5, 0.4],
                 [0.1, 0.3, 0.4, 0.6, 1.0, 0.5, 0.4, 0.6, 0.7, 0.3],
                 [0.5, 0.2, 0.3, 0.4, 0.5, 1.0, 0.3, 0.2, 0.4, 0.6],
                 [0.2, 0.5, 0.6, 0.3, 0.4, 0.3, 1.0, 0.5, 0.3, 0.7],
                 [0.3, 0.4, 0.5, 0.7, 0.6, 0.2, 0.5, 1.0, 0.4, 0.5],
                 [0.4, 0.3, 0.2, 0.5, 0.7, 0.4, 0.3, 0.4, 1.0, 0.6],
                 [0.1, 0.6, 0.7, 0.4, 0.3, 0.6, 0.7, 0.5, 0.6, 1.0]]

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
        
        defect_rate = {}
        
        if use_predicted_rework and predicted_rework_rate is not None:
            base_rate = predicted_rework_rate / 100
            self.predicted_rework_rate = predicted_rework_rate
            
            for s in range(S):
                for i in R:
                    defaut = max(0.001, min(0.25, np.random.normal(base_rate, std_defect)))
                    defect_rate[(s, i)] = defaut
        else:
            for s in range(S):
                for i in R:
                    defaut = max(0.001, min(0.25, np.random.normal(mean_defect, std_defect)))
                    defect_rate[(s, i)] = defaut

        self.parameters = {
            'S': S, 'T': T, 'R': R, 'EDI': EDI_dict, 'p': p_dict, 'D': D_array,
            'CAPchaine': CAPchaine, 'm': m, 'defect_rate': defect_rate,
            'alpha_rework': alpha_rework, 'beta': beta, 'shortage_penalty': shortage_penalty,
            'mean_capacity': mean_capacity, 'std_capacity': std_capacity,
            'mean_defect': mean_defect, 'std_defect': std_defect,
            'cost_weight': cost_weight, 'satisfaction_weight': satisfaction_weight,
            'utilization_weight': utilization_weight, 'stability_weight': stability_weight,
            'shortage_weight': shortage_weight,
            'use_predicted_rework': use_predicted_rework,
            'predicted_rework_rate': predicted_rework_rate
        }

    def create_model(self):
        """Create optimization model"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']

        self.model = plp.LpProblem("Stochastic_Planning", plp.LpMinimize)

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

        self.variables['shortage'] = plp.LpVariable.dicts(
            "shortage",
            [(s, i) for s in range(S) for i in R],
            lowBound=0,
            cat='Continuous'
        )

    def add_constraints(self):
        """Add model constraints"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        x, q, shortage = self.variables['x'], self.variables['q'], self.variables['shortage']

        for s in range(S):
            for i in R:
                demand_satisfied = plp.lpSum([
                    q[(s, i, t)] * (1 - params['defect_rate'][(s, i)]) +
                    params['alpha_rework'] * q[(s, i, t)] * params['defect_rate'][(s, i)]
                    for t in range(T)
                ])
                self.model += (
                    demand_satisfied + shortage[(s, i)] >= params['EDI'][i],
                    f"Demand_s{s}_i{i}"
                )

        for s in range(S):
            for t in range(T):
                capacity_used = plp.lpSum([
                    q[(s, i, t)] * (1 + params['beta'] * params['defect_rate'][(s, i)])
                    for i in R
                ])
                self.model += (
                    capacity_used <= params['CAPchaine'][(s, t)],
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
                    defect_rate_si = params['defect_rate'][(s, i)]
                    if defect_rate_si < 0.99:
                        production_required = params['m'] / (1 - defect_rate_si + params['alpha_rework'] * defect_rate_si)
                    else:
                        production_required = params['m'] * 2
                    
                    self.model += (
                        q[(s, i, t)] >= production_required * plp.lpSum([x[(i, j, s, t)] for j in range(len(R))]),
                        f"Production_min_s{s}_i{i}_t{t}"
                    )

    def set_objective(self):
        """Define objective function"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        q, shortage = self.variables['q'], self.variables['shortage']

        production_cost = plp.lpSum([
            20 * q[(s, i, t)]
            for s in range(S) for i in R for t in range(T)
        ])

        shortage_cost = plp.lpSum([
            params['shortage_penalty'] * shortage[(s, i)]
            for s in range(S) for i in R
        ])

        self.model += production_cost + shortage_cost

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
            st.error(f"Error during solving: {e}")
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

        shortage_results = {}
        for s in range(S):
            for i in R:
                key = (s, i)
                value = self.variables['shortage'][key].value()
                shortage_results[key] = value if value is not None else 0

        sequencing_results = {}
        for s in range(S):
            for t in range(T):
                sequence = {}
                for i in R:
                    for j in range(len(R)):
                        key = (i, j, s, t)
                        value = self.variables['x'][key].value()
                        if value is not None and value > 0.5:
                            sequence[j] = i
                sequencing_results[(s, t)] = sequence

        self.results = {
            'production': production_results,
            'shortage': shortage_results,
            'sequencing': sequencing_results,
            'total_cost': self.model.objective.value()
        }

    def analyze_scenarios_detailed(self):
        """Detailed scenario analysis"""
        if not self.results:
            return

        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        production = self.results['production']
        shortage = self.results['shortage']
        sequencing = self.results['sequencing']

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

                sequence = sequencing.get((s, t), {})
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
                        defect_rate_i = params['defect_rate'][(s, i)]
                        capacity_used += qty * (1 + params['beta'] * defect_rate_i)

                shift_info['capacity_used'] = capacity_used
                if shift_info['capacity_available'] > 0:
                    shift_info['capacity_utilization'] = (capacity_used / shift_info['capacity_available']) * 100
                else:
                    shift_info['capacity_utilization'] = 0

                total_capacity_used += capacity_used
                total_capacity_available += shift_info['capacity_available']

                scenario_data['shifts_details'][t+1] = shift_info

            total_useful_production = 0
            total_demand = sum(params['EDI'].values())
            total_shortage = 0

            for i in R:
                total_prod = sum(production[(s, i, t)] for t in range(T))
                
                total_useful = 0
                for t in range(T):
                    qty = production[(s, i, t)]
                    defect_rate_i = params['defect_rate'][(s, i)]
                    good_pieces = qty * (1 - defect_rate_i)
                    rework_ok = qty * defect_rate_i * params['alpha_rework']
                    total_useful += good_pieces + rework_
         
