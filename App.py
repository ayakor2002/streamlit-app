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

# Streamlit page configuration
st.set_page_config(
    page_title="Advanced Prediction-Planning System",
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
            raise ValueError(f"'Day' and 'Volume' columns required!")

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
            search = GridSearchCV(model, param_grids[name], cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error', n_jobs=-1) if search_method == 'grid' else RandomizedSearchCV(model, param_grids[name], n_iter=n_iter, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
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

        return {
            'day': day,
            'volume': volume,
            'predictions_stations': predictions_stations,
            'predictions_chain': predictions_chain,
            'rework_rate_stations': rework_rate_stations,
            'rework_rate_chain': rework_rate_chain
        }

class IntegratedPredictionPlanningSystem:
    def __init__(self):
        self.predictor = None
        self.predicted_rework_rate = None

    def setup_prediction_system(self, data):
        self.predictor = MultiPosteDefectPredictor()
        self.predictor.original_data = data.copy()
        results, stations = self.predictor.train_all_stations(data, search_method='grid')
        return results, stations

    def make_prediction_for_planning(self, day, volume, method='weighted_average'):
        if self.predictor is None:
            raise ValueError("Prediction system must be configured first!")

        prediction_result = self.predictor.predict_single_scenario(day, volume)
        rework_rate_chain = prediction_result['rework_rate_chain'][method]
        self.predicted_rework_rate = rework_rate_chain

        return {
            'prediction_details': prediction_result,
            'rework_rate_for_planning': rework_rate_chain,
            'method_used': method
        }

def create_header():
    """Create header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏭 Advanced Prediction-Planning Integration System</h1>
        <p>Defect Prediction & Multicriteria Stochastic Planning</p>
    </div>
    """, unsafe_allow_html=True)

def load_data_section():
    """Load data section"""
    st.header("📊 Load Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your Excel file with historical data",
            type=['xlsx', 'xls'],
            help="File must contain: Day, Volume_production, and defect columns per station"
        )
    
    with col2:
        if st.button("📝 Use Demo Data", use_container_width=True):
            with st.spinner("Generating demo data..."):
                demo_data = create_demo_data()
                st.success(f"✅ Demo data generated: {len(demo_data)} rows")
                display_data_info(demo_data)
                return demo_data
    
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.success(f"✅ Data loaded: {len(data)} rows, {len(data.columns)} columns")
            display_data_info(data)
            return data
        except Exception as e:
            st.error(f"❌ Error loading: {e}")
            return None
    
    return None

def display_data_info(data):
    """Display data information"""
    with st.expander("👀 Data Preview", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**First rows:**")
            st.dataframe(data.head(10), use_container_width=True)
        
        with col2:
            st.write("**Statistics:**")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(data[numeric_cols].describe().round(2))

def create_demo_data(n_days=100):
    """Create demo data"""
    np.random.seed(42)
    days = range(1, n_days + 1)
    data = []

    for day in days:
        day_of_week = ((day - 1) % 7) + 1
        volume_base = 800 if day_of_week in [6, 7] else 1200
        volume = volume_base + np.random.normal(0, 100)
        volume = max(volume, 100)

        defects_1 = max(0, volume * 0.02 + day_of_week * 0.5 + np.random.normal(0, 2))
        defects_2 = max(0, volume * 0.015 + day_of_week * 0.3 + np.random.normal(0, 1.5))
        defects_3 = max(0, volume * 0.025 + day_of_week * 0.4 + np.random.normal(0, 2.5))

        data.append({
            'Day': day_of_week,
            'Production_Volume': volume,
            'Station1_Defects': defects_1,
            'Station2_Defects': defects_2,
            'Station3_Defects': defects_3
        })

    return pd.DataFrame(data)

def prediction_section(system, data):
    """Prediction section"""
    st.header("🔮 Defect Prediction")
    
    if data is None:
        st.warning("⚠️ Please load data first")
        return False
    
    try:
        with st.spinner("🧠 Training prediction models..."):
            results, stations = system.setup_prediction_system(data)
        
        if results and stations:
            st.success("✅ Models trained successfully!")
            
            st.subheader("🏆 Optimal Models Selected")
            
            model_data = []
            for station in stations:
                if station in results:
                    result = results[station]
                    model_data.append({
                        'Station': station,
                        'Optimal Model': result['model_name'],
                        'MSE': f"{result['mse']:.4f}",
                        'MAE': f"{result['mae']:.4f}",
                        'R²': f"{result['r2']:.4f}"
                    })
            
            st.dataframe(pd.DataFrame(model_data), use_container_width=True)
            return True
        else:
            st.error("❌ Model training failed")
            return False
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return False

def new_prediction_section(system):
    """Make new prediction"""
    st.header("🎯 Make Prediction")
    
    if system.predictor is None:
        st.warning("⚠️ Configure prediction system first")
        return None
    
    col1, col2 = st.columns(2)
    
    with col1:
        day = st.selectbox(
            "Day of week",
            options=[1, 2, 3, 4, 5, 6, 7],
            format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x-1],
            index=2
        )
    
    with col2:
        volume = st.number_input("Production volume", min_value=1, max_value=10000, value=1200, step=50)
    
    if st.button("🔮 Make Prediction", use_container_width=True):
        with st.spinner("Calculating..."):
            prediction_result = system.make_prediction_for_planning(day, volume)
        
        st.subheader("📊 Prediction Results")
        
        pred_details = prediction_result['prediction_details']
        rework_rate = prediction_result['rework_rate_for_planning']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rework Rate", f"{rework_rate:.2f}%")
        with col2:
            st.metric("Predicted Defects", f"{pred_details['predictions_chain']['weighted_average']:.1f}")
        with col3:
            st.metric("Volume", f"{volume:,.0f}")
        with col4:
            day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day-1]
            st.metric("Day", day_name)
        
        st.subheader("📈 Defects by Station")
        
        stations = list(pred_details['predictions_stations'].keys())
        defects = list(pred_details['predictions_stations'].values())
        
        fig = px.bar(x=stations, y=defects, title="Predicted Defects by Station", color=defects, color_continuous_scale='reds')
        st.plotly_chart(fig, use_container_width=True)
        
        return prediction_result
    
    return None

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
    
    with st.sidebar:
        st.header("🧭 Navigation")
        
        step = st.radio("Choose a step:", ["📊 1. Load Data", "🔮 2. Train Prediction", "🎯 3. Make Prediction"])
        
        st.markdown("---")
        st.header("📋 Process Status")
        
        if st.session_state.data is not None:
            st.success("✅ Data loaded")
        else:
            st.error("❌ No data")
        
        if st.session_state.prediction_trained:
            st.success("✅ Models trained")
        else:
            st.error("❌ Not trained")
        
        if st.session_state.prediction_result is not None:
            st.success("✅ Prediction done")
        else:
            st.error("❌ No prediction")
        
        if st.button("🔄 Restart", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    if step == "📊 1. Load Data":
        data = load_data_section()
        if data is not None:
            st.session_state.data = data
    elif step == "🔮 2. Train Prediction":
        if st.session_state.data is not None:
            success = prediction_section(st.session_state.system, st.session_state.data)
            if success:
                st.session_state.prediction_trained = True
        else:
            st.warning("⚠️ Load data first")
    elif step == "🎯 3. Make Prediction":
        if st.session_state.prediction_trained:
            prediction_result = new_prediction_section(st.session_state.system)
            if prediction_result is not None:
                st.session_state.prediction_result = prediction_result
        else:
            st.warning("⚠️ Train models first")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <div style='color: #666; margin-top: 8px; font-size: 14px; font-style: italic;'>
            Advanced Prediction-Planning System | v1.0 | © 2024
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
          
    
 
