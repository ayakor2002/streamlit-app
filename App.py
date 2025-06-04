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
from scipy import stats
warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Programmation Stochastique à Deux Étapes - YAZAKI x ENSAM",
    page_icon="🎲",
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
    .stochastic-info {
        background: linear-gradient(135deg, #e8f4fd, #d4e7f7);
        border: 2px solid #2e86ab;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class MultiPosteDefectPredictor:
    """Système de prédiction exactement comme dans le code original"""
    
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
        """Identifie automatiquement les colonnes de défauts par poste"""
        patterns = ['_defauts', '_defaut', 'defauts', 'defaut', 'Defauts', 'Defaut', 'DEFAUTS', 'DEFAUT']
        postes_cols = []
        
        for pattern in patterns:
            cols_found = [col for col in data.columns if pattern in col]
            postes_cols.extend(cols_found)
        
        postes_cols = list(set(postes_cols))
        
        if not postes_cols:
            excluded_keywords = ['jour', 'volume', 'production', 'date', 'time', 'timestamp', 'id']
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            postes_cols = []
            for col in numeric_cols:
                is_excluded = any(keyword.lower() in col.lower() for keyword in excluded_keywords)
                if not is_excluded:
                    postes_cols.append(col)
        
        if not postes_cols:
            all_cols = list(data.columns)
            if len(all_cols) > 2:
                postes_cols = all_cols[2:]
            else:
                raise ValueError("Impossible d'identifier des colonnes de défauts dans les données")
        
        self.postes = postes_cols
        st.info(f"📍 **Postes identifiés automatiquement:** {', '.join(postes_cols)}")
        return postes_cols

    def prepare_data_for_poste(self, data, poste_col):
        """Prépare les données pour l'entraînement d'un poste"""
        try:
            data_copy = data.copy()

            # Recherche flexible des colonnes jour et volume
            jour_col = None
            volume_col = None
            
            jour_patterns = ['jour', 'day', 'date', 'time']
            for pattern in jour_patterns:
                for col in data.columns:
                    if pattern.lower() in col.lower():
                        jour_col = col
                        break
                if jour_col:
                    break
            
            volume_patterns = ['volume', 'production', 'prod', 'quantity', 'qty']
            for pattern in volume_patterns:
                for col in data.columns:
                    if pattern.lower() in col.lower():
                        volume_col = col
                        break
                if volume_col:
                    break
            
            if not jour_col or not volume_col:
                cols = list(data.columns)
                if len(cols) >= 2:
                    jour_col = jour_col or cols[0]
                    volume_col = volume_col or cols[1]
                else:
                    raise ValueError("Pas assez de colonnes dans les données")

            if not jour_col or not volume_col:
                raise ValueError("Les colonnes 'jour' et 'volume de production' sont requises")

            self.jour_col = jour_col
            self.volume_col = volume_col
            
            st.info(f"📊 **Colonnes utilisées:** Jour='{jour_col}', Volume='{volume_col}', Défauts='{poste_col}'")

            # Conversion intelligente de la colonne jour
            try:
                data_copy['jour_numerique'] = pd.to_datetime(data_copy[jour_col])
                data_copy['jour_numerique'] = data_copy['jour_numerique'].dt.dayofweek + 1
                jour_col_final = 'jour_numerique'
                st.info("✅ Conversion automatique jour → jour de semaine (1-7)")
            except:
                try:
                    data_copy['jour_numerique'] = pd.to_numeric(data_copy[jour_col])
                    jour_col_final = 'jour_numerique'
                    st.info("✅ Utilisation directe des valeurs numériques du jour")
                except:
                    jour_col_final = jour_col
                    st.warning("⚠️ Impossible de convertir la colonne jour - utilisation directe")

            # Vérification des types et nettoyage
            try:
                data_copy[volume_col] = pd.to_numeric(data_copy[volume_col], errors='coerce')
                data_copy[poste_col] = pd.to_numeric(data_copy[poste_col], errors='coerce')
                data_copy[jour_col_final] = pd.to_numeric(data_copy[jour_col_final], errors='coerce')
                
                initial_rows = len(data_copy)
                data_copy = data_copy.dropna(subset=[volume_col, poste_col, jour_col_final])
                final_rows = len(data_copy)
                
                if final_rows < initial_rows:
                    st.warning(f"⚠️ {initial_rows - final_rows} lignes supprimées (valeurs manquantes)")
                
                if final_rows < 10:
                    raise ValueError(f"Pas assez de données valides pour l'entraînement ({final_rows} lignes)")
                
            except Exception as e:
                st.error(f"❌ Erreur lors du nettoyage des données: {e}")
                raise

            X = data_copy[[volume_col, jour_col_final]]
            y = data_copy[poste_col]
            
            st.success(f"✅ Données préparées: {len(X)} échantillons pour le poste '{poste_col}'")

            return X, y, jour_col_final, volume_col
            
        except Exception as e:
            st.error(f"❌ Erreur dans prepare_data_for_poste pour {poste_col}: {e}")
            raise

    def train_model_for_poste(self, X, y, poste_name, jour_col, volume_col, search_method='grid', n_iter=20):
        """Entraîne un modèle pour un poste donné avec la méthode exacte du code original"""
        try:
            if len(X) == 0 or len(y) == 0:
                raise ValueError(f"Données vides pour le poste {poste_name}")
            
            if len(X) != len(y):
                raise ValueError(f"Tailles incompatibles X({len(X)}) et y({len(y)}) pour {poste_name}")
            
            if y.var() < 1e-10:
                st.warning(f"⚠️ Variance très faible pour {poste_name} - résultats peuvent être peu fiables")
            
            test_size = 0.2 if len(X) > 50 else 0.3 if len(X) > 20 else 0.5
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            st.info(f"📊 **{poste_name}:** {len(X_train)} train, {len(X_test)} test")

            numerical_features = [volume_col, jour_col]
            preprocessor = ColumnTransformer(
                transformers=[('num', StandardScaler(), numerical_features)],
                remainder='passthrough'
            )

            transformer = preprocessor.fit(X_train)
            self.transformers[poste_name] = transformer

            # Modèles avec paramètres adaptés - EXACTEMENT COMME LE CODE ORIGINAL
            models = {
                'DecisionTree': Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', DecisionTreeRegressor(random_state=42, min_samples_leaf=max(1, len(X_train)//20)))
                ]),
                'RandomForest': Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', RandomForestRegressor(random_state=42, n_estimators=min(100, max(10, len(X_train)//2))))
                ]),
                'GradientBoosting': Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', GradientBoostingRegressor(random_state=42, n_estimators=min(100, max(10, len(X_train)//2))))
                ]),
                'NeuralNetwork': Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', MLPRegressor(max_iter=1000, random_state=42, hidden_layer_sizes=(min(100, max(10, len(X_train)//4)),)))
                ])
            }

            # Grilles de paramètres EXACTEMENT COMME LE CODE ORIGINAL
            param_grids = {
                'DecisionTree': {
                    'model__max_depth': [None, 3, 5, 10] if len(X_train) > 50 else [None, 3, 5],
                    'model__min_samples_split': [2, 5] if len(X_train) > 50 else [2],
                    'model__min_samples_leaf': [1, 2] if len(X_train) > 50 else [1]
                },
                'RandomForest': {
                    'model__n_estimators': [10, 50, 100] if len(X_train) > 50 else [10, 50],
                    'model__max_depth': [None, 5, 10] if len(X_train) > 50 else [None, 5],
                    'model__min_samples_split': [2, 5] if len(X_train) > 50 else [2]
                },
                'GradientBoosting': {
                    'model__n_estimators': [10, 50, 100] if len(X_train) > 50 else [10, 50],
                    'model__learning_rate': [0.05, 0.1, 0.2] if len(X_train) > 50 else [0.1],
                    'model__max_depth': [3, 5] if len(X_train) > 50 else [3]
                },
                'NeuralNetwork': {
                    'model__hidden_layer_sizes': [(50,), (100,)] if len(X_train) > 50 else [(50,)],
                    'model__alpha': [0.001, 0.01] if len(X_train) > 50 else [0.01]
                }
            }

            best_score = float('inf')
            best_model_name = None
            best_model = None
            best_params = None
            model_scores = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, (name, model) in enumerate(models.items()):
                try:
                    status_text.text(f"🔍 Entraînement {name} pour {poste_name}...")
                    progress_bar.progress((idx + 1) / len(models))
                    
                    cv_folds = min(5, max(3, len(X_train) // 10))
                    
                    if search_method == 'grid' and len(X_train) > 30:
                        search = GridSearchCV(
                            model,
                            param_grids[name],
                            cv=cv_folds,
                            scoring='neg_mean_squared_error',
                            n_jobs=1,
                            error_score='raise'
                        )
                    else:
                        search = RandomizedSearchCV(
                            model,
                            param_grids[name],
                            n_iter=min(n_iter, 5),
                            cv=cv_folds,
                            scoring='neg_mean_squared_error',
                            random_state=42,
                            n_jobs=1,
                            error_score='raise'
                        )

                    search.fit(X_train, y_train)
                    y_pred_test = search.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred_test)
                    model_scores[name] = mse

                    if mse < best_score:
                        best_score = mse
                        best_model_name = name
                        best_model = search.best_estimator_
                        best_params = search.best_params_
                        
                except Exception as e:
                    st.warning(f"⚠️ Erreur avec {name} pour {poste_name}: {e}")
                    model_scores[name] = float('inf')
                    continue

            progress_bar.empty()
            status_text.empty()

            if best_model is None:
                raise ValueError(f"Aucun modèle n'a pu être entraîné pour {poste_name}")

            self.models[poste_name] = best_model
            self.best_model_names[poste_name] = best_model_name

            # Métriques finales
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Feature importance si disponible
            if best_model_name in ['DecisionTree', 'RandomForest', 'GradientBoosting']:
                try:
                    model_step = best_model.named_steps['model']
                    if hasattr(model_step, 'feature_importances_'):
                        self.feature_importances[poste_name] = pd.DataFrame({
                            'feature': numerical_features,
                            'importance': model_step.feature_importances_
                        }).sort_values('importance', ascending=False)
                except:
                    pass

            st.success(f"✅ **{poste_name}:** {best_model_name} | MSE={mse:.3f} | R²={r2:.3f}")

            return {
                'model_name': best_model_name,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'y_test': y_test,
                'y_pred': y_pred,
                'X_test': X_test,
                'best_params': best_params,
                'model_scores': model_scores
            }
            
        except Exception as e:
            st.error(f"❌ Erreur critique lors de l'entraînement pour {poste_name}: {e}")
            raise

    def train_all_postes(self, data, search_method='grid'):
        """Entraîne tous les modèles EXACTEMENT COMME LE CODE ORIGINAL"""
        try:
            st.info("🔍 **Identification des postes...**")
            postes = self.identify_postes(data)

            if not postes:
                raise ValueError("Aucun poste identifié dans les données!")

            st.info(f"🎯 **Entraînement de {len(postes)} postes:** {', '.join(postes)}")
            
            results = {}
            successful_postes = []
            failed_postes = []
            
            overall_progress = st.progress(0)
            
            for idx, poste in enumerate(postes):
                try:
                    st.subheader(f"🔧 Entraînement Poste: {poste}")
                    
                    X, y, jour_col, volume_col = self.prepare_data_for_poste(data, poste)
                    result = self.train_model_for_poste(X, y, poste, jour_col, volume_col, search_method=search_method)
                    results[poste] = result
                    successful_postes.append(poste)
                    
                except Exception as e:
                    st.error(f"❌ Échec pour {poste}: {e}")
                    failed_postes.append(poste)
                    continue
                
                overall_progress.progress((idx + 1) / len(postes))

            overall_progress.empty()

            if not successful_postes:
                raise ValueError("Aucun modèle n'a pu être entraîné avec succès!")

            self.postes = successful_postes
            self.set_poste_weights()
            
            st.success(f"✅ **Entraînement terminé:** {len(successful_postes)} modèles créés")
            
            if failed_postes:
                st.warning(f"⚠️ **Postes échoués:** {', '.join(failed_postes)}")
            
            if successful_postes:
                st.subheader("📊 Résumé des Performances")
                summary_data = []
                for poste in successful_postes:
                    result = results[poste]
                    summary_data.append({
                        'Poste': poste,
                        'Modèle': result['model_name'],
                        'MSE': round(result['mse'], 3),
                        'MAE': round(result['mae'], 3),
                        'R²': round(result['r2'], 3)
                    })
                
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)

            return results, successful_postes
            
        except Exception as e:
            st.error(f"❌ Erreur critique dans train_all_postes: {e}")
            raise

    def predict_single_scenario(self, jour, volume):
        """Effectue une prédiction EXACTEMENT COMME LE CODE ORIGINAL"""
        try:
            if not self.models:
                raise ValueError("Les modèles doivent être entraînés avant de faire des prédictions!")

            if volume <= 0:
                raise ValueError("Le volume de production doit être positif")
            
            if jour < 1 or jour > 7:
                raise ValueError("Le jour doit être entre 1 et 7")

            # Construire les données d'entrée selon le format utilisé lors de l'entraînement
            first_transformer = list(self.transformers.values())[0]
            feature_names = first_transformer.transformers_[0][2]
            
            if 'jour_numerique' in feature_names:
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
            prediction_errors = {}
            
            # Prédiction pour chaque poste
            for poste, model in self.models.items():
                try:
                    prediction = model.predict(X_new)[0]
                    predictions_postes[poste] = max(0, prediction)
                except Exception as e:
                    st.warning(f"⚠️ Erreur prédiction pour {poste}: {e}")
                    prediction_errors[poste] = str(e)
                    predictions_postes[poste] = 0

            if not predictions_postes or all(v == 0 for v in predictions_postes.values()):
                raise ValueError("Aucune prédiction valide n'a pu être générée")

            # Calculs des agrégations
            valid_predictions = [v for v in predictions_postes.values() if v > 0]
            
            predictions_chaine = {
                'max': max(predictions_postes.values()) if predictions_postes else 0,
                'moyenne': np.mean(list(predictions_postes.values())) if predictions_postes else 0,
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
                'taux_rework_chaine': taux_rework_chaine,
                'errors': prediction_errors if prediction_errors else None
            }
            
            self.predictions_history.append(prediction_record)

            if prediction_errors:
                st.warning(f"⚠️ Erreurs sur {len(prediction_errors)} postes: {list(prediction_errors.keys())}")

            return prediction_record
            
        except Exception as e:
            st.error(f"❌ Erreur dans predict_single_scenario: {e}")
            raise

class TwoStageStochasticPlanner:
    """Modèle de programmation stochastique à deux étapes CORRIGÉ"""
    
    def __init__(self):
        self.model = None
        self.variables = {}
        self.results = {}
        self.parameters = {}
        self.scenarios = []
        self.scenario_analysis = {}
        self.predicted_rework_rate = None
        
    def generate_stochastic_scenarios(self, S: int, T: int, R: List[str], 
                                    predicted_rework_rate: float,
                                    mean_capacity: float,
                                    rework_std: float = 0.01,
                                    capacity_std_pct: float = 25.0,
                                    base_demands: List[float] = None,
                                    alpha_rework: float = 0.8,  # FIXE maintenant
                                    beta_factor: float = 1.2,   # FIXE maintenant
                                    **other_params):
        """
        Génère les scénarios stochastiques avec ALPHA et BETA FIXES
        Variables aléatoires: SEULEMENT Taux rework et Capacité
        """
        
        np.random.seed(42)
        
        self.predicted_rework_rate = predicted_rework_rate
        
        rework_mean = predicted_rework_rate / 100
        capacity_std = mean_capacity * (capacity_std_pct / 100)
        
        st.info(f"🎲 Génération de {S} scénarios avec:")
        st.info(f"   • Taux rework ~ N({predicted_rework_rate:.3f}%, {rework_std:.3f})")
        st.info(f"   • Capacité ~ N({mean_capacity:.0f}, ±{capacity_std_pct:.0f}%)")
        st.info(f"   • Alpha rework = {alpha_rework:.2f} (FIXE)")
        st.info(f"   • Beta factor = {beta_factor:.2f} (FIXE)")
        
        if base_demands is None:
            base_demands = [25, 40, 35, 30, 45, 38, 28, 42, 33, 37]
        
        EDI_dict = {}
        for i, ref in enumerate(R):
            EDI_dict[ref] = base_demands[i] if i < len(base_demands) else 30
        
        scenarios = []
        
        for s in range(S):
            scenario = {
                'id': s,
                'probability': 1.0 / S,
                'rework_rates': {},
                'capacities': {},
                'alpha_rework': alpha_rework,  # FIXE POUR TOUS LES SCÉNARIOS
                'beta_factor': beta_factor     # FIXE POUR TOUS LES SCÉNARIOS
            }
            
            # 1. Taux de rework stochastiques (par référence)
            for ref in R:
                rework_rate = np.random.normal(rework_mean, rework_std)
                rework_rate = max(0.001, min(0.30, rework_rate))
                scenario['rework_rates'][ref] = rework_rate
            
            # 2. Capacités stochastiques (par shift)
            for t in range(T):
                capacity = np.random.normal(mean_capacity, capacity_std)
                capacity = max(50, capacity)
                scenario['capacities'][t] = capacity
            
            scenarios.append(scenario)
        
        self.scenarios = scenarios
        
        self.parameters = {
            'S': S, 'T': T, 'R': R,
            'EDI': EDI_dict,
            'predicted_rework_rate': predicted_rework_rate,
            'rework_std': rework_std,
            'mean_capacity': mean_capacity,
            'capacity_std_pct': capacity_std_pct,
            'capacity_std': capacity_std,
            'base_demands': base_demands,
            'alpha_rework': alpha_rework,  # PARAMÈTRE FIXE
            'beta_factor': beta_factor,    # PARAMÈTRE FIXE
            **other_params
        }
        
        self.validate_scenarios()
        return scenarios
    
    def validate_scenarios(self):
        """Valide que les scénarios sont bien différenciés"""
        if not self.scenarios:
            return
        
        st.subheader("🔍 Validation des Scénarios Stochastiques")
        
        rework_means = []
        capacity_means = []
        
        for scenario in self.scenarios:
            rework_mean = np.mean(list(scenario['rework_rates'].values())) * 100
            capacity_mean = np.mean(list(scenario['capacities'].values()))
            rework_means.append(rework_mean)
            capacity_means.append(capacity_mean)
        
        rework_variance = np.var(rework_means)
        capacity_variance = np.var(capacity_means)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Variance Taux Rework", f"{rework_variance:.6f}")
            if rework_variance > 0.001:
                st.success("✅ Bonne différenciation")
            else:
                st.warning("⚠️ Faible variance")
        
        with col2:
            st.metric("Variance Capacité", f"{capacity_variance:.2f}")
            if capacity_variance > 100:
                st.success("✅ Bonne différenciation")
            else:
                st.warning("⚠️ Faible variance")
        
        with col3:
            predicted_mean = np.mean(rework_means)
            st.metric("Centrage sur Prédiction", f"{abs(predicted_mean - self.predicted_rework_rate):.4f}")
            if abs(predicted_mean - self.predicted_rework_rate) < 0.1:
                st.success("✅ Bien centré")
            else:
                st.warning("⚠️ Décentré")
        
        # Tableau des scénarios
        scenario_data = []
        for i, scenario in enumerate(self.scenarios):
            rework_mean = np.mean(list(scenario['rework_rates'].values())) * 100
            capacity_mean = np.mean(list(scenario['capacities'].values()))
            
            scenario_data.append({
                'Scénario': f'S{i+1}',
                'Taux Rework Moyen (%)': f"{rework_mean:.3f}",
                'Capacité Moyenne': f"{capacity_mean:.0f}",
                'Alpha (FIXE)': f"{scenario['alpha_rework']:.3f}",
                'Beta (FIXE)': f"{scenario['beta_factor']:.2f}",
                'Probabilité': f"{scenario['probability']:.3f}"
            })
        
        df_scenarios = pd.DataFrame(scenario_data)
        st.dataframe(df_scenarios, use_container_width=True)
    
    def create_two_stage_model(self):
        """Crée le modèle de programmation stochastique à deux étapes"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        
        self.model = plp.LpProblem("Programmation_Stochastique_Deux_Etapes", plp.LpMinimize)
        
        # VARIABLES DE PREMIÈRE ÉTAPE (communes à tous les scénarios)
        self.variables['x_sequence'] = plp.LpVariable.dicts(
            "x_sequence",
            [(i, j, t) for i in R for j in range(len(R)) for t in range(T)],
            cat='Binary'
        )
        
        # VARIABLES DE DEUXIÈME ÉTAPE (par scénario)
        self.variables['q'] = plp.LpVariable.dicts(
            "q_production",
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
        
        return True
    
    def add_two_stage_constraints(self):
        """Ajoute les contraintes du modèle à deux étapes"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        
        x_seq = self.variables['x_sequence']
        q = self.variables['q']
        shortage = self.variables['shortage']
        
        # CONTRAINTES DE PREMIÈRE ÉTAPE
        for t in range(T):
            for j in range(len(R)):
                self.model += (
                    plp.lpSum([x_seq[(i, j, t)] for i in R]) == 1,
                    f"Une_ref_par_position_t{t}_j{j}"
                )
        
        for t in range(T):
            for i in R:
                self.model += (
                    plp.lpSum([x_seq[(i, j, t)] for j in range(len(R))]) <= 1,
                    f"Une_position_par_ref_t{t}_i{i}"
                )
        
        # CONTRAINTES DE DEUXIÈME ÉTAPE (par scénario)
        for s in range(S):
            scenario = self.scenarios[s]
            
            # Contrainte de satisfaction de la demande
            for i in R:
                rework_rate = scenario['rework_rates'][i]
                alpha_rework = scenario['alpha_rework']  # FIXE
                
                production_utile = plp.lpSum([
                    q[(s, i, t)] * (1 - rework_rate) +
                    q[(s, i, t)] * rework_rate * alpha_rework
                    for t in range(T)
                ])
                
                self.model += (
                    production_utile + shortage[(s, i)] >= params['EDI'][i],
                    f"Demande_s{s}_i{i}"
                )
            
            # Contrainte de capacité
            for t in range(T):
                realized_capacity = scenario['capacities'][t]
                beta_factor = scenario['beta_factor']  # FIXE
                
                capacite_utilisee = plp.lpSum([
                    q[(s, i, t)] * (1 + beta_factor * scenario['rework_rates'][i])
                    for i in R
                ])
                
                self.model += (
                    capacite_utilisee <= realized_capacity,
                    f"Capacite_s{s}_t{t}"
                )
            
            # Production minimale si référence sélectionnée
            for i in R:
                for t in range(T):
                    rework_rate = scenario['rework_rates'][i]
                    production_min = params.get('m', 5) / (1 - rework_rate)
                    
                    self.model += (
                        q[(s, i, t)] >= production_min * plp.lpSum([x_seq[(i, j, t)] for j in range(len(R))]),
                        f"Production_min_s{s}_i{i}_t{t}"
                    )
    
    def set_two_stage_objective(self):
        """Fonction objectif à deux étapes"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        
        q = self.variables['q']
        shortage = self.variables['shortage']
        
        # Coûts de deuxième étape (espérance sur les scénarios)
        cout_production_esperance = plp.lpSum([
            self.scenarios[s]['probability'] * 20 * q[(s, i, t)]
            for s in range(S) for i in R for t in range(T)
        ])
        
        cout_penuries_esperance = plp.lpSum([
            self.scenarios[s]['probability'] * params.get('penalite_penurie', 1000) * shortage[(s, i)]
            for s in range(S) for i in R
        ])
        
        self.model += cout_production_esperance + cout_penuries_esperance
    
    def solve_two_stage_model(self, solver_name='PULP_CBC_CMD', time_limit=300):
        """Résout le modèle stochastique à deux étapes"""
        try:
            if solver_name == 'PULP_CBC_CMD':
                solver = plp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
            else:
                solver = plp.getSolver(solver_name)
            
            st.info("🔍 Résolution du modèle d'optimisation stochastique...")
            self.model.solve(solver)
            
            status = plp.LpStatus[self.model.status]
            st.info(f"📊 Statut de résolution: {status}")
            
            if self.model.status == plp.LpStatusOptimal:
                st.success("✅ Solution optimale trouvée!")
                self._extract_two_stage_results()
                return True
            else:
                st.error(f"❌ Pas de solution optimale: {status}")
                return False
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la résolution: {e}")
            return False
    
    def _extract_two_stage_results(self):
        """Extrait les résultats du modèle à deux étapes"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        
        # RÉSULTATS DE PREMIÈRE ÉTAPE
        sequencement_first_stage = {}
        for t in range(T):
            sequence = {}
            for i in R:
                for j in range(len(R)):
                    var_value = self.variables['x_sequence'][(i, j, t)].value()
                    if var_value and var_value > 0.5:
                        sequence[j] = i
            sequencement_first_stage[t] = sequence
        
        # RÉSULTATS DE DEUXIÈME ÉTAPE
        production_second_stage = {}
        for s in range(S):
            for i in R:
                for t in range(T):
                    key = (s, i, t)
                    production_second_stage[key] = self.variables['q'][key].value() or 0
        
        penuries_second_stage = {}
        for s in range(S):
            for i in R:
                key = (s, i)
                penuries_second_stage[key] = self.variables['shortage'][key].value() or 0
        
        cout_total = self.model.objective.value() or 0
        
        self.results = {
            'first_stage': {
                'sequencement': sequencement_first_stage
            },
            'second_stage': {
                'production': production_second_stage,
                'penuries': penuries_second_stage
            },
            'cout_total_esperance': cout_total,
            'statut_optimal': True
        }
        
        st.success(f"✅ Résultats extraits: Coût total = {cout_total:,.0f}")
    
    def analyze_two_stage_results(self):
        """Analyse détaillée des résultats pour TOUS LES SCÉNARIOS"""
        if not self.results:
            return
        
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        
        first_stage = self.results['first_stage']
        second_stage = self.results['second_stage']
        
        self.scenario_analysis = {}
        
        # ANALYSER CHAQUE SCÉNARIO INDIVIDUELLEMENT
        for s in range(S):
            scenario = self.scenarios[s]
            
            scenario_data = {
                'scenario_id': s + 1,
                'probability': scenario['probability'],
                'stochastic_realizations': {
                    'rework_rates_pct': {ref: rate * 100 for ref, rate in scenario['rework_rates'].items()},
                    'capacities': scenario['capacities'],
                    'alpha_rework': scenario['alpha_rework'],
                    'beta_factor': scenario['beta_factor']
                },
                'first_stage_decisions': {
                    'sequencement': first_stage['sequencement']
                },
                'second_stage_decisions': {},
                'performance_metrics': {},
                'shift_details': {}
            }
            
            # Décisions de deuxième étape pour ce scénario
            production_scenario = {}
            penuries_scenario = {}
            
            for i in R:
                total_prod_ref = 0
                for t in range(T):
                    prod = second_stage['production'][(s, i, t)]
                    total_prod_ref += prod
                    if prod > 0:
                        if i not in production_scenario:
                            production_scenario[i] = {}
                        production_scenario[i][t] = prod
                
                penuries_scenario[i] = second_stage['penuries'][(s, i)]
            
            scenario_data['second_stage_decisions'] = {
                'production': production_scenario,
                'penuries': penuries_scenario
            }
            
            # Détails par shift pour ce scénario
            for t in range(T):
                shift_data = {
                    'sequencement': first_stage['sequencement'].get(t, {}),
                    'production': {},
                    'capacity_used': 0,
                    'capacity_available': scenario['capacities'][t],
                    'utilization_pct': 0
                }
                
                capacity_used = 0
                for i in R:
                    prod = second_stage['production'][(s, i, t)]
                    if prod > 0:
                        shift_data['production'][i] = prod
                        rework_rate = scenario['rework_rates'][i]
                        beta_factor = scenario['beta_factor']
                        capacity_used += prod * (1 + beta_factor * rework_rate)
                
                shift_data['capacity_used'] = capacity_used
                shift_data['utilization_pct'] = (capacity_used / shift_data['capacity_available']) * 100 if shift_data['capacity_available'] > 0 else 0
                
                scenario_data['shift_details'][t] = shift_data
            
            # Métriques de performance pour ce scénario
            total_production_utile = 0
            total_demande = sum(params['EDI'].values())
            total_penuries = sum(penuries_scenario.values())
            
            for i in R:
                for t in range(T):
                    prod = second_stage['production'][(s, i, t)]
                    rework_rate = scenario['rework_rates'][i]
                    alpha = scenario['alpha_rework']
                    prod_utile = prod * (1 - rework_rate) + prod * rework_rate * alpha
                    total_production_utile += prod_utile
            
            satisfaction_rate = (total_production_utile / total_demande) * 100 if total_demande > 0 else 0
            
            total_capacity_used = sum([shift_data['capacity_used'] for shift_data in scenario_data['shift_details'].values()])
            total_capacity_available = sum([shift_data['capacity_available'] for shift_data in scenario_data['shift_details'].values()])
            utilization_rate = (total_capacity_used / total_capacity_available) * 100 if total_capacity_available > 0 else 0
            
            # Calcul du coût pour ce scénario
            cout_production = sum([second_stage['production'][(s, i, t)] * 20 for i in R for t in range(T)])
            cout_penuries = total_penuries * params.get('penalite_penurie', 1000)
            cout_total_scenario = cout_production + cout_penuries
            
            scenario_data['performance_metrics'] = {
                'satisfaction_rate': satisfaction_rate,
                'utilization_rate': utilization_rate,
                'total_penuries': total_penuries,
                'total_production_utile': total_production_utile,
                'cout_production': cout_production,
                'cout_penuries': cout_penuries,
                'cout_total': cout_total_scenario
            }
            
            self.scenario_analysis[s] = scenario_data
        
        st.success(f"✅ Analyse terminée pour {len(self.scenario_analysis)} scénarios")

class IntegratedTwoStageSystem:
    """Système intégré avec programmation stochastique à deux étapes"""
    
    def __init__(self):
        self.predictor = None
        self.planner = None
        self.predicted_rework_rate = None
        self.integration_results = {}
    
    def setup_prediction_system(self, data):
        """Configure le système de prédiction"""
        self.predictor = MultiPosteDefectPredictor()
        self.predictor.original_data = data.copy()
        results, postes = self.predictor.train_all_postes(data, search_method='grid')
        return True
    
    def make_prediction_for_planning(self, jour, volume, method='moyenne_ponderee'):
        """Fait une prédiction pour alimenter le modèle stochastique"""
        if self.predictor is None:
            raise ValueError("Système de prédiction non configuré")
        
        prediction = self.predictor.predict_single_scenario(jour, volume)
        self.predicted_rework_rate = prediction['taux_rework_chaine'][method]
        
        return {
            'prediction_details': prediction,
            'rework_rate_for_stochastic': self.predicted_rework_rate,
            'method_used': method
        }
    
    def setup_two_stage_planning(self, S: int, T: int, mean_capacity: float,
                                rework_std: float = 0.01, capacity_std_pct: float = 25.0,
                                base_demands: List[float] = None, **other_params):
        """Configure le modèle stochastique à deux étapes"""
        if self.predicted_rework_rate is None:
            raise ValueError("Aucune prédiction disponible")
        
        self.planner = TwoStageStochasticPlanner()
        
        R = [f'REF_{i+1:02d}' for i in range(10)]
        
        self.planner.generate_stochastic_scenarios(
            S=S, T=T, R=R,
            predicted_rework_rate=self.predicted_rework_rate,
            mean_capacity=mean_capacity,
            rework_std=rework_std,
            capacity_std_pct=capacity_std_pct,
            base_demands=base_demands,
            **other_params
        )
        
        return True
    
    def run_two_stage_optimization(self, time_limit=300):
        """Lance l'optimisation stochastique à deux étapes"""
        if self.planner is None:
            raise ValueError("Modèle stochastique non configuré")
        
        try:
            st.info("🏗️ Création du modèle à deux étapes...")
            self.planner.create_two_stage_model()
            
            st.info("📋 Ajout des contraintes...")
            self.planner.add_two_stage_constraints()
            
            st.info("🎯 Configuration de la fonction objectif...")
            self.planner.set_two_stage_objective()
            
            st.info("🔄 Lancement de l'optimisation...")
            success = self.planner.solve_two_stage_model(time_limit=time_limit)
            
            if success:
                st.info("📊 Analyse des résultats...")
                self.planner.analyze_two_stage_results()
                
                self.integration_results = {
                    'predicted_rework_rate': self.predicted_rework_rate,
                    'two_stage_results': self.planner.results,
                    'scenario_analysis': self.planner.scenario_analysis,
                    'parameters': self.planner.parameters,
                    'scenarios': self.planner.scenarios
                }
                
                return self.integration_results
            else:
                return None
                
        except Exception as e:
            st.error(f"❌ Erreur: {e}")
            return None

# Fonctions d'interface Streamlit

def create_header():
    """Créer l'en-tête"""
    st.markdown("""
    <div class="main-header">
        <h1>🎲 Programmation Stochastique à Deux Étapes</h1>
        <p>YAZAKI × ENSAM - Variables Aléatoires: Capacité & Taux Rework | Alpha/Beta FIXES</p>
    </div>
    """, unsafe_allow_html=True)

def create_demo_data(n_samples=100):
    """Crée des données de démonstration"""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        jour = np.random.randint(1, 8)
        volume = np.random.normal(1000, 200)
        volume = max(100, volume)
        
        # Défauts corrélés au volume et au jour
        defaut_poste1 = volume * 0.02 + jour * 0.5 + np.random.normal(0, 2)
        defaut_poste2 = volume * 0.015 + jour * 0.3 + np.random.normal(0, 1.5)
        defaut_poste3 = volume * 0.025 + jour * 0.4 + np.random.normal(0, 2.5)
        
        data.append({
            'Jour': jour,
            'Volume_production': max(0, volume),
            'Poste1_defauts': max(0, defaut_poste1),
            'Poste2_defauts': max(0, defaut_poste2),
            'Poste3_defauts': max(0, defaut_poste3)
        })
    
    return pd.DataFrame(data)

def load_data_section():
    """Section de chargement des données"""
    st.header("📊 Chargement des Données")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Chargez votre fichier Excel",
            type=['xlsx', 'xls'],
            help="Colonnes requises: Jour, Volume_production, Colonnes_defauts"
        )
    
    with col2:
        if st.button("📝 Données Démo", use_container_width=True):
            demo_data = create_demo_data()
            st.success(f"✅ Données générées: {len(demo_data)} lignes")
            return demo_data
    
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.success(f"✅ Données chargées: {len(data)} lignes, {len(data.columns)} colonnes")
            
            with st.expander("👀 Aperçu"):
                st.dataframe(data.head())
            
            return data
        except Exception as e:
            st.error(f"❌ Erreur: {e}")
    
    return None

def prediction_section(system, data):
    """Section de prédiction"""
    st.header("🔮 Prédiction de Défauts avec ML")
    
    if data is None:
        st.warning("⚠️ Chargez d'abord des données")
        return False
    
    with st.spinner("🧠 Entraînement des modèles ML (GridSearchCV/RandomizedSearchCV)..."):
        success = system.setup_prediction_system(data)
    
    if success:
        st.success("✅ Modèles ML entraînés avec GridSearchCV!")
        return True
    else:
        st.error("❌ Échec de l'entraînement")
        return False

def new_prediction_section(system):
    """Section de nouvelle prédiction"""
    st.header("🎯 Prédiction pour Modèle Stochastique")
    
    if system.predictor is None:
        st.warning("⚠️ Configurez d'abord le système de prédiction")
        return None
    
    col1, col2 = st.columns(2)
    
    with col1:
        jour = st.selectbox(
            "Jour de la semaine",
            options=[1, 2, 3, 4, 5, 6, 7],
            format_func=lambda x: ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][x-1],
            index=2
        )
    
    with col2:
        volume = st.number_input(
            "Volume de production",
            min_value=1, max_value=5000, value=1200, step=50
        )
    
    # Méthode d'agrégation
    method = st.selectbox(
        "Méthode d'agrégation",
        options=['moyenne_ponderee', 'moyenne', 'max', 'somme'],
        format_func=lambda x: {
            'moyenne_ponderee': 'Moyenne Pondérée (Recommandé)',
            'moyenne': 'Moyenne Simple',
            'max': 'Maximum',
            'somme': 'Somme'
        }[x],
        index=0
    )
    
    if st.button("🔮 Prédire Taux Rework", use_container_width=True):
        try:
            prediction = system.make_prediction_for_planning(jour, volume, method=method)
            
            st.subheader("📊 Résultat de Prédiction ML")
            
            pred_details = prediction['prediction_details']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Taux Rework Prédit", f"{prediction['rework_rate_for_stochastic']:.3f}%")
            
            with col2:
                st.metric("Volume Analysé", f"{volume:,}")
            
            with col3:
                jour_name = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][jour-1]
                st.metric("Jour", jour_name)
            
            with col4:
                st.metric("Méthode", method.replace('_', ' ').title())
            
            # Détails par poste
            st.subheader("📋 Détails Prédictions par Poste")
            
            detail_data = []
            for poste, defauts in pred_details['predictions_postes'].items():
                taux_poste = pred_details['taux_rework_postes'][poste]
                model_name = system.predictor.best_model_names.get(poste, 'N/A')
                
                detail_data.append({
                    'Poste': poste,
                    'Modèle ML': model_name,
                    'Défauts Prédits': round(defauts, 2),
                    'Taux Rework (%)': round(taux_poste, 3)
                })
            
            df_detail = pd.DataFrame(detail_data)
            st.dataframe(df_detail, use_container_width=True)
            
            st.info("✅ Ce taux sera utilisé comme moyenne μ de la distribution stochastique N(μ, 0.01)")
            
            return prediction
            
        except Exception as e:
            st.error(f"❌ Erreur: {e}")
    
    return None

def two_stage_planning_section(system, prediction_result):
    """Section de planification stochastique à deux étapes"""
    st.header("🎲 Programmation Stochastique à Deux Étapes")
    
    if prediction_result is None:
        st.warning("⚠️ Effectuez d'abord une prédiction")
        return None
    
    taux_predit = prediction_result['rework_rate_for_stochastic']
    st.info(f"📊 Taux de rework prédit: {taux_predit:.3f}% (utilisé comme moyenne μ)")
    
    # Information sur le modèle à deux étapes
    st.markdown("""
    <div class="stochastic-info">
        <h4>🔬 Modèle Stochastique à Deux Étapes CORRIGÉ:</h4>
        <ul>
            <li><strong>1ère Étape:</strong> Décisions avant réalisation (séquencement commun)</li>
            <li><strong>2ème Étape:</strong> Décisions après réalisation (production par scénario)</li>
            <li><strong>Variables Aléatoires:</strong> Taux Rework ~ N(μ=prédit, σ=0.01), Capacité ~ N(μ, ±25%)</li>
            <li><strong>NOUVEAU:</strong> Alpha et Beta sont maintenant FIXES (pas de variation aléatoire)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("⚙️ Configuration du Modèle Stochastique")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        S = st.number_input("Nombre de scénarios", min_value=2, max_value=8, value=3)
        T = st.number_input("Nombre de shifts", min_value=1, max_value=5, value=3)
    
    with col2:
        mean_capacity = st.number_input("Capacité moyenne", min_value=50, max_value=1000, value=200)
        capacity_std_pct = st.slider("Écart-type capacité (±%)", 5, 50, 25)
        
        cap_min = mean_capacity * (1 - capacity_std_pct/100)
        cap_max = mean_capacity * (1 + capacity_std_pct/100)
        st.info(f"Intervalle: [{cap_min:.0f}, {cap_max:.0f}]")
    
    with col3:
        rework_std = st.number_input("Écart-type taux rework", 0.001, 0.05, 0.01, 0.001, format="%.3f")
        
        ic_lower = max(0, taux_predit - 2*rework_std*100)
        ic_upper = min(30, taux_predit + 2*rework_std*100)
        st.info(f"IC 95%: [{ic_lower:.3f}%, {ic_upper:.3f}%]")
        
        time_limit = st.number_input("Limite temps (sec)", 30, 600, 300)
    
    # Paramètres FIXES (pas aléatoires)
    with st.expander("🔧 Paramètres FIXES (Alpha/Beta)"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alpha_rework = st.slider("Alpha rework (FIXE)", 0.5, 1.0, 0.8, 0.05)
            st.info("🔒 Même valeur pour tous les scénarios")
        
        with col2:
            beta_factor = st.slider("Beta factor (FIXE)", 1.0, 2.0, 1.2, 0.1)
            st.info("🔒 Même valeur pour tous les scénarios")
        
        with col3:
            penalite_penurie = st.number_input("Pénalité pénurie", 100, 10000, 1000)
            m = st.number_input("Production minimale", 1, 20, 5)
    
    if st.button("🚀 Lancer Optimisation Stochastique", use_container_width=True):
        with st.spinner("🎲 Génération des scénarios et optimisation..."):
            try:
                # Configuration
                success_setup = system.setup_two_stage_planning(
                    S=S, T=T,
                    mean_capacity=mean_capacity,
                    rework_std=rework_std,
                    capacity_std_pct=capacity_std_pct,
                    alpha_rework=alpha_rework,   # FIXE
                    beta_factor=beta_factor,     # FIXE
                    penalite_penurie=penalite_penurie,
                    m=m
                )
                
                if success_setup:
                    # Optimisation
                    results = system.run_two_stage_optimization(time_limit=time_limit)
                    
                    if results:
                        st.success("✅ Optimisation réussie!")
                        return results
                    else:
                        st.error("❌ Échec de l'optimisation")
                else:
                    st.error("❌ Erreur de configuration")
                    
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
    
    return None

def display_all_scenarios_results(system, results):
    """Affiche les résultats COMPLETS de TOUS les scénarios"""
    st.header("📊 Résultats COMPLETS - Tous les Scénarios")
    
    if not results:
        st.warning("⚠️ Aucun résultat disponible")
        return
    
    # Métriques globales
    st.subheader("🎯 Métriques Globales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Coût Total (Espérance)", f"{results['two_stage_results']['cout_total_esperance']:,.0f}€")
    
    with col2:
        st.metric("Taux Rework Utilisé", f"{results['predicted_rework_rate']:.3f}%")
    
    with col3:
        st.metric("Scénarios Analysés", len(results['scenario_analysis']))
    
    with col4:
        demande_totale = sum(results['parameters']['EDI'].values())
        st.metric("Demande Totale", f"{demande_totale:,}")
    
    # Décisions de première étape (communes)
    st.subheader("🎯 Décisions de Première Étape (Communes à tous les scénarios)")
    
    first_stage = results['two_stage_results']['first_stage']
    sequencement = first_stage['sequencement']
    
    st.write("**📋 Séquencement Optimal (appliqué à tous les scénarios):**")
    
    seq_data = []
    for t, sequence in sequencement.items():
        if sequence:
            ordre = ' → '.join([sequence.get(j, f'Pos{j+1}') for j in sorted(sequence.keys())])
            seq_data.append({
                'Shift': f'Shift {t+1}',
                'Séquence': ordre,
                'Nb Références': len(sequence)
            })
        else:
            seq_data.append({
                'Shift': f'Shift {t+1}',
                'Séquence': 'Aucune production',
                'Nb Références': 0
            })
    
    df_seq = pd.DataFrame(seq_data)
    st.dataframe(df_seq, use_container_width=True)
    
    # RÉSULTATS DÉTAILLÉS PAR SCÉNARIO
    st.subheader("🎲 Résultats Détaillés par Scénario")
    
    # Comparaison rapide
    scenario_comparison = []
    for s, analysis in results['scenario_analysis'].items():
        metrics = analysis['performance_metrics']
        realizations = analysis['stochastic_realizations']
        
        scenario_comparison.append({
            'Scénario': f'S{s+1}',
            'Satisfaction (%)': f"{metrics['satisfaction_rate']:.1f}",
            'Utilisation (%)': f"{metrics['utilization_rate']:.1f}",
            'Pénuries': f"{metrics['total_penuries']:.1f}",
            'Coût Total': f"{metrics['cout_total']:,.0f}€",
            'Taux Rework Moyen (%)': f"{np.mean(list(realizations['rework_rates_pct'].values())):.3f}",
            'Capacité Moyenne': f"{np.mean(list(realizations['capacities'].values())):.0f}",
            'Alpha (FIXE)': f"{realizations['alpha_rework']:.3f}",
            'Beta (FIXE)': f"{realizations['beta_factor']:.2f}"
        })
    
    df_comparison = pd.DataFrame(scenario_comparison)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Sélection de scénario pour analyse détaillée
    st.subheader("🔍 Analyse Détaillée par Scénario")
    
    selected_scenario = st.selectbox(
        "Choisir un scénario à analyser en détail:",
        options=[f"S{s+1}" for s in range(len(results['scenario_analysis']))],
        index=0
    )
    
    scenario_id = int(selected_scenario.replace('S', '')) - 1
    selected_analysis = results['scenario_analysis'][scenario_id]
    
    # Réalisations des variables aléatoires pour ce scénario
    st.write(f"**🎲 Réalisations des Variables Aléatoires - {selected_scenario}:**")
    
    realizations = selected_analysis['stochastic_realizations']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rework_mean = np.mean(list(realizations['rework_rates_pct'].values()))
        st.metric("Taux Rework Moyen", f"{rework_mean:.3f}%")
    
    with col2:
        capacity_mean = np.mean(list(realizations['capacities'].values()))
        st.metric("Capacité Moyenne", f"{capacity_mean:.0f}")
    
    with col3:
        st.metric("Alpha (FIXE)", f"{realizations['alpha_rework']:.3f}")
    
    with col4:
        st.metric("Beta (FIXE)", f"{realizations['beta_factor']:.2f}")
    
    # Production détaillée par shift pour ce scénario
    st.write(f"**⚡ Décisions de Production - {selected_scenario}:**")
    
    shift_details = selected_analysis['shift_details']
    
    for t, shift_data in shift_details.items():
        st.write(f"**🔄 Shift {t+1}:**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if shift_data['production']:
                st.write("Production planifiée:")
                for ref, qty in shift_data['production'].items():
                    st.write(f"  • {ref}: {qty:.0f} unités")
            else:
                st.write("Aucune production planifiée")
        
        with col2:
            st.metric("Capacité Utilisée", f"{shift_data['capacity_used']:.0f}/{shift_data['capacity_available']:.0f}")
            st.metric("Utilisation", f"{shift_data['utilization_pct']:.1f}%")
    
    # Métriques de performance pour ce scénario
    metrics = selected_analysis['performance_metrics']
    
    st.write(f"**📊 Performances - {selected_scenario}:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Satisfaction", f"{metrics['satisfaction_rate']:.1f}%")
    
    with col2:
        st.metric("Utilisation Globale", f"{metrics['utilization_rate']:.1f}%")
    
    with col3:
        st.metric("Pénuries", f"{metrics['total_penuries']:.1f}")
    
    with col4:
        st.metric("Coût Total", f"{metrics['cout_total']:,.0f}€")
    
    # Graphiques comparatifs
    st.subheader("📈 Comparaisons Graphiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Satisfaction par scénario
        satisfactions = [analysis['performance_metrics']['satisfaction_rate'] 
                        for analysis in results['scenario_analysis'].values()]
        scenarios = [f"S{s+1}" for s in range(len(satisfactions))]
        
        fig = px.bar(x=scenarios, y=satisfactions, 
                     title="Taux de Satisfaction par Scénario",
                     labels={'x': 'Scénario', 'y': 'Satisfaction (%)'})
        fig.update_traces(text=[f"{s:.1f}%" for s in satisfactions], textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Coût par scénario
        couts = [analysis['performance_metrics']['cout_total'] 
                for analysis in results['scenario_analysis'].values()]
        
        fig = px.bar(x=scenarios, y=couts,
                     title="Coût Total par Scénario", 
                     labels={'x': 'Scénario', 'y': 'Coût (€)'})
        fig.update_traces(text=[f"{c:,.0f}€" for c in couts], textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Variables aléatoires
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution taux rework
        rework_rates = [np.mean(list(analysis['stochastic_realizations']['rework_rates_pct'].values()))
                       for analysis in results['scenario_analysis'].values()]
        
        fig = px.histogram(x=rework_rates, nbins=max(3, len(rework_rates)//2),
                          title="Distribution Taux Rework Réalisés",
                          labels={'x': 'Taux Rework (%)', 'y': 'Fréquence'})
        fig.add_vline(x=results['predicted_rework_rate'], line_dash="dash", line_color="red",
                     annotation_text=f"Prédiction: {results['predicted_rework_rate']:.3f}%")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution capacités
        capacities = [np.mean(list(analysis['stochastic_realizations']['capacities'].values()))
                     for analysis in results['scenario_analysis'].values()]
        
        fig = px.histogram(x=capacities, nbins=max(3, len(capacities)//2),
                          title="Distribution Capacités Réalisées",
                          labels={'x': 'Capacité', 'y': 'Fréquence'})
        fig.add_vline(x=results['parameters']['mean_capacity'], line_dash="dash", line_color="blue",
                     annotation_text=f"Moyenne: {results['parameters']['mean_capacity']}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse de robustesse
    st.subheader("🛡️ Analyse de Robustesse")
    
    variance_satisfaction = np.var([analysis['performance_metrics']['satisfaction_rate'] 
                                   for analysis in results['scenario_analysis'].values()])
    variance_cout = np.var([analysis['performance_metrics']['cout_total'] 
                           for analysis in results['scenario_analysis'].values()])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Variance Satisfaction", f"{variance_satisfaction:.3f}")
        if variance_satisfaction < 1.0:
            st.success("✅ Solution robuste")
        else:
            st.warning("⚠️ Solution sensible")
    
    with col2:
        st.metric("Variance Coût", f"{variance_cout:,.0f}")
        cv_cout = np.std([analysis['performance_metrics']['cout_total'] 
                         for analysis in results['scenario_analysis'].values()]) / np.mean([analysis['performance_metrics']['cout_total'] 
                         for analysis in results['scenario_analysis'].values()]) * 100
        st.metric("CV Coût (%)", f"{cv_cout:.2f}")
    
    with col3:
        best_scenario = max(results['scenario_analysis'].items(), 
                           key=lambda x: x[1]['performance_metrics']['satisfaction_rate'])
        worst_scenario = min(results['scenario_analysis'].items(), 
                            key=lambda x: x[1]['performance_metrics']['satisfaction_rate'])
        
        st.metric("Meilleur Scénario", f"S{best_scenario[0]+1}")
        st.metric("Pire Scénario", f"S{worst_scenario[0]+1}")

def main():
    """Application principale"""
    create_header()
    
    # Initialisation de la session
    if 'two_stage_system' not in st.session_state:
        st.session_state.two_stage_system = IntegratedTwoStageSystem()
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'prediction_trained' not in st.session_state:
        st.session_state.prediction_trained = False
    
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    if 'two_stage_results' not in st.session_state:
        st.session_state.two_stage_results = None
    
    # Sidebar
    with st.sidebar:
        st.header("🧭 Navigation")
        
        step = st.radio("Étapes:", [
            "📊 1. Données",
            "🔮 2. Prédiction ML", 
            "🎯 3. Prédiction Stochastique",
            "🎲 4. Programmation 2-Étapes",
            "📈 5. Résultats Complets"
        ])
        
        st.markdown("---")
        st.header("📋 État")
        
        states = [
            ("📊 Données", st.session_state.data is not None),
            ("🔮 ML Entraîné", st.session_state.prediction_trained),
            ("🎯 Prédiction", st.session_state.prediction_result is not None),
            ("🎲 Optimisé", st.session_state.two_stage_results is not None)
        ]
        
        for label, status in states:
            st.write(f"{label}: {'✅' if status else '❌'}")
        
        st.markdown("---")
        
        if st.button("🔄 Reset", use_container_width=True):
            for key in ['two_stage_system', 'data', 'prediction_trained', 
                       'prediction_result', 'two_stage_results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        st.header("ℹ️ Modèle CORRIGÉ")
        st.markdown("""
        **🎲 Programmation Stochastique à Deux Étapes**
        
        **Variables Aléatoires:**
        - Taux Rework ~ N(μ=ML, σ=0.01)
        - Capacité ~ N(μ, ±25%)
        
        **FIXES (pas aléatoires):**
        - Alpha rework (efficacité)
        - Beta factor (surcharge)
        
        **1ère Étape:** Séquencement commun
        
        **2ème Étape:** Production par scénario
        """)
    
    # Navigation
    if step == "📊 1. Données":
        data = load_data_section()
        if data is not None:
            st.session_state.data = data
            st.session_state.prediction_trained = False
            st.session_state.prediction_result = None
            st.session_state.two_stage_results = None
    
    elif step == "🔮 2. Prédiction ML":
        if st.session_state.data is not None:
            success = prediction_section(st.session_state.two_stage_system, st.session_state.data)
            if success:
                st.session_state.prediction_trained = True
        else:
            st.warning("⚠️ Chargez d'abord des données")
    
    elif step == "🎯 3. Prédiction Stochastique":
        if st.session_state.prediction_trained:
            prediction = new_prediction_section(st.session_state.two_stage_system)
            if prediction is not None:
                st.session_state.prediction_result = prediction
                st.session_state.two_stage_results = None
        else:
            st.warning("⚠️ Entraînez d'abord les modèles ML")
    
    elif step == "🎲 4. Programmation 2-Étapes":
        if st.session_state.prediction_result is not None:
            results = two_stage_planning_section(
                st.session_state.two_stage_system, 
                st.session_state.prediction_result
            )
            if results is not None:
                st.session_state.two_stage_results = results
        else:
            st.warning("⚠️ Effectuez d'abord une prédiction")
    
    elif step == "📈 5. Résultats Complets":
        if st.session_state.two_stage_results is not None:
            display_all_scenarios_results(
                st.session_state.two_stage_system,
                st.session_state.two_stage_results
            )
        else:
            st.warning("⚠️ Lancez d'abord l'optimisation")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <div style='font-weight: bold; margin-bottom: 10px;'>
            🎲 Programmation Stochastique à Deux Étapes CORRIGÉE - YAZAKI × ENSAM
        </div>
        <div style='font-size: 14px;'>
            ✅ Variables Aléatoires: Taux Rework ~ N(μ=ML, σ=0.01) | Capacité ~ N(μ, ±25%) 
            <br>🔒 Alpha/Beta FIXES | 📊 Tous les Scénarios Affichés
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
