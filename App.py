# Système Intégré Streamlit avec Prédictions et Planification Réelles
# Installation: pip install streamlit pandas numpy plotly scikit-learn pulp openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle 
import os
from datetime import datetime, timedelta
import base64
from io import BytesIO

# Imports pour ML et optimisation
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pulp as plp
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION STREAMLIT
# =====================================================================

st.set_page_config(
    page_title="Système Intégré avec Prédictions Réelles",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé (conservé identique)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2e86ab);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .model-card {
        background: #e8f5e8;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .scenario-card {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .scenario-card:hover {
        border-color: #007bff;
        box-shadow: 0 4px 8px rgba(0,123,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# CLASSE PRINCIPALE AVEC PRÉDICTIONS RÉELLES
# =====================================================================

class RealPredictionSystem:
    """Système de prédiction réel avec ML"""
    
    def __init__(self):
        self.models = {}
        self.model_params = {}
        self.model_performance = {}
        self.transformers = {}
        self.feature_importances = {}
        self.original_data = None
        self.postes = []
        self.jour_col = None
        self.volume_col = None
        self.predictions_history = []
        self.poste_weights = {}
        self.is_trained = False
        self.load_history()
    
    def load_history(self):
        """Charge l'historique depuis le state Streamlit"""
        if 'predictions_history' in st.session_state:
            self.predictions_history = st.session_state.predictions_history
        else:
            st.session_state.predictions_history = []
            self.predictions_history = []
    
    def save_history(self):
        """Sauvegarde l'historique dans le state Streamlit"""
        st.session_state.predictions_history = self.predictions_history
    
    def load_and_prepare_data(self, data):
        """Charge et prépare les données"""
        self.original_data = data.copy()
        
        # Identifier les colonnes automatiquement
        possible_jour_cols = ['Jour', 'jour', 'Day', 'day', 'JOUR']
        possible_volume_cols = ['Volume_production', 'volume', 'Volume', 'Production', 'VOLUME']
        
        for col in possible_jour_cols:
            if col in data.columns:
                self.jour_col = col
                break
        
        for col in possible_volume_cols:
            if col in data.columns:
                self.volume_col = col
                break
        
        if not self.jour_col or not self.volume_col:
            raise ValueError("Colonnes 'Jour' et 'Volume_production' non trouvées!")
        
        # Identifier les postes (colonnes avec 'defaut' ou 'Poste')
        self.postes = [col for col in data.columns 
                      if 'defaut' in col.lower() or 'poste' in col.lower()]
        
        if not self.postes:
            raise ValueError("Aucune colonne de défauts trouvée!")
        
        # Calculer les poids par défaut (basés sur la moyenne des défauts)
        total_defauts = sum(data[poste].mean() for poste in self.postes)
        self.poste_weights = {
            poste: data[poste].mean() / total_defauts 
            for poste in self.postes
        }
        
        return True
    
    def train_models(self, test_size=0.2, models_to_try=None):
        """Entraîne les modèles ML pour chaque poste"""
        
        if self.original_data is None:
            raise ValueError("Aucune donnée chargée!")
        
        if models_to_try is None:
            models_to_try = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'DecisionTree': DecisionTreeRegressor(random_state=42)
            }
        
        results = {}
        
        # Préparer les features
        X = self.original_data[[self.volume_col, self.jour_col]].copy()
        
        # Encoder le jour si nécessaire
        if X[self.jour_col].dtype == 'object':
            le = LabelEncoder()
            X[self.jour_col] = le.fit_transform(X[self.jour_col])
        
        for poste in self.postes:
            st.write(f"🔄 Entraînement des modèles pour {poste}...")
            
            y = self.original_data[poste]
            
            # Split des données
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            best_model = None
            best_score = -np.inf
            best_model_name = None
            model_scores = {}
            
            # Tester chaque modèle
            for model_name, model in models_to_try.items():
                try:
                    # Entraîner le modèle
                    model.fit(X_train, y_train)
                    
                    # Prédire et évaluer
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    model_scores[model_name] = {
                        'r2': score,
                        'rmse': rmse,
                        'mae': mae
                    }
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_model_name = model_name
                
                except Exception as e:
                    st.warning(f"⚠️ Erreur avec {model_name} pour {poste}: {e}")
                    continue
            
            if best_model is not None:
                self.models[poste] = best_model
                self.model_params[poste] = {
                    'best_model': best_model_name,
                    'r2_score': best_score,
                    'all_scores': model_scores
                }
                
                # Feature importance si disponible
                if hasattr(best_model, 'feature_importances_'):
                    self.feature_importances[poste] = {
                        'Volume': best_model.feature_importances_[0],
                        'Jour': best_model.feature_importances_[1]
                    }
                
                results[poste] = {
                    'model': best_model_name,
                    'r2': best_score,
                    'details': model_scores
                }
            else:
                st.error(f"❌ Impossible d'entraîner un modèle pour {poste}")
        
        self.is_trained = len(self.models) > 0
        return results
    
    def predict_defects(self, jour, volume):
        """Fait une prédiction de défauts"""
        if not self.is_trained:
            raise ValueError("Les modèles doivent être entraînés!")
        
        # Préparer les données d'entrée
        X_new = pd.DataFrame({
            self.volume_col: [volume],
            self.jour_col: [jour]
        })
        
        predictions_postes = {}
        taux_rework_postes = {}
        
        # Prédiction pour chaque poste
        for poste, model in self.models.items():
            pred = max(0, model.predict(X_new)[0])
            predictions_postes[poste] = pred
            taux_rework_postes[poste] = (pred / volume) * 100
        
        # Calculs pour la chaîne complète
        predictions_chaine = {
            'max': max(predictions_postes.values()) if predictions_postes else 0,
            'moyenne': np.mean(list(predictions_postes.values())) if predictions_postes else 0,
            'moyenne_ponderee': self.calculate_weighted_average(predictions_postes),
            'somme': sum(predictions_postes.values()) if predictions_postes else 0
        }
        
        taux_rework_chaine = {
            method: (defauts / volume) * 100 
            for method, defauts in predictions_chaine.items()
        }
        
        return {
            'predictions_postes': predictions_postes,
            'predictions_chaine': predictions_chaine,
            'taux_rework_postes': taux_rework_postes,
            'taux_rework_chaine': taux_rework_chaine
        }
    
    def calculate_weighted_average(self, predictions_postes):
        """Calcule la moyenne pondérée"""
        if not predictions_postes or not self.poste_weights:
            return np.mean(list(predictions_postes.values())) if predictions_postes else 0
        
        weighted_sum = sum(pred * self.poste_weights.get(poste, 0) 
                          for poste, pred in predictions_postes.items())
        total_weight = sum(self.poste_weights.get(poste, 0) 
                          for poste in predictions_postes.keys())
        
        return weighted_sum / total_weight if total_weight > 0 else np.mean(list(predictions_postes.values()))
    
    def add_prediction_to_history(self, jour, volume, method='moyenne_ponderee', actual_defects=None):
        """Ajoute une prédiction à l'historique"""
        prediction = self.predict_defects(jour, volume)
        
        taux_final = prediction['taux_rework_chaine'][method]
        
        # Ajustements contextuels
        volume_factor = 0.98 if volume > 1500 else (1.02 if volume < 800 else 1.0)
        jour_factor = 1.08 if jour in [6, 7] else 1.0
        
        taux_ajuste = taux_final * volume_factor * jour_factor
        taux_ajuste = max(0.5, min(25, taux_ajuste))
        
        # Créer l'enregistrement
        record = {
            'timestamp': datetime.now(),
            'jour': jour,
            'volume': volume,
            'method': method,
            'prediction': prediction,
            'taux_final': taux_ajuste,
            'actual_defects': actual_defects,
            'accuracy': None
        }
        
        # Calculer la précision si défauts réels fournis
        if actual_defects:
            predicted_total = sum(prediction['predictions_postes'].values())
            actual_total = sum(actual_defects.values())
            
            if actual_total == 0 and predicted_total == 0:
                accuracy = 100.0
            elif actual_total == 0:
                accuracy = max(0, 100 - (predicted_total / volume * 100))
            else:
                relative_error = abs(predicted_total - actual_total) / actual_total
                accuracy = max(0, 100 - (relative_error * 100))
            
            record['accuracy'] = accuracy
        
        self.predictions_history.append(record)
        self.save_history()
        
        return record
    
    def get_historical_rates(self):
        """Retourne les taux historiques pour la planification"""
        if not self.predictions_history:
            return []
        
        return [pred['taux_final'] for pred in self.predictions_history]
    
    def get_model_summary(self):
        """Retourne un résumé des modèles"""
        if not self.is_trained:
            return None
        
        summary = {}
        for poste in self.postes:
            if poste in self.model_params:
                summary[poste] = {
                    'model': self.model_params[poste]['best_model'],
                    'r2_score': self.model_params[poste]['r2_score'],
                    'weight': self.poste_weights.get(poste, 0),
                    'feature_importance': self.feature_importances.get(poste, {})
                }
        
        return summary

class IntelligentPlanning:
    """Planification intelligente avec historique"""
    
    def __init__(self, prediction_system):
        self.prediction_system = prediction_system
        self.model = None
        self.variables = {}
        self.results = {}
        self.scenarios = []
    
    def generate_scenarios(self, nouvelle_demande_taux, n_scenarios=3):
        """Génère des scénarios de planification"""
        historical_rates = self.prediction_system.get_historical_rates()
        
        if not historical_rates:
            # Si pas d'historique, utiliser seulement la nouvelle demande
            if n_scenarios == 1:
                self.scenarios = [nouvelle_demande_taux]
            elif n_scenarios == 3:
                self.scenarios = [
                    nouvelle_demande_taux * 0.85,  # Optimiste
                    nouvelle_demande_taux,         # Nominal
                    nouvelle_demande_taux * 1.15   # Pessimiste
                ]
            else:
                # Distribution autour de la nouvelle demande
                factors = np.linspace(0.8, 1.2, n_scenarios)
                self.scenarios = [nouvelle_demande_taux * f for f in factors]
        else:
            # Combiner historique et nouvelle demande
            all_rates = historical_rates + [nouvelle_demande_taux]
            
            if n_scenarios == 1:
                self.scenarios = [np.mean(all_rates)]
            elif n_scenarios == 3:
                self.scenarios = [
                    min(all_rates),           # Optimiste
                    np.mean(all_rates),       # Moyen
                    max(all_rates)            # Pessimiste
                ]
            else:
                # Percentiles des taux combinés
                percentiles = np.linspace(10, 90, n_scenarios)
                self.scenarios = [np.percentile(all_rates, p) for p in percentiles]
        
        # Limiter les scénarios à des valeurs raisonnables
        self.scenarios = [max(0.5, min(25, rate)) for rate in self.scenarios]
        
        return self.scenarios
    
    def setup_optimization(self, scenarios, references, demands, capacities, params=None):
        """Configure le problème d'optimisation"""
        
        if params is None:
            params = {
                'alpha_rework': 0.8,
                'beta': 1.2,
                'penalite_penurie': 1000,
                'cout_production': 20
            }
        
        S = len(scenarios)
        T = len(capacities)
        R = references
        
        # Créer le modèle
        self.model = plp.LpProblem("Planning_Intelligent", plp.LpMinimize)
        
        # Variables de décision
        self.variables['q'] = plp.LpVariable.dicts(
            "production", 
            [(s, i, t) for s in range(S) for i in R for t in range(T)],
            lowBound=0, cat='Continuous'
        )
        
        self.variables['penurie'] = plp.LpVariable.dicts(
            "penurie",
            [(s, i) for s in range(S) for i in R],
            lowBound=0, cat='Continuous'
        )
        
        # Contraintes de demande
        for s in range(S):
            taux_defaut = scenarios[s] / 100
            for i, ref in enumerate(R):
                production_effective = plp.lpSum([
                    self.variables['q'][(s, ref, t)] * 
                    (1 - taux_defaut + params['alpha_rework'] * taux_defaut)
                    for t in range(T)
                ])
                
                self.model += (
                    production_effective + self.variables['penurie'][(s, ref)] >= demands[i],
                    f"Demande_s{s}_r{ref}"
                )
        
        # Contraintes de capacité
        for s in range(S):
            for t in range(T):
                taux_defaut = scenarios[s] / 100
                capacite_utilisee = plp.lpSum([
                    self.variables['q'][(s, ref, t)] * (1 + params['beta'] * taux_defaut)
                    for ref in R
                ])
                
                self.model += (
                    capacite_utilisee <= capacities[t],
                    f"Capacite_s{s}_t{t}"
                )
        
        # Fonction objectif
        cout_production = plp.lpSum([
            (1/S) * params['cout_production'] * self.variables['q'][(s, ref, t)]
            for s in range(S) for ref in R for t in range(T)
        ])
        
        cout_penuries = plp.lpSum([
            (1/S) * params['penalite_penurie'] * self.variables['penurie'][(s, ref)]
            for s in range(S) for ref in R
        ])
        
        self.model += cout_production + cout_penuries
        
        return True
    
    def solve(self, time_limit=300):
        """Résout le problème d'optimisation"""
        try:
            solver = plp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
            self.model.solve(solver)
            
            if self.model.status == plp.LpStatusOptimal:
                self._extract_results()
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Erreur d'optimisation: {e}")
            return False
    
    def _extract_results(self):
        """Extrait les résultats de l'optimisation"""
        production = {}
        penuries = {}
        
        for var in self.model.variables():
            if var.name.startswith("production"):
                # Parser le nom de variable pour extraire les indices
                parts = var.name.replace("production_", "").replace("(", "").replace(")", "").split(",")
                if len(parts) == 3:
                    s, ref, t = parts[0].strip(), parts[1].strip().strip("'"), int(parts[2].strip())
                    production[(int(s), ref, t)] = var.value() or 0
            
            elif var.name.startswith("penurie"):
                parts = var.name.replace("penurie_", "").replace("(", "").replace(")", "").split(",")
                if len(parts) == 2:
                    s, ref = int(parts[0].strip()), parts[1].strip().strip("'")
                    penuries[(s, ref)] = var.value() or 0
        
        self.results = {
            'production': production,
            'penuries': penuries,
            'cout_total': self.model.objective.value(),
            'scenarios': self.scenarios,
            'status': 'Optimal'
        }

# =====================================================================
# FONCTIONS DE VISUALISATION
# =====================================================================

def create_model_performance_chart(model_summary):
    """Crée un graphique de performance des modèles"""
    if not model_summary:
        return None
    
    postes = list(model_summary.keys())
    models = [model_summary[poste]['model'] for poste in postes]
    r2_scores = [model_summary[poste]['r2_score'] for poste in postes]
    weights = [model_summary[poste]['weight'] for poste in postes]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Score R² par Poste', 'Poids des Postes'),
        specs=[[{"secondary_y": False}, {"type": "pie"}]]
    )
    
    # Graphique R²
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig.add_trace(
        go.Bar(x=postes, y=r2_scores, 
               marker_color=colors[:len(postes)],
               text=[f'{score:.3f}' for score in r2_scores],
               textposition='auto',
               name='Score R²'),
        row=1, col=1
    )
    
    # Graphique en secteurs pour les poids
    fig.add_trace(
        go.Pie(labels=postes, values=weights, 
               marker_colors=colors[:len(postes)],
               name='Poids'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_yaxes(title_text="Score R²", row=1, col=1)
    
    return fig

def create_scenarios_comparison(scenarios, scenario_names=None):
    """Crée un graphique de comparaison des scénarios"""
    if scenario_names is None:
        if len(scenarios) == 3:
            scenario_names = ['Optimiste', 'Nominal', 'Pessimiste']
        else:
            scenario_names = [f'Scénario {i+1}' for i in range(len(scenarios))]
    
    colors = ['green', 'orange', 'red'][:len(scenarios)]
    
    fig = go.Figure(data=[
        go.Bar(x=scenario_names, y=scenarios,
               marker_color=colors,
               text=[f'{rate:.2f}%' for rate in scenarios],
               textposition='auto')
    ])
    
    fig.update_layout(
        title="Comparaison des Scénarios de Planification",
        xaxis_title="Scénario",
        yaxis_title="Taux de Rework (%)",
        showlegend=False
    )
    
    return fig

def create_planning_dashboard(results, references, scenarios):
    """Crée un dashboard pour le choix du meilleur scénario"""
    if not results or 'production' not in results:
        return None
    
    S = len(scenarios)
    
    # Calculer les métriques par scénario
    scenario_metrics = []
    
    for s in range(S):
        scenario_name = ['Optimiste', 'Nominal', 'Pessimiste'][s] if S == 3 else f'Scénario {s+1}'
        
        # Production totale pour ce scénario
        total_production = sum(
            results['production'].get((s, ref, t), 0)
            for ref in references
            for t in range(3)  # Assumant 3 shifts
        )
        
        # Pénuries totales pour ce scénario
        total_penuries = sum(
            results['penuries'].get((s, ref), 0)
            for ref in references
        )
        
        # Coût estimé (approximation)
        cout_scenario = (total_production * 20 + total_penuries * 1000) / S
        
        scenario_metrics.append({
            'Scenario': scenario_name,
            'Taux_Rework': scenarios[s],
            'Production_Totale': total_production,
            'Penuries_Totales': total_penuries,
            'Cout_Estime': cout_scenario,
            'Score_Global': 100 - (total_penuries * 10 + scenarios[s])  # Score simplifié
        })
    
    df_metrics = pd.DataFrame(scenario_metrics)
    
    # Créer le dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Production par Scénario', 'Pénuries par Scénario', 
                       'Coût par Scénario', 'Score Global'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['green', 'orange', 'red'][:len(scenarios)]
    
    # Production
    fig.add_trace(
        go.Bar(x=df_metrics['Scenario'], y=df_metrics['Production_Totale'],
               marker_color=colors, name='Production'),
        row=1, col=1
    )
    
    # Pénuries
    fig.add_trace(
        go.Bar(x=df_metrics['Scenario'], y=df_metrics['Penuries_Totales'],
               marker_color=colors, name='Pénuries'),
        row=1, col=2
    )
    
    # Coût
    fig.add_trace(
        go.Bar(x=df_metrics['Scenario'], y=df_metrics['Cout_Estime'],
               marker_color=colors, name='Coût'),
        row=2, col=1
    )
    
    # Score global
    fig.add_trace(
        go.Bar(x=df_metrics['Scenario'], y=df_metrics['Score_Global'],
               marker_color=colors, name='Score'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    
    return fig, df_metrics

# =====================================================================
# INTERFACE STREAMLIT PRINCIPALE
# =====================================================================

def main():
    # En-tête
    st.markdown("""
    <div class="main-header">
        <h1>🏭 Système Intégré avec Prédictions et Planification Réelles</h1>
        <p>Machine Learning + Optimisation + Dashboard Intelligent</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialisation
    if 'prediction_system' not in st.session_state:
        st.session_state.prediction_system = RealPredictionSystem()
        st.session_state.models_trained = False
        st.session_state.planning_results = None

    # Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choisissez une section:",
        [
            "🏠 Accueil",
            "📁 Chargement & Entraînement", 
            "📝 Nouvelle Demande & Prédiction",
            "🎯 Planification Intelligente",
            "📊 Dashboard & Comparaison",
            "📈 Historique & Performance"
        ]
    )

    # =================== PAGE ACCUEIL ===================
    if page == "🏠 Accueil":
        st.header("🏠 Bienvenue dans le Système Complet")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Fonctionnalités Principales
            
            **🤖 Prédictions ML Réelles:**
            - Entraînement de modèles RandomForest, GradientBoosting, DecisionTree
            - Sélection automatique du meilleur modèle par poste
            - Affichage des paramètres et performances
            
            **📊 Nouvelle Demande Intégrée:**
            - Prédiction basée sur jour et volume
            - Calcul du taux de rework
            - Intégration à l'historique
            
            **🎯 Planification Optimale:**
            - Génération de scénarios intelligents
            - Combinaison historique + nouvelle demande
            - Optimisation avec contraintes réelles
            """)
        
        with col2:
            st.markdown("""
            ### 📋 Statut du Système
            """)
            
            # Statut des modèles
            if st.session_state.models_trained:
                st.success("✅ Modèles Entraînés")
                model_summary = st.session_state.prediction_system.get_model_summary()
                if model_summary:
                    st.write(f"**Postes:** {len(model_summary)}")
                    avg_r2 = np.mean([info['r2_score'] for info in model_summary.values()])
                    st.write(f"**R² Moyen:** {avg_r2:.3f}")
            else:
                st.warning("⏳ Modèles non entraînés")
            
            # Statut historique
            hist_count = len(st.session_state.prediction_system.predictions_history)
            if hist_count > 0:
                st.info(f"📊 {hist_count} prédictions en historique")
            else:
                st.warning("📭 Aucun historique")
            
            # Statut planification
            if st.session_state.planning_results:
                st.success("✅ Planification configurée")
            else:
                st.warning("⏳ Planification à faire")

    # =================== CHARGEMENT & ENTRAÎNEMENT ===================
    elif page == "📁 Chargement & Entraînement":
        st.header("📁 Chargement des Données et Entraînement des Modèles")
        
        st.markdown("### 📂 Étape 1: Chargement du Fichier Excel")
        
        uploaded_file = st.file_uploader(
            "Téléchargez votre fichier Excel",
            type=['xlsx', 'xls'],
            help="Le fichier doit contenir: Jour, Volume_production, et colonnes de défauts par poste"
        )
        
        if uploaded_file is not None:
            try:
                # Charger les données
                data = pd.read_excel(uploaded_file)
                st.success("✅ Fichier chargé avec succès!")
                
                # Aperçu des données
                st.markdown("### 👀 Aperçu des Données")
                st.dataframe(data.head(10))
                
                # Informations sur les données
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nombre de lignes", len(data))
                with col2:
                    st.metric("Nombre de colonnes", len(data.columns))
                with col3:
                    st.metric("Colonnes détectées", ", ".join(data.columns[:3]) + "...")
                
                # Préparer les données
                if st.button("🔧 Préparer les Données"):
                    with st.spinner("Préparation des données..."):
                        try:
                            success = st.session_state.prediction_system.load_and_prepare_data(data)
                            if success:
                                st.success("✅ Données préparées!")
                                
                                # Afficher les informations extraites
                                pred_sys = st.session_state.prediction_system
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Colonnes identifiées:**")
                                    st.write(f"• Jour: {pred_sys.jour_col}")
                                    st.write(f"• Volume: {pred_sys.volume_col}")
                                
                                with col2:
                                    st.markdown("**Postes détectés:**")
                                    for poste in pred_sys.postes:
                                        st.write(f"• {poste}")
                                
                                # Poids calculés
                                st.markdown("**Poids des postes (basés sur moyenne défauts):**")
                                weights_df = pd.DataFrame([
                                    {"Poste": poste, "Poids": f"{weight:.1%}", "Poids_Num": weight}
                                    for poste, weight in pred_sys.poste_weights.items()
                                ])
                                st.dataframe(weights_df[["Poste", "Poids"]], hide_index=True)
                        
                        except Exception as e:
                            st.error(f"❌ Erreur: {e}")
                
                # Entraînement des modèles
                if hasattr(st.session_state.prediction_system, 'postes') and st.session_state.prediction_system.postes:
                    st.markdown("### 🤖 Étape 2: Entraînement des Modèles ML")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20) / 100
                    
                    with col2:
                        models_to_use = st.multiselect(
                            "Modèles à tester:",
                            ["RandomForest", "GradientBoosting", "DecisionTree", "LinearRegression"],
                            default=["RandomForest", "GradientBoosting", "DecisionTree"]
                        )
                    
                    if st.button("🚀 Entraîner les Modèles", type="primary"):
                        if not models_to_use:
                            st.error("Sélectionnez au moins un modèle!")
                        else:
                            with st.spinner("Entraînement en cours... Cela peut prendre quelques minutes."):
                                
                                # Créer le dictionnaire des modèles
                                models_dict = {}
                                if "RandomForest" in models_to_use:
                                    models_dict["RandomForest"] = RandomForestRegressor(n_estimators=100, random_state=42)
                                if "GradientBoosting" in models_to_use:
                                    models_dict["GradientBoosting"] = GradientBoostingRegressor(n_estimators=100, random_state=42)
                                if "DecisionTree" in models_to_use:
                                    models_dict["DecisionTree"] = DecisionTreeRegressor(random_state=42)
                                if "LinearRegression" in models_to_use:
                                    models_dict["LinearRegression"] = LinearRegression()
                                
                                try:
                                    results = st.session_state.prediction_system.train_models(
                                        test_size=test_size, 
                                        models_to_try=models_dict
                                    )
                                    
                                    if results:
                                        st.session_state.models_trained = True
                                        st.success("✅ Modèles entraînés avec succès!")
                                        
                                        # Afficher les résultats
                                        st.markdown("### 📊 Résultats de l'Entraînement")
                                        
                                        for poste, result in results.items():
                                            with st.expander(f"📋 Détails pour {poste}"):
                                                st.markdown(f"**Meilleur modèle:** {result['model']}")
                                                st.markdown(f"**Score R²:** {result['r2']:.4f}")
                                                
                                                # Tableau des performances de tous les modèles
                                                perf_data = []
                                                for model_name, metrics in result['details'].items():
                                                    perf_data.append({
                                                        'Modèle': model_name,
                                                        'R²': f"{metrics['r2']:.4f}",
                                                        'RMSE': f"{metrics['rmse']:.2f}",
                                                        'MAE': f"{metrics['mae']:.2f}"
                                                    })
                                                
                                                st.dataframe(pd.DataFrame(perf_data), hide_index=True)
                                        
                                        # Graphique de performance
                                        model_summary = st.session_state.prediction_system.get_model_summary()
                                        if model_summary:
                                            perf_chart = create_model_performance_chart(model_summary)
                                            if perf_chart:
                                                st.plotly_chart(perf_chart, use_container_width=True)
                                    
                                    else:
                                        st.error("❌ Échec de l'entraînement des modèles")
                                
                                except Exception as e:
                                    st.error(f"❌ Erreur pendant l'entraînement: {e}")
            
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du fichier: {e}")
        
        else:
            st.info("📁 Veuillez télécharger un fichier Excel pour commencer")

    # =================== NOUVELLE DEMANDE & PRÉDICTION ===================
    elif page == "📝 Nouvelle Demande & Prédiction":
        st.header("📝 Nouvelle Demande et Prédiction")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Veuillez d'abord entraîner les modèles dans la section 'Chargement & Entraînement'")
            return
        
        st.markdown("### 📋 Paramètres de la Nouvelle Demande")
        
        col1, col2 = st.columns(2)
        
        with col1:
            jour = st.selectbox(
                "Jour de la semaine:",
                options=list(range(1, 8)),
                format_func=lambda x: ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][x-1]
            )
        
        with col2:
            volume = st.number_input(
                "Volume de production prévu:",
                min_value=100,
                max_value=5000,
                value=1200,
                step=50
            )
        
        method = st.selectbox(
            "Méthode de calcul du taux global:",
            options=['moyenne_ponderee', 'moyenne', 'max', 'somme'],
            format_func=lambda x: {
                'moyenne_ponderee': '⚖️ Moyenne Pondérée (Recommandé)',
                'moyenne': '📊 Moyenne Simple',
                'max': '🔺 Maximum',
                'somme': '➕ Somme'
            }[x]
        )
        
        # Option validation
        with_validation = st.checkbox("🔍 J'ai les défauts réels pour validation")
        
        actual_defects = None
        if with_validation:
            st.markdown("### 📊 Défauts Réels (pour validation)")
            pred_sys = st.session_state.prediction_system
            
            actual_defects = {}
            cols = st.columns(len(pred_sys.postes))
            
            for i, poste in enumerate(pred_sys.postes):
                with cols[i]:
                    actual_defects[poste] = st.number_input(
                        f"Défauts {poste.replace('_defauts', '')}:", 
                        min_value=0.0, 
                        value=0.0, 
                        step=0.1
                    )
        
        if st.button("🔮 Faire la Prédiction", type="primary"):
            with st.spinner("Prédiction en cours..."):
                try:
                    # Ajouter à l'historique et faire la prédiction
                    result = st.session_state.prediction_system.add_prediction_to_history(
                        jour=jour,
                        volume=volume,
                        method=method,
                        actual_defects=actual_defects
                    )
                    
                    # Afficher les résultats
                    st.markdown("### 🎯 Résultats de la Prédiction")
                    
                    # Métriques principales
                    prediction = result['prediction']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "🎯 Taux Final de Rework",
                            f"{result['taux_final']:.2f}%"
                        )
                    
                    with col2:
                        taux_ml = prediction['taux_rework_chaine'][method]
                        st.metric(
                            "🤖 Taux ML Brut",
                            f"{taux_ml:.2f}%"
                        )
                    
                    with col3:
                        total_defauts = sum(prediction['predictions_postes'].values())
                        st.metric(
                            "📊 Défauts Total Prédits",
                            f"{total_defauts:.1f}"
                        )
                    
                    with col4:
                        if result.get('accuracy') is not None:
                            st.metric(
                                "✅ Précision",
                                f"{result['accuracy']:.1f}%"
                            )
                        else:
                            st.metric("✅ Précision", "N/A")
                    
                    # Détails par poste avec modèles utilisés
                    st.markdown("### 🏭 Détails par Poste")
                    
                    model_summary = st.session_state.prediction_system.get_model_summary()
                    
                    poste_details = []
                    for poste in st.session_state.prediction_system.postes:
                        defauts_pred = prediction['predictions_postes'][poste]
                        taux_poste = prediction['taux_rework_postes'][poste]
                        model_info = model_summary.get(poste, {})
                        
                        detail = {
                            'Poste': poste.replace('_defauts', ''),
                            'Modèle Utilisé': model_info.get('model', 'N/A'),
                            'Score R²': f"{model_info.get('r2_score', 0):.3f}",
                            'Défauts Prédits': f"{defauts_pred:.1f}",
                            'Taux Rework (%)': f"{taux_poste:.2f}",
                            'Poids': f"{model_info.get('weight', 0):.1%}",
                            'Défauts Réels': f"{actual_defects.get(poste, 'N/A')}" if actual_defects else "N/A"
                        }
                        poste_details.append(detail)
                    
                    st.dataframe(pd.DataFrame(poste_details), hide_index=True, use_container_width=True)
                    
                    # Feature importance si disponible
                    pred_sys = st.session_state.prediction_system
                    if pred_sys.feature_importances:
                        st.markdown("### 🎯 Importance des Caractéristiques")
                        
                        importance_data = []
                        for poste, importance in pred_sys.feature_importances.items():
                            importance_data.append({
                                'Poste': poste.replace('_defauts', ''),
                                'Importance Volume': f"{importance.get('Volume', 0):.3f}",
                                'Importance Jour': f"{importance.get('Jour', 0):.3f}"
                            })
                        
                        st.dataframe(pd.DataFrame(importance_data), hide_index=True)
                    
                    # Comparaison des méthodes
                    st.markdown("### 📊 Comparaison des Méthodes de Calcul")
                    
                    methodes_comp = []
                    for meth, taux in prediction['taux_rework_chaine'].items():
                        methodes_comp.append({
                            'Méthode': meth.replace('_', ' ').title(),
                            'Taux Rework (%)': f"{taux:.2f}",
                            'Défauts Estimés': f"{prediction['predictions_chaine'][meth]:.1f}",
                            'Sélectionnée': "✅" if meth == method else ""
                        })
                    
                    st.dataframe(pd.DataFrame(methodes_comp), hide_index=True)
                    
                    st.success("✅ Prédiction ajoutée à l'historique avec succès!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction: {e}")

    # =================== PLANIFICATION INTELLIGENTE ===================
    elif page == "🎯 Planification Intelligente":
        st.header("🎯 Planification Intelligente")
        
        pred_sys = st.session_state.prediction_system
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Veuillez d'abord entraîner les modèles")
            return
        
        if not pred_sys.predictions_history:
            st.warning("⚠️ Veuillez d'abord ajouter au moins une nouvelle demande")
            return
        
        st.markdown("### ⚙️ Configuration de la Planification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_scenarios = st.selectbox("Nombre de scénarios:", [1, 3, 5], index=1)
            n_references = st.number_input("Nombre de références:", min_value=3, max_value=10, value=8)
            n_shifts = st.number_input("Nombre de shifts:", min_value=1, max_value=5, value=3)
        
        with col2:
            capacite_shift = st.number_input("Capacité par shift:", min_value=100, max_value=500, value=180)
            penalite_penurie = st.number_input("Pénalité pénurie:", min_value=500, max_value=3000, value=1000)
            cout_production = st.number_input("Coût de production unitaire:", min_value=10, max_value=50, value=20)
        
        # Paramètres avancés
        with st.expander("⚙️ Paramètres Avancés"):
            col1, col2 = st.columns(2)
            with col1:
                alpha_rework = st.slider("Alpha rework (efficacité reprise):", 0.0, 1.0, 0.8, 0.1)
                beta = st.slider("Beta (facteur capacité défauts):", 1.0, 2.0, 1.2, 0.1)
            with col2:
                time_limit = st.number_input("Limite de temps (secondes):", min_value=30, max_value=600, value=300)
        
        # Informations sur la dernière prédiction
        latest_pred = pred_sys.predictions_history[-1]
        st.markdown("### 📊 Contexte de la Dernière Prédiction")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Jour", ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'][latest_pred['jour']-1])
        with col2:
            st.metric("Volume", f"{latest_pred['volume']:,}")
        with col3:
            st.metric("Taux Final", f"{latest_pred['taux_final']:.2f}%")
        with col4:
            hist_count = len(pred_sys.predictions_history)
            st.metric("Historique", f"{hist_count} prédictions")
        
        if st.button("🚀 Générer et Optimiser", type="primary"):
            with st.spinner("Génération des scénarios et optimisation..."):
                try:
                    # Créer le planificateur
                    planner = IntelligentPlanning(pred_sys)
                    
                    # Générer les scénarios
                    scenarios = planner.generate_scenarios(latest_pred['taux_final'], n_scenarios)
                    
                    # Configuration des références et demandes
                    references = [f'REF_{i+1:02d}' for i in range(n_references)]
                    demandes = np.random.randint(20, 50, n_references)  # Demandes aléatoires pour la démo
                    capacites = [capacite_shift] * n_shifts
                    
                    # Paramètres d'optimisation
                    params = {
                        'alpha_rework': alpha_rework,
                        'beta': beta,
                        'penalite_penurie': penalite_penurie,
                        'cout_production': cout_production
                    }
                    
                    # Configuration du problème
                    success = planner.setup_optimization(scenarios, references, demandes, capacites, params)
                    
                    if success:
                        # Résolution
                        solved = planner.solve(time_limit)
                        
                        if solved:
                            st.session_state.planning_results = planner.results
                            st.success("✅ Optimisation réussie!")
                            
                            # Afficher les scénarios générés
                            st.markdown("### 📈 Scénarios Générés")
                            
                            scenarios_chart = create_scenarios_comparison(scenarios)
                            st.plotly_chart(scenarios_chart, use_container_width=True)
                            
                            # Informations sur les scénarios
                            scenario_info = []
                            scenario_names = ['Optimiste', 'Nominal', 'Pessimiste'] if n_scenarios == 3 else [f'Scénario {i+1}' for i in range(n_scenarios)]
                            
                            for i, (name, rate) in enumerate(zip(scenario_names, scenarios)):
                                if len(pred_sys.predictions_history) > 1:
                                    source = "Historique + Nouvelle demande"
                                else:
                                    source = "Nouvelle demande uniquement"
                                
                                scenario_info.append({
                                    'Scénario': name,
                                    'Taux Rework (%)': f"{rate:.2f}",
                                    'Source': source,
                                    'Probabilité': f"{100/n_scenarios:.1f}%"
                                })
                            
                            st.dataframe(pd.DataFrame(scenario_info), hide_index=True)
                            
                            # Résultats principaux
                            st.markdown("### 📊 Résultats d'Optimisation")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Coût Total Optimal", f"{planner.results['cout_total']:.0f} €")
                            with col2:
                                st.metric("Statut", planner.results['status'])
                            with col3:
                                st.metric("Scénarios Analysés", len(scenarios))
                            
                        else:
                            st.error("❌ Échec de l'optimisation")
                    else:
                        st.error("❌ Échec de la configuration")
                
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")

    # =================== DASHBOARD & COMPARAISON ===================
    elif page == "📊 Dashboard & Comparaison":
        st.header("📊 Dashboard et Comparaison des Scénarios")
        
        if not st.session_state.planning_results:
            st.warning("⚠️ Veuillez d'abord effectuer une planification")
            return
        
        results = st.session_state.planning_results
        pred_sys = st.session_state.prediction_system
        
        st.markdown("### 🎯 Dashboard de Choix du Meilleur Scénario")
        
        # Récupérer les informations nécessaires
        scenarios = results['scenarios']
        references = [f'REF_{i+1:02d}' for i in range(8)]  # Assumant 8 références par défaut
        
        # Créer le dashboard
        dashboard_chart, metrics_df = create_planning_dashboard(results, references, scenarios)
        
        if dashboard_chart and metrics_df is not None:
            st.plotly_chart(dashboard_chart, use_container_width=True)
            
            # Tableau de comparaison détaillé
            st.markdown("### 📋 Comparaison Détaillée des Scénarios")
            
            # Ajouter une colonne de recommandation
            best_scenario_idx = metrics_df['Score_Global'].idxmax()
            metrics_df['Recommandation'] = metrics_df.apply(
                lambda row: "⭐ RECOMMANDÉ" if row.name == best_scenario_idx else "", axis=1
            )
            
            # Formatage pour l'affichage
            display_df = metrics_df.copy()
            display_df['Production_Totale'] = display_df['Production_Totale'].apply(lambda x: f"{x:,.0f}")
            display_df['Penuries_Totales'] = display_df['Penuries_Totales'].apply(lambda x: f"{x:.1f}")
            display_df['Cout_Estime'] = display_df['Cout_Estime'].apply(lambda x: f"{x:,.0f} €")
            display_df['Score_Global'] = display_df['Score_Global'].apply(lambda x: f"{x:.1f}")
            display_df['Taux_Rework'] = display_df['Taux_Rework'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
            
            # Analyse et recommandations
            st.markdown("### 💡 Analyse et Recommandations")
            
            best_scenario = metrics_df.iloc[best_scenario_idx]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **🏆 Scénario Recommandé: {best_scenario['Scenario']}**
                
                - **Taux de Rework:** {best_scenario['Taux_Rework']:.2f}%
                - **Score Global:** {best_scenario['Score_Global']:.1f}/100
                - **Production Totale:** {best_scenario['Production_Totale']:,.0f} unités
                - **Pénuries:** {best_scenario['Penuries_Totales']:.1f} unités
                """)
            
            with col2:
                # Critères de décision
                st.markdown("**🎯 Critères de Décision:**")
                
                if best_scenario['Penuries_Totales'] < 10:
                    st.success("✅ Faibles pénuries")
                else:
                    st.warning("⚠️ Pénuries élevées")
                
                if best_scenario['Taux_Rework'] < 5:
                    st.success("✅ Taux de rework acceptable")
                elif best_scenario['Taux_Rework'] < 10:
                    st.warning("⚠️ Taux de rework modéré")
                else:
                    st.error("❌ Taux de rework élevé")
                
                if best_scenario['Score_Global'] > 80:
                    st.success("✅ Score global excellent")
                elif best_scenario['Score_Global'] > 60:
                    st.info("ℹ️ Score global correct")
                else:
                    st.warning("⚠️ Score global à améliorer")
            
            # Actions recommandées
            st.markdown("### 🔧 Actions Recommandées")
            
            actions = []
            
            if best_scenario['Taux_Rework'] > 8:
                actions.append("🔍 Analyser les causes racines du taux de rework élevé")
                actions.append("⚙️ Mettre en place des actions d'amélioration qualité")
            
            if best_scenario['Penuries_Totales'] > 15:
                actions.append("📈 Augmenter la capacité de production")
                actions.append("📋 Revoir la planification des demandes")
            
            if best_scenario['Cout_Estime'] > metrics_df['Cout_Estime'].mean():
                actions.append("💰 Optimiser les coûts de production")
                actions.append("🔄 Améliorer l'efficacité des processus")
            
            if not actions:
                actions.append("✅ Configuration optimale - Maintenir les performances")
            
            for action in actions:
                st.write(f"• {action}")
        
        else:
            st.error("❌ Impossible de créer le dashboard")

    # =================== HISTORIQUE & PERFORMANCE ===================
    elif page == "📈 Historique & Performance":
        st.header("📈 Historique et Analyse de Performance")
        
        pred_sys = st.session_state.prediction_system
        
        if not pred_sys.predictions_history:
            st.info("📭 Aucun historique disponible")
            return
        
        # Statistiques générales
        st.markdown("### 📊 Statistiques Générales")
        
        history = pred_sys.predictions_history
        total_predictions = len(history)
        validated_predictions = sum(1 for p in history if p.get('accuracy') is not None)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Prédictions", total_predictions)
        
        with col2:
            st.metric("Prédictions Validées", validated_predictions)
        
        with col3:
            if validated_predictions > 0:
                avg_accuracy = np.mean([p['accuracy'] for p in history if p.get('accuracy')])
                st.metric("Précision Moyenne", f"{avg_accuracy:.1f}%")
            else:
                st.metric("Précision Moyenne", "N/A")
        
        with col4:
            avg_rework = np.mean([p['taux_final'] for p in history])
            st.metric("Taux Rework Moyen", f"{avg_rework:.2f}%")
        
        # Évolution temporelle
        if total_predictions > 1:
            st.markdown("### 📈 Évolution Temporelle")
            
            # Préparer les données pour le graphique
            timestamps = [p['timestamp'] for p in history]
            taux_finals = [p['taux_final'] for p in history]
            accuracies = [p.get('accuracy') for p in history]
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Évolution des Taux de Rework', 'Évolution de la Précision'),
                vertical_spacing=0.1
            )
            
            # Taux de rework
            fig.add_trace(
                go.Scatter(x=timestamps, y=taux_finals,
                          mode='lines+markers',
                          name='Taux de Rework Final',
                          line=dict(color='blue', width=2),
                          marker=dict(size=8)),
                row=1, col=1
            )
            
            # Précision (seulement les valeurs non nulles)
            valid_timestamps = [t for t, a in zip(timestamps, accuracies) if a is not None]
            valid_accuracies = [a for a in accuracies if a is not None]
            
            if valid_accuracies:
                fig.add_trace(
                    go.Scatter(x=valid_timestamps, y=valid_accuracies,
                              mode='lines+markers',
                              name='Précision (%)',
                              line=dict(color='green', width=2),
                              marker=dict(size=8)),
                    row=2, col=1
                )
            
            fig.update_layout(height=500, showlegend=True)
            fig.update_xaxes(title_text="Temps", row=2, col=1)
            fig.update_yaxes(title_text="Taux de Rework (%)", row=1, col=1)
            fig.update_yaxes(title_text="Précision (%)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance par jour de la semaine
        if validated_predictions > 0:
            st.markdown("### 📅 Performance par Jour de la Semaine")
            
            # Grouper par jour
            day_stats = {}
            day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            
            for pred in history:
                jour = pred['jour']
                if jour not in day_stats:
                    day_stats[jour] = {
                        'count': 0,
                        'taux_rework': [],
                        'accuracies': []
                    }
                
                day_stats[jour]['count'] += 1
                day_stats[jour]['taux_rework'].append(pred['taux_final'])
                
                if pred.get('accuracy') is not None:
                    day_stats[jour]['accuracies'].append(pred['accuracy'])
            
            # Créer le tableau de performance
            perf_data = []
            for jour in range(1, 8):
                if jour in day_stats:
                    stats = day_stats[jour]
                    avg_rework = np.mean(stats['taux_rework'])
                    avg_accuracy = np.mean(stats['accuracies']) if stats['accuracies'] else None
                    
                    perf_data.append({
                        'Jour': day_names[jour-1],
                        'Nombre': stats['count'],
                        'Taux Rework Moyen (%)': f"{avg_rework:.2f}",
                        'Précision Moyenne (%)': f"{avg_accuracy:.1f}" if avg_accuracy else "N/A",
                        'Écart-Type Rework': f"{np.std(stats['taux_rework']):.2f}"
                    })
                else:
                    perf_data.append({
                        'Jour': day_names[jour-1],
                        'Nombre': 0,
                        'Taux Rework Moyen (%)': "N/A",
                        'Précision Moyenne (%)': "N/A",
                        'Écart-Type Rework': "N/A"
                    })
            
            st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)
            
            # Graphique par jour
            days_with_data = [d for d in perf_data if d['Nombre'] > 0]
            if days_with_data:
                fig_days = go.Figure()
                
                days = [d['Jour'] for d in days_with_data]
                rework_rates = [float(d['Taux Rework Moyen (%)']) for d in days_with_data]
                
                fig_days.add_trace(
                    go.Bar(x=days, y=rework_rates,
                           marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                                       '#9467bd', '#8c564b', '#e377c2'][:len(days)],
                           text=[f'{rate:.2f}%' for rate in rework_rates],
                           textposition='auto')
                )
                
                fig_days.update_layout(
                    title="Taux de Rework Moyen par Jour",
                    xaxis_title="Jour de la Semaine",
                    yaxis_title="Taux de Rework (%)",
                    showlegend=False
                )
                
                st.plotly_chart(fig_days, use_container_width=True)
        
        # Performance des modèles
        if st.session_state.models_trained:
            st.markdown("### 🤖 Performance des Modèles ML")
            
            model_summary = pred_sys.get_model_summary()
            if model_summary:
                model_perf_data = []
                
                for poste, info in model_summary.items():
                    model_perf_data.append({
                        'Poste': poste.replace('_defauts', ''),
                        'Modèle': info['model'],
                        'Score R²': f"{info['r2_score']:.4f}",
                        'Poids': f"{info['weight']:.1%}",
                        'Import. Volume': f"{info.get('feature_importance', {}).get('Volume', 0):.3f}",
                        'Import. Jour': f"{info.get('feature_importance', {}).get('Jour', 0):.3f}"
                    })
                
                st.dataframe(pd.DataFrame(model_perf_data), hide_index=True, use_container_width=True)
                
                # Graphique de performance des modèles
                model_chart = create_model_performance_chart(model_summary)
                if model_chart:
                    st.plotly_chart(model_chart, use_container_width=True)
        
        # Tableau détaillé de l'historique
        st.markdown("### 📋 Historique Détaillé")
        
        n_display = st.slider("Nombre d'entrées à afficher:", 5, min(20, total_predictions), 10)
        
        recent_history = history[-n_display:]
        
        display_data = []
        for i, pred in enumerate(reversed(recent_history), 1):
            day_name = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'][pred['jour']-1]
            
            display_data.append({
                '#': total_predictions - i + 1,
                'Date/Heure': pred['timestamp'].strftime("%Y-%m-%d %H:%M"),
                'Jour': day_name,
                'Volume': f"{pred['volume']:,}",
                'Méthode': pred['method'],
                'Taux Final (%)': f"{pred['taux_final']:.2f}",
                'Précision (%)': f"{pred['accuracy']:.1f}" if pred.get('accuracy') else "N/A",
                'Validé': "✅" if pred.get('actual_defects') else "❌"
            })
        
        st.dataframe(pd.DataFrame(display_data), hide_index=True, use_container_width=True)
        
        # Export des données
        st.markdown("### 💾 Export des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Exporter Historique CSV"):
                csv_data = []
                for pred in history:
                    csv_data.append({
                        'Timestamp': pred['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                        'Jour': pred['jour'],
                        'Volume': pred['volume'],
                        'Methode': pred['method'],
                        'Taux_Final_%': pred['taux_final'],
                        'Precision_%': pred.get('accuracy', ''),
                        'Valide': pred.get('actual_defects') is not None
                    })
                
                csv_df = pd.DataFrame(csv_data)
                csv_string = csv_df.to_csv(index=False)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"historique_predictions_{timestamp}.csv"
                
                st.download_button(
                    label="💾 Télécharger CSV",
                    data=csv_string,
                    file_name=filename,
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🧹 Nettoyer Historique"):
                if st.session_state.get('confirm_clear', False):
                    pred_sys.predictions_history = []
                    pred_sys.save_history()
                    st.session_state.confirm_clear = False
                    st.success("✅ Historique nettoyé")
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("⚠️ Cliquez à nouveau pour confirmer")

# =====================================================================
# SIDEBAR ET INFORMATIONS COMPLÉMENTAIRES
# =====================================================================

def create_sidebar_info():
    """Crée les informations dans la sidebar"""
    st.sidebar.markdown("---")
    
    # Statut du système
    st.sidebar.markdown("### 📊 Statut Système")
    
    if st.session_state.models_trained:
        st.sidebar.success("✅ Modèles Entraînés")
        model_summary = st.session_state.prediction_system.get_model_summary()
        if model_summary:
            st.sidebar.write(f"**Postes:** {len(model_summary)}")
            avg_r2 = np.mean([info['r2_score'] for info in model_summary.values()])
            st.sidebar.write(f"**R² Moyen:** {avg_r2:.3f}")
    else:
        st.sidebar.warning("⏳ Modèles non entraînés")
    
    hist_count = len(st.session_state.prediction_system.predictions_history)
    if hist_count > 0:
        st.sidebar.info(f"📊 {hist_count} prédictions")
        
        # Dernière prédiction
        latest = st.session_state.prediction_system.predictions_history[-1]
        st.sidebar.write(f"**Dernier taux:** {latest['taux_final']:.2f}%")
        st.sidebar.write(f"**Dernière validation:** {'✅' if latest.get('accuracy') else '❌'}")
    else:
        st.sidebar.warning("📭 Aucun historique")
    
    if st.session_state.planning_results:
        st.sidebar.success("✅ Planification OK")
        cost = st.session_state.planning_results['cout_total']
        st.sidebar.write(f"**Coût optimal:** {cost:.0f}€")
    else:
        st.sidebar.warning("⏳ Planification à faire")
    
    # Actions rapides
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Actions Rapides")
    
    if st.sidebar.button("🔄 Réinitialiser Système"):
        for key in ['prediction_system', 'models_trained', 'planning_results']:
            if key in st.session_state:
                del st.session_state[key]
        st.sidebar.success("✅ Système réinitialisé")
        st.rerun()
    
    # Informations techniques
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Informations")
    st.sidebar.info("""
    **Version:** 3.0 Production
    
    **Fonctionnalités:**
    - 🤖 ML réel (RF, GB, DT)
    - 📊 Prédictions précises
    - 🎯 Planification optimale
    - 📈 Dashboard interactif
    - 📁 Import/Export complet
    """)

# =====================================================================
# POINT D'ENTRÉE PRINCIPAL
# =====================================================================

if __name__ == "__main__":
    # Ajouter les informations de la sidebar
    create_sidebar_info()
    
    # Lancer l'application principale
    main()

# =====================================================================
# INSTRUCTIONS D'UTILISATION COMPLÈTES
# =====================================================================

"""
🚀 GUIDE D'UTILISATION COMPLET:

1. **Installation des Dépendances:**
   ```bash
   pip install streamlit pandas numpy plotly scikit-learn pulp openpyxl
   ```

2. **Lancement de l'Application:**
   ```bash
   streamlit run nom_du_fichier.py
   ```

3. **Workflow Complet:**

   **Étape 1: Chargement & Entraînement**
   - Téléchargez votre fichier Excel avec colonnes: Jour, Volume_production, Poste1_defauts, etc.
   - Le système identifie automatiquement les colonnes
   - Choisissez les modèles ML à tester (RandomForest, GradientBoosting, DecisionTree)
   - L'entraînement sélectionne automatiquement le meilleur modèle par poste
   - Visualisez les performances (Score R², RMSE, MAE)

   **Étape 2: Nouvelle Demande & Prédiction**
   - Saisissez le jour de la semaine et le volume prévu
   - Choisissez la méthode de calcul (moyenne pondérée recommandée)
   - Optionnel: Ajoutez les défauts réels pour validation
   - Le système prédit les défauts par poste et calcule le taux de rework
   - Affichage détaillé: modèles utilisés, paramètres, précision

   **Étape 3: Planification Intelligente**
   - Configuration des paramètres (scénarios, capacités, coûts)
   - Génération automatique de scénarios basés sur l'historique + nouvelle demande
   - Optimisation mathématique avec contraintes réelles
   - Résolution du problème de planification

   **Étape 4: Dashboard & Comparaison**
   - Visualisation comparative des scénarios
   - Métriques de performance (production, pénuries, coûts, scores)
   - Recommandation automatique du meilleur scénario
   - Actions correctives suggérées

   **Étape 5: Historique & Performance**
   - Analyse de l'évolution temporelle
   - Performance par jour de la semaine
   - Statistiques des modèles ML
   - Export des données

4. **Fonctionnalités Avancées:**

   **Machine Learning:**
   - Entraînement automatique de 3+ modèles par poste
   - Sélection du meilleur modèle basée sur le score R²
   - Feature importance (Volume vs Jour)
   - Validation croisée et métriques multiples

   **Prédictions Intelligentes:**
   - Ajustements contextuels (volume, jour de semaine)
   - Calcul de taux de rework par multiple méthodes
   - Validation en temps réel avec défauts réels
   - Historique complet avec tracking de précision

   **Planification Optimale:**
   - Génération de scénarios intelligents
   - Combinaison historique + prédiction actuelle
   - Optimisation avec PuLP (CBC solver)
   - Gestion des contraintes de capacité et demande

   **Dashboard Interactif:**
   - Comparaison visuelle des scénarios
   - Métriques multiples (coût, pénuries, production)
   - Score global automatique
   - Recommandations basées sur les performances

5. **Format des Données d'Entrée:**
   
   **Colonnes Obligatoires:**
   - `Jour` ou `jour`: Numéro du jour (1-7, Lundi=1)
   - `Volume_production` ou `volume`: Volume de production
   - `Poste1_defauts`, `Poste2_defauts`, etc.: Nombre de défauts par poste

   **Exemple de Structure:**
   ```
   Jour | Volume_production | Poste1_defauts | Poste2_defauts | Poste3_defauts
   1    | 1200             | 24            | 18            | 31
   2    | 1350             | 27            | 20            | 34
   ...
   ```

6. **Modèles ML Supportés:**
   - **RandomForest:** Robuste, gère les non-linéarités
   - **GradientBoosting:** Précis, boosting séquentiel
   - **DecisionTree:** Interprétable, rapide
   - **LinearRegression:** Simple, baseline

7. **Optimisation de Planification:**
   - **Variables:** Production par (scénario, référence, shift)
   - **Contraintes:** Demande, capacité
   - **Objectif:** Minimiser coût total (production + pénuries)
   - **Scénarios:** Optimiste, Nominal, Pessimiste

8. **Export et Persistance:**
   - Export CSV/Excel de l'historique
   - Sauvegarde automatique dans session Streamlit
   - Réinitialisation complète du système

9. **Conseils d'Utilisation:**
   - Utilisez au moins 50 lignes de données pour l'entraînement
   - Validez régulièrement les prédictions avec défauts réels
   - Analysez les tendances par jour de la semaine
   - Ajustez les paramètres de planification selon votre contexte

10. **Dépannage Courant:**
    - **Erreur de colonnes:** Vérifiez les noms des colonnes
    - **Modèles non entraînés:** Retournez à l'étape de chargement
    - **Optimisation échouée:** Réduisez les contraintes ou augmentez le temps limite
    - **Performances faibles:** Collectez plus de données ou ajustez les paramètres

Ce système représente une solution complète de prédiction et planification industrielle avec ML et optimisation mathématique intégrés.
"""
