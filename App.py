# Système Intégré Streamlit avec Gestion Avancée de l'Historique
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

# =====================================================================
# CONFIGURATION STREAMLIT
# =====================================================================

st.set_page_config(
    page_title="Système Intégré avec Historique",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
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
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .history-card {
        background: #e8f5e8;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# CLASSES AMÉLIORÉES AVEC HISTORIQUE
# =====================================================================

class EnhancedStreamlitDefectPredictor:
    """Prédicteur amélioré avec gestion de l'historique pour Streamlit"""
    
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
        self.historical_accuracy = {}
        self.trend_weights = {}
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
    
    def add_new_demand_prediction(self, jour, volume, actual_defects=None, method='moyenne_ponderee'):
        """Ajoute une nouvelle demande et fait une prédiction avec historique"""
        
        # 1. Prédiction ML standard
        ml_prediction = self._make_ml_prediction(jour, volume)
        
        # 2. Ajustement avec historique
        adjusted_prediction = self._adjust_with_historical_trend(ml_prediction, jour, volume)
        
        # 3. Calcul du taux final
        final_rework_rate = self._calculate_final_rework_rate(adjusted_prediction, jour, volume, method)
        
        # 4. Création de l'enregistrement
        prediction_record = {
            'timestamp': datetime.now(),
            'jour': jour,
            'volume': volume,
            'method': method,
            'ml_prediction': ml_prediction,
            'adjusted_prediction': adjusted_prediction,
            'final_rework_rate': final_rework_rate,
            'actual_defects': actual_defects,
            'accuracy': None
        }
        
        # 5. Validation si défauts réels fournis
        if actual_defects:
            accuracy = self._calculate_accuracy(adjusted_prediction, actual_defects, volume)
            prediction_record['accuracy'] = accuracy
            self._update_model_performance(accuracy, jour)
        
        # 6. Ajout à l'historique
        self.predictions_history.append(prediction_record)
        self.save_history()
        
        return prediction_record
    
    def _make_ml_prediction(self, jour, volume):
        """Prédiction ML standard"""
        if not self.models:
            raise ValueError("Les modèles doivent être entraînés!")
        
        # Préparation des données
        new_data = pd.DataFrame({
            self.volume_col: [volume],
            'jour_numerique': [jour]
        })
        X_new = new_data[[self.volume_col, 'jour_numerique']]
        
        # Prédictions par poste
        predictions_postes = {}
        for poste, model in self.models.items():
            predictions_postes[poste] = max(0, model.predict(X_new)[0])
        
        # Prédictions chaîne
        predictions_chaine = {
            'max': max(predictions_postes.values()) if predictions_postes else 0,
            'moyenne': np.mean(list(predictions_postes.values())) if predictions_postes else 0,
            'moyenne_ponderee': self.calculate_weighted_average(predictions_postes),
            'somme': sum(predictions_postes.values()) if predictions_postes else 0
        }
        
        # Taux de rework
        taux_rework_postes = {poste: (defauts / volume) * 100 
                             for poste, defauts in predictions_postes.items()}
        
        taux_rework_chaine = {method: (defauts / volume) * 100 
                             for method, defauts in predictions_chaine.items()}
        
        return {
            'predictions_postes': predictions_postes,
            'predictions_chaine': predictions_chaine,
            'taux_rework_postes': taux_rework_postes,
            'taux_rework_chaine': taux_rework_chaine
        }
    
    def _adjust_with_historical_trend(self, ml_prediction, jour, volume):
        """Ajuste avec les tendances historiques"""
        if len(self.predictions_history) < 3:
            return ml_prediction
        
        # Analyser les tendances récentes
        recent_predictions = [p for p in self.predictions_history[-10:] 
                            if p.get('accuracy') is not None]
        
        if not recent_predictions:
            return ml_prediction
        
        # Calculer le facteur de correction
        correction_factor = self._calculate_correction_factor(recent_predictions, jour)
        
        # Appliquer la correction
        adjusted_prediction = ml_prediction.copy()
        for method in adjusted_prediction['taux_rework_chaine']:
            original_rate = adjusted_prediction['taux_rework_chaine'][method]
            adjusted_rate = original_rate * correction_factor
            adjusted_prediction['taux_rework_chaine'][method] = max(0.1, min(30, adjusted_rate))
        
        return adjusted_prediction
    
    def _calculate_correction_factor(self, recent_predictions, jour):
        """Calcule le facteur de correction basé sur l'historique"""
        jour_errors = []
        all_errors = []
        
        for pred in recent_predictions:
            accuracy = pred.get('accuracy', 0)
            error_rate = max(0, (100 - accuracy) / 100)
            all_errors.append(error_rate)
            
            if pred['jour'] == jour:
                jour_errors.append(error_rate)
        
        if jour_errors:
            avg_error = np.mean(jour_errors)
            correction = 1.0 + (avg_error * 0.3)  # Correction modérée
        elif all_errors:
            avg_error = np.mean(all_errors)
            correction = 1.0 + (avg_error * 0.2)  # Correction plus légère
        else:
            correction = 1.0
        
        return max(0.7, min(1.5, correction))  # Borner la correction
    
    def _calculate_final_rework_rate(self, adjusted_prediction, jour, volume, method):
        """Calcule le taux final pour la planification"""
        base_rate = adjusted_prediction['taux_rework_chaine'][method]
        
        # Facteurs d'ajustement
        volume_factor = 0.98 if volume > 1500 else (1.02 if volume < 800 else 1.0)
        jour_factor = 1.08 if jour in [6, 7] else 1.0
        
        final_rate = base_rate * volume_factor * jour_factor
        return max(0.5, min(25, final_rate))
    
    def _calculate_accuracy(self, prediction, actual_defects, volume):
        """Calcule la précision de la prédiction"""
        predicted_total = sum(prediction['predictions_postes'].values())
        actual_total = sum(actual_defects.values())
        
        if actual_total == 0 and predicted_total == 0:
            return 100.0
        elif actual_total == 0:
            return max(0, 100 - (predicted_total / volume * 100))
        
        relative_error = abs(predicted_total - actual_total) / actual_total
        return max(0, 100 - (relative_error * 100))
    
    def _update_model_performance(self, accuracy, jour):
        """Met à jour les performances des modèles"""
        if jour not in self.historical_accuracy:
            self.historical_accuracy[jour] = []
        
        self.historical_accuracy[jour].append(accuracy)
        if len(self.historical_accuracy[jour]) > 15:
            self.historical_accuracy[jour] = self.historical_accuracy[jour][-15:]
    
    def get_historical_statistics(self):
        """Retourne les statistiques de l'historique"""
        if not self.predictions_history:
            return None
        
        # Statistiques générales
        total_predictions = len(self.predictions_history)
        predictions_with_validation = [p for p in self.predictions_history 
                                     if p.get('accuracy') is not None]
        
        stats = {
            'total_predictions': total_predictions,
            'validated_predictions': len(predictions_with_validation),
            'recent_trend': self._get_recent_trend()
        }
        
        # Statistiques de précision
        if predictions_with_validation:
            accuracies = [p['accuracy'] for p in predictions_with_validation]
            stats.update({
                'avg_accuracy': np.mean(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'std_accuracy': np.std(accuracies)
            })
        
        # Statistiques par jour
        stats['by_day'] = {}
        for jour in range(1, 8):
            day_preds = [p for p in predictions_with_validation if p['jour'] == jour]
            if day_preds:
                day_accuracies = [p['accuracy'] for p in day_preds]
                stats['by_day'][jour] = {
                    'count': len(day_preds),
                    'avg_accuracy': np.mean(day_accuracies)
                }
        
        return stats
    
    def _get_recent_trend(self):
        """Analyse la tendance récente"""
        if len(self.predictions_history) < 3:
            return "Historique insuffisant"
        
        recent_rates = [p['final_rework_rate'] for p in self.predictions_history[-5:]]
        
        if recent_rates[-1] > recent_rates[0] * 1.1:
            return "📈 Tendance à la hausse"
        elif recent_rates[-1] < recent_rates[0] * 0.9:
            return "📉 Tendance à la baisse"
        else:
            return "📊 Tendance stable"
    
    def export_history_to_excel(self):
        """Exporte l'historique vers Excel"""
        if not self.predictions_history:
            return None
        
        data = []
        for pred in self.predictions_history:
            row = {
                'Timestamp': pred['timestamp'],
                'Jour': pred['jour'],
                'Jour_Nom': ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'][pred['jour']-1],
                'Volume': pred['volume'],
                'Méthode': pred['method'],
                'Taux_ML_%': pred['ml_prediction']['taux_rework_chaine']['moyenne_ponderee'],
                'Taux_Ajusté_%': pred['adjusted_prediction']['taux_rework_chaine']['moyenne_ponderee'],
                'Taux_Final_%': pred['final_rework_rate'],
                'Précision_%': pred.get('accuracy', ''),
                'A_Défauts_Réels': 'Oui' if pred.get('actual_defects') else 'Non'
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Historique_Predictions', index=False)
            
            # Feuille avec statistiques
            stats = self.get_historical_statistics()
            if stats:
                stats_data = [
                    ['Métrique', 'Valeur'],
                    ['Total prédictions', stats['total_predictions']],
                    ['Prédictions validées', stats['validated_predictions']],
                    ['Précision moyenne (%)', f"{stats.get('avg_accuracy', 0):.1f}"],
                    ['Précision min (%)', f"{stats.get('min_accuracy', 0):.1f}"],
                    ['Précision max (%)', f"{stats.get('max_accuracy', 0):.1f}"],
                    ['Tendance récente', stats['recent_trend']]
                ]
                
                stats_df = pd.DataFrame(stats_data[1:], columns=stats_data[0])
                stats_df.to_excel(writer, sheet_name='Statistiques', index=False)
        
        return output.getvalue()
    
    # Méthodes héritées (simplifiées)
    def calculate_weighted_average(self, predictions_postes):
        """Calcule la moyenne pondérée"""
        if not predictions_postes:
            return 0
        
        if not self.poste_weights:
            return np.mean(list(predictions_postes.values()))
        
        weighted_sum = sum(pred * self.poste_weights.get(poste, 0) 
                          for poste, pred in predictions_postes.items())
        total_weight = sum(self.poste_weights.get(poste, 0) 
                          for poste in predictions_postes.keys())
        
        return weighted_sum / total_weight if total_weight > 0 else np.mean(list(predictions_postes.values()))

class StreamlitPlanningWithHistory:
    """Planification intégrée avec historique pour Streamlit"""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.model = None
        self.variables = {}
        self.results = {}
        self.parameters = {}
    
    def configure_with_history(self, S=3, T=3, **params):
        """Configure la planification avec données historiques"""
        
        # Obtenir le dernier taux prédit
        if not self.predictor.predictions_history:
            raise ValueError("Aucun historique disponible!")
        
        latest_pred = self.predictor.predictions_history[-1]
        base_rate = latest_pred['final_rework_rate']
        
        # Calculer l'incertitude basée sur l'historique
        uncertainty = self._calculate_uncertainty()
        
        # Générer les scénarios
        scenario_rates = self._generate_scenarios(base_rate, uncertainty, S)
        
        # Configuration des paramètres
        R = params.get('R', [f'REF_{i+1:02d}' for i in range(8)])
        EDI = params.get('EDI', [25, 40, 35, 30, 45, 28, 38, 42])
        EDI_dict = {R[i]: EDI[i] for i in range(len(R))}
        
        # Capacités
        CAPchaine = {}
        mean_capacity = params.get('mean_capacity', 180)
        for s in range(S):
            for t in range(T):
                CAPchaine[(s, t)] = mean_capacity
        
        # Taux de défaut par scénario
        taux_defaut = {}
        for s in range(S):
            for i in R:
                taux_defaut[(s, i)] = scenario_rates[s] / 100
        
        self.parameters = {
            'S': S, 'T': T, 'R': R, 'EDI': EDI_dict,
            'CAPchaine': CAPchaine, 'taux_defaut': taux_defaut,
            'alpha_rework': params.get('alpha_rework', 0.8),
            'beta': params.get('beta', 1.2),
            'm': params.get('m', 5),
            'penalite_penurie': params.get('penalite_penurie', 1000),
            'base_rate': base_rate,
            'uncertainty': uncertainty,
            'scenario_rates': scenario_rates
        }
        
        return True
    
    def _calculate_uncertainty(self):
        """Calcule l'incertitude basée sur l'historique"""
        if len(self.predictor.predictions_history) < 3:
            return 0.15
        
        recent_rates = [p['final_rework_rate'] for p in self.predictor.predictions_history[-10:]]
        
        if len(recent_rates) < 2:
            return 0.15
        
        mean_rate = np.mean(recent_rates)
        std_rate = np.std(recent_rates)
        
        cv = std_rate / mean_rate if mean_rate > 0 else 0.15
        return max(0.05, min(0.3, cv))
    
    def _generate_scenarios(self, base_rate, uncertainty, S):
        """Génère les scénarios de taux de rework"""
        scenarios = []
        
        if S == 1:
            scenarios = [base_rate]
        elif S == 3:
            scenarios = [
                base_rate * (1 - uncertainty),  # Optimiste
                base_rate,                      # Moyen
                base_rate * (1 + uncertainty)   # Pessimiste
            ]
        else:
            for s in range(S):
                factor = 1 + uncertainty * (2 * s / (S - 1) - 1)
                scenarios.append(base_rate * factor)
        
        return [max(0.5, min(25, rate)) for rate in scenarios]
    
    def solve_optimization(self, time_limit=300):
        """Résout le problème d'optimisation"""
        if not self.parameters:
            raise ValueError("Configurez d'abord la planification!")
        
        # Créer le modèle
        self._create_model()
        
        try:
            solver = plp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
            self.model.solve(solver)
            
            if self.model.status == plp.LpStatusOptimal:
                self._extract_results()
                return True
            else:
                st.error(f"Échec d'optimisation: {plp.LpStatus[self.model.status]}")
                return False
                
        except Exception as e:
            st.error(f"Erreur lors de la résolution: {e}")
            return False
    
    def _create_model(self):
        """Crée le modèle d'optimisation"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        
        self.model = plp.LpProblem("Planning_With_History", plp.LpMinimize)
        
        # Variables
        self.variables['q'] = plp.LpVariable.dicts(
            "q", [(s, i, t) for s in range(S) for i in R for t in range(T)],
            lowBound=0, cat='Continuous'
        )
        
        self.variables['penurie'] = plp.LpVariable.dicts(
            "penurie", [(s, i) for s in range(S) for i in R],
            lowBound=0, cat='Continuous'
        )
        
        # Contraintes
        self._add_constraints()
        self._set_objective()
    
    def _add_constraints(self):
        """Ajoute les contraintes"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        q, penurie = self.variables['q'], self.variables['penurie']
        
        # Contraintes de demande
        for s in range(S):
            for i in R:
                prod_effective = plp.lpSum([
                    q[(s, i, t)] * (1 - params['taux_defaut'][(s, i)] + 
                                   params['alpha_rework'] * params['taux_defaut'][(s, i)])
                    for t in range(T)
                ])
                
                self.model += (
                    prod_effective + penurie[(s, i)] >= params['EDI'][i],
                    f"Demande_s{s}_i{i}"
                )
        
        # Contraintes de capacité
        for s in range(S):
            for t in range(T):
                cap_utilisee = plp.lpSum([
                    q[(s, i, t)] * (1 + params['beta'] * params['taux_defaut'][(s, i)])
                    for i in R
                ])
                
                self.model += (
                    cap_utilisee <= params['CAPchaine'][(s, t)],
                    f"Capacite_s{s}_t{t}"
                )
    
    def _set_objective(self):
        """Définit la fonction objectif"""
        params = self.parameters
        S, T, R = params['S'], params['T'], params['R']
        q, penurie = self.variables['q'], self.variables['penurie']
        
        cout_production = plp.lpSum([
            (1/S) * 20 * q[(s, i, t)]
            for s in range(S) for i in R for t in range(T)
        ])
        
        cout_penuries = plp.lpSum([
            (1/S) * params['penalite_penurie'] * penurie[(s, i)]
            for s in range(S) for i in R
        ])
        
        # Prime de risque basée sur l'incertitude
        prime_risque = params['uncertainty'] * 50 * plp.lpSum([
            penurie[(s, i)] for s in range(S) for i in R
        ])
        
        self.model += cout_production + cout_penuries + prime_risque
    
    def _extract_results(self):
        """Extrait les résultats"""
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
        
        self.results = {
            'production': production_results,
            'penuries': penuries_results,
            'cout_total': self.model.objective.value(),
            'historical_context': {
                'base_rate': params['base_rate'],
                'uncertainty': params['uncertainty'],
                'scenario_rates': params['scenario_rates']
            }
        }

# =====================================================================
# FONCTIONS DE VISUALISATION STREAMLIT
# =====================================================================

def create_history_trend_chart(predictions_history):
    """Crée un graphique des tendances historiques"""
    if not predictions_history:
        return None
    
    # Préparer les données
    data = []
    for pred in predictions_history:
        data.append({
            'Timestamp': pred['timestamp'],
            'Jour': ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'][pred['jour']-1],
            'Volume': pred['volume'],
            'Taux_ML': pred['ml_prediction']['taux_rework_chaine']['moyenne_ponderee'],
            'Taux_Ajusté': pred['adjusted_prediction']['taux_rework_chaine']['moyenne_ponderee'],
            'Taux_Final': pred['final_rework_rate'],
            'Précision': pred.get('accuracy', None)
        })
    
    df = pd.DataFrame(data)
    
    # Graphique principal
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Évolution des Taux de Rework', 'Précision des Prédictions'),
        vertical_spacing=0.1
    )
    
    # Taux de rework
    fig.add_trace(
        go.Scatter(x=df['Timestamp'], y=df['Taux_ML'], 
                  name='Prédiction ML', line=dict(color='blue'), mode='lines+markers'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['Timestamp'], y=df['Taux_Ajusté'], 
                  name='Taux Ajusté', line=dict(color='orange'), mode='lines+markers'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['Timestamp'], y=df['Taux_Final'], 
                  name='Taux Final', line=dict(color='red', width=3), mode='lines+markers'),
        row=1, col=1
    )
    
    # Précision (si disponible)
    precision_data = df[df['Précision'].notna()]
    if not precision_data.empty:
        fig.add_trace(
            go.Scatter(x=precision_data['Timestamp'], y=precision_data['Précision'],
                      name='Précision (%)', line=dict(color='green'), mode='lines+markers'),
            row=2, col=1
        )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Temps", row=2, col=1)
    fig.update_yaxes(title_text="Taux de Rework (%)", row=1, col=1)
    fig.update_yaxes(title_text="Précision (%)", row=2, col=1)
    
    return fig

def create_accuracy_by_day_chart(predictions_history):
    """Crée un graphique de précision par jour"""
    if not predictions_history:
        return None
    
    # Grouper par jour
    day_data = {}
    for pred in predictions_history:
        if pred.get('accuracy') is not None:
            jour = pred['jour']
            if jour not in day_data:
                day_data[jour] = []
            day_data[jour].append(pred['accuracy'])
    
    if not day_data:
        return None
    
    # Calculer les moyennes
    days = []
    accuracies = []
    day_names = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    
    for jour in range(1, 8):
        if jour in day_data:
            days.append(day_names[jour-1])
            accuracies.append(np.mean(day_data[jour]))
    
    fig = go.Figure(data=[
        go.Bar(x=days, y=accuracies, 
               marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                           '#9467bd', '#8c564b', '#e377c2'][:len(days)])
    ])
    
    fig.update_layout(
        title="Précision Moyenne par Jour de la Semaine",
        xaxis_title="Jour",
        yaxis_title="Précision (%)",
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_planning_scenarios_chart(planning_results):
    """Crée un graphique de comparaison des scénarios de planification"""
    if not planning_results or 'historical_context' not in planning_results:
        return None
    
    context = planning_results['historical_context']
    scenario_rates = context['scenario_rates']
    
    # Données pour le graphique
    scenario_names = []
    if len(scenario_rates) == 3:
        scenario_names = ['Optimiste', 'Moyen', 'Pessimiste']
    else:
        scenario_names = [f'Scénario {i+1}' for i in range(len(scenario_rates))]
    
    fig = go.Figure(data=[
        go.Bar(x=scenario_names, y=scenario_rates,
               marker_color=['green', 'orange', 'red'][:len(scenario_rates)],
               text=[f'{rate:.2f}%' for rate in scenario_rates],
               textposition='auto')
    ])
    
    fig.update_layout(
        title="Scénarios de Taux de Rework pour la Planification",
        xaxis_title="Scénario",
        yaxis_title="Taux de Rework (%)",
        showlegend=False
    )
    
    return fig

def create_demo_data_for_download():
    """Crée des données de démonstration téléchargeables"""
    np.random.seed(42)
    data = []
    
    for day in range(1, 101):
        jour_semaine = ((day - 1) % 7) + 1
        
        if jour_semaine in [6, 7]:  # Weekend
            volume_base = 800
        else:
            volume_base = 1200
        
        volume = volume_base + np.random.normal(0, 100)
        volume = max(volume, 500)
        
        # Simulation de défauts avec corrélation jour/volume
        poste1_defauts = volume * 0.02 + jour_semaine * 0.5 + np.random.normal(0, 2)
        poste2_defauts = volume * 0.015 + jour_semaine * 0.3 + np.random.normal(0, 1.5)
        poste3_defauts = volume * 0.025 + jour_semaine * 0.4 + np.random.normal(0, 2.5)
        
        data.append({
            'Jour': jour_semaine,
            'Volume_production': max(0, volume),
            'Poste1_defauts': max(0, poste1_defauts),
            'Poste2_defauts': max(0, poste2_defauts),
            'Poste3_defauts': max(0, poste3_defauts)
        })
    
    return pd.DataFrame(data)

# =====================================================================
# INTERFACE STREAMLIT PRINCIPALE
# =====================================================================

def main():
    # En-tête principal
    st.markdown("""
    <div class="main-header">
        <h1>🏭 Système Intégré avec Historique Intelligent</h1>
        <p>Prédiction adaptative + Planification basée sur l'historique + Apprentissage continu</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialisation des variables de session
    if 'enhanced_predictor' not in st.session_state:
        st.session_state.enhanced_predictor = EnhancedStreamlitDefectPredictor()
        st.session_state.system_configured = False
        st.session_state.planning_configured = False
        st.session_state.planner = None

    # Sidebar pour navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choisissez une section:",
        [
            "🏠 Accueil",
            "📊 Configuration Système", 
            "📝 Nouvelle Demande",
            "📈 Historique & Tendances",
            "🎯 Planification Intelligente",
            "📋 Résultats Intégrés",
            "💾 Export & Statistiques"
        ]
    )

    # =================== PAGE ACCUEIL ===================
    if page == "🏠 Accueil":
        st.header("Bienvenue dans le Système Amélioré")
        
        st.markdown("""
        ### 🎯 Nouvelles Fonctionnalités
        
        **🧠 Apprentissage Continu:**
        - Historique complet des prédictions
        - Ajustement automatique basé sur les erreurs passées
        - Amélioration de la précision au fil du temps
        
        **📊 Analyse Avancée:**
        - Tendances par jour de la semaine
        - Statistiques de performance
        - Facteurs de correction intelligents
        
        **🎯 Planification Robuste:**
        - Scénarios basés sur l'incertitude historique
        - Gestion du risque adaptive
        - Optimisation avec contexte
        """)
        
        # Statut du système
        st.markdown("### 📋 Statut Actuel")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.system_configured:
                st.success("✅ Système Configuré")
            else:
                st.warning("⏳ À Configurer")
        
        with col2:
            hist_count = len(st.session_state.enhanced_predictor.predictions_history)
            if hist_count > 0:
                st.info(f"📊 {hist_count} Prédictions")
            else:
                st.warning("📭 Aucun Historique")
        
        with col3:
            if st.session_state.planning_configured:
                st.success("✅ Planning Configuré")
            else:
                st.warning("⏳ Planning À Faire")
        
        with col4:
            stats = st.session_state.enhanced_predictor.get_historical_statistics()
            if stats and 'avg_accuracy' in stats:
                st.metric("🎯 Précision Moy.", f"{stats['avg_accuracy']:.1f}%")
            else:
                st.warning("⏳ Pas de Validation")
        
        # Tendance récente
        if stats:
            st.markdown("### 📈 Tendance Récente")
            st.info(stats['recent_trend'])

    # =================== CONFIGURATION SYSTÈME ===================
    elif page == "📊 Configuration Système":
        st.header("📊 Configuration du Système")
        
        st.markdown("### 🔧 Étape 1: Source des Données")
        
        data_source = st.radio(
            "Choisissez la source:",
            ["💡 Données de démonstration", "📁 Fichier Excel"]
        )
        
        if data_source == "💡 Données de démonstration":
            n_days = st.slider("Nombre de jours à générer:", 50, 200, 100)
            
            if st.button("🚀 Configurer avec Données Demo"):
                with st.spinner("Configuration en cours..."):
                    try:
                        # Créer les données de démo
                        demo_data = create_demo_data_for_download()
                        
                        # Simuler la configuration du système
                        predictor = st.session_state.enhanced_predictor
                        predictor.postes = ['Poste1_defauts', 'Poste2_defauts', 'Poste3_defauts']
                        predictor.jour_col = 'Jour'
                        predictor.volume_col = 'Volume_production'
                        predictor.poste_weights = {
                            'Poste1_defauts': 0.4, 
                            'Poste2_defauts': 0.35, 
                            'Poste3_defauts': 0.25
                        }
                        
                        # Simuler des modèles entraînés
                        class DummyModel:
                            def __init__(self, base_rate):
                                self.base_rate = base_rate
                            
                            def predict(self, X):
                                volume = X.iloc[0, 0] if hasattr(X, 'iloc') else X[0][0]
                                jour = X.iloc[0, 1] if hasattr(X, 'iloc') else X[0][1]
                                
                                base_defects = volume * self.base_rate
                                jour_factor = 1.0 + (jour - 4) * 0.02
                                noise = np.random.normal(0, volume * 0.005)
                                
                                return [max(0, base_defects * jour_factor + noise)]
                        
                        predictor.models = {
                            'Poste1_defauts': DummyModel(0.020),
                            'Poste2_defauts': DummyModel(0.015),
                            'Poste3_defauts': DummyModel(0.025)
                        }
                        
                        st.session_state.system_configured = True
                        st.success("✅ Système configuré avec succès!")
                        
                        # Afficher les informations
                        st.markdown("### 🧠 Modèles Simulés")
                        model_info = {
                            'Poste': ['Poste1_defauts', 'Poste2_defauts', 'Poste3_defauts'],
                            'Taux Base (%)': ['2.0%', '1.5%', '2.5%'],
                            'Poids': ['40%', '35%', '25%'],
                            'Statut': ['✅ Actif', '✅ Actif', '✅ Actif']
                        }
                        st.dataframe(pd.DataFrame(model_info))
                        
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")
        
        else:  # Fichier Excel
            uploaded_file = st.file_uploader(
                "Téléchargez votre fichier Excel",
                type=['xlsx', 'xls'],
                help="Colonnes requises: Jour, Volume_production, et colonnes de défauts"
            )
            
            if uploaded_file is not None:
                try:
                    data = pd.read_excel(uploaded_file)
                    st.write("**Aperçu des données:**")
                    st.dataframe(data.head())
                    
                    if st.button("🚀 Configurer avec Fichier"):
                        # Ici vous pourriez ajouter la logique d'entraînement réelle
                        st.warning("⚠️ Entraînement réel des modèles non implémenté dans cette démo")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lecture fichier: {e}")
        
        # Bouton pour télécharger les données de démo
        st.markdown("### 💾 Télécharger Données de Démonstration")
        if st.button("📥 Générer Fichier Excel Demo"):
            demo_data = create_demo_data_for_download()
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                demo_data.to_excel(writer, sheet_name='Données_Demo', index=False)
            
            st.download_button(
                label="💾 Télécharger données_demo.xlsx",
                data=output.getvalue(),
                file_name="donnees_demo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # =================== NOUVELLE DEMANDE ===================
    elif page == "📝 Nouvelle Demande":
        st.header("📝 Nouvelle Demande de Prédiction")
        
        if not st.session_state.system_configured:
            st.warning("⚠️ Veuillez d'abord configurer le système.")
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
                max_value=3000,
                value=1200,
                step=50
            )
        
        method = st.selectbox(
            "Méthode de calcul:",
            options=['moyenne_ponderee', 'moyenne', 'max', 'somme'],
            format_func=lambda x: {
                'moyenne_ponderee': '⚖️ Moyenne Pondérée (Recommandé)',
                'moyenne': '📊 Moyenne Simple',
                'max': '🔺 Maximum',
                'somme': '➕ Somme'
            }[x]
        )
        
        # Option pour ajouter les défauts réels
        with_validation = st.checkbox("🔍 J'ai les défauts réels pour validation")
        
        actual_defects = None
        if with_validation:
            st.markdown("### 📊 Défauts Réels (pour validation)")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                poste1_real = st.number_input("Défauts Poste1:", min_value=0.0, value=0.0, step=0.1)
            with col2:
                poste2_real = st.number_input("Défauts Poste2:", min_value=0.0, value=0.0, step=0.1)
            with col3:
                poste3_real = st.number_input("Défauts Poste3:", min_value=0.0, value=0.0, step=0.1)
            
            actual_defects = {
                'Poste1_defauts': poste1_real,
                'Poste2_defauts': poste2_real,
                'Poste3_defauts': poste3_real
            }
        
        if st.button("🔮 Faire la Prédiction", type="primary"):
            with st.spinner("Prédiction en cours..."):
                try:
                    # Faire la prédiction avec historique
                    result = st.session_state.enhanced_predictor.add_new_demand_prediction(
                        jour=jour,
                        volume=volume,
                        actual_defects=actual_defects,
                        method=method
                    )
                    
                    # Afficher les résultats
                    st.markdown("### 🎯 Résultats de la Prédiction")
                    
                    # Métriques principales
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "🎯 Taux Final",
                            f"{result['final_rework_rate']:.2f}%"
                        )
                    
                    with col2:
                        ml_rate = result['ml_prediction']['taux_rework_chaine'][method]
                        st.metric(
                            "🤖 Prédiction ML",
                            f"{ml_rate:.2f}%"
                        )
                    
                    with col3:
                        adj_rate = result['adjusted_prediction']['taux_rework_chaine'][method]
                        st.metric(
                            "🔧 Taux Ajusté",
                            f"{adj_rate:.2f}%"
                        )
                    
                    with col4:
                        if result.get('accuracy') is not None:
                            st.metric(
                                "📊 Précision",
                                f"{result['accuracy']:.1f}%"
                            )
                        else:
                            st.metric("📊 Précision", "N/A")
                    
                    # Détails par poste
                    st.markdown("### 🏭 Détails par Poste")
                    
                    poste_data = []
                    for poste, defauts in result['ml_prediction']['predictions_postes'].items():
                        taux = result['ml_prediction']['taux_rework_postes'][poste]
                        real_defects = actual_defects.get(poste, "N/A") if actual_defects else "N/A"
                        
                        poste_data.append({
                            'Poste': poste,
                            'Défauts Prédits': f"{defauts:.1f}",
                            'Taux Rework (%)': f"{taux:.2f}",
                            'Défauts Réels': real_defects,
                            'Poids': f"{st.session_state.enhanced_predictor.poste_weights.get(poste, 0):.1%}"
                        })
                    
                    st.dataframe(pd.DataFrame(poste_data), use_container_width=True)
                    
                    # Facteur de correction appliqué
                    ml_final_diff = result['final_rework_rate'] - result['ml_prediction']['taux_rework_chaine'][method]
                    if abs(ml_final_diff) > 0.1:
                        correction_info = "📈 Correction à la hausse" if ml_final_diff > 0 else "📉 Correction à la baisse"
                        st.info(f"{correction_info} appliquée: {ml_final_diff:+.2f}%")
                    
                    # Message de succès
                    st.success("✅ Prédiction ajoutée à l'historique avec succès!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction: {e}")

    # =================== HISTORIQUE & TENDANCES ===================
    elif page == "📈 Historique & Tendances":
        st.header("📈 Historique et Analyse des Tendances")
        
        predictor = st.session_state.enhanced_predictor
        history = predictor.predictions_history
        
        if not history:
            st.info("📭 Aucun historique disponible. Ajoutez des prédictions d'abord.")
            return
        
        # Statistiques générales
        stats = predictor.get_historical_statistics()
        
        st.markdown("### 📊 Statistiques Générales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Prédictions", stats['total_predictions'])
        
        with col2:
            st.metric("Avec Validation", stats['validated_predictions'])
        
        with col3:
            if 'avg_accuracy' in stats:
                st.metric("Précision Moyenne", f"{stats['avg_accuracy']:.1f}%")
            else:
                st.metric("Précision Moyenne", "N/A")
        
        with col4:
            st.metric("Tendance", stats['recent_trend'].replace('📈', '').replace('📉', '').replace('📊', '').strip())
        
        # Graphique des tendances
        st.markdown("### 📈 Évolution des Prédictions")
        
        trend_chart = create_history_trend_chart(history)
        if trend_chart:
            st.plotly_chart(trend_chart, use_container_width=True)
        
        # Précision par jour
        if stats['validated_predictions'] > 0:
            st.markdown("### 📊 Performance par Jour de la Semaine")
            
            accuracy_chart = create_accuracy_by_day_chart(history)
            if accuracy_chart:
                st.plotly_chart(accuracy_chart, use_container_width=True)
        
        # Tableau détaillé de l'historique
        st.markdown("### 📋 Historique Détaillé")
        
        n_display = st.slider("Nombre d'entrées à afficher:", 5, min(50, len(history)), 10)
        
        recent_history = history[-n_display:]
        
        display_data = []
        for i, pred in enumerate(reversed(recent_history), 1):
            day_name = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'][pred['jour']-1]
            
            display_data.append({
                '#': len(history) - i + 1,
                'Date/Heure': pred['timestamp'].strftime("%Y-%m-%d %H:%M"),
                'Jour': day_name,
                'Volume': f"{pred['volume']:,}",
                'Taux Final (%)': f"{pred['final_rework_rate']:.2f}",
                'Précision (%)': f"{pred['accuracy']:.1f}" if pred.get('accuracy') else "N/A",
                'Validé': "✅" if pred.get('actual_defects') else "❌"
            })
        
        st.dataframe(pd.DataFrame(display_data), use_container_width=True)

    # =================== PLANIFICATION INTELLIGENTE ===================
    elif page == "🎯 Planification Intelligente":
        st.header("🎯 Planification Intelligente avec Historique")
        
        predictor = st.session_state.enhanced_predictor
        
        if not predictor.predictions_history:
            st.warning("⚠️ Aucune prédiction disponible. Ajoutez d'abord une demande.")
            return
        
        st.markdown("### 📋 Configuration de la Planification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            S = st.number_input("Nombre de scénarios:", min_value=1, max_value=5, value=3)
            T = st.number_input("Nombre de shifts:", min_value=1, max_value=5, value=3)
        
        with col2:
            mean_capacity = st.number_input("Capacité par shift:", min_value=100, max_value=500, value=180)
            penalite_penurie = st.number_input("Pénalité pénurie:", min_value=500, max_value=2000, value=1000)
        
        # Informations sur la dernière prédiction
        latest_pred = predictor.predictions_history[-1]
        
        st.markdown("### 🔮 Contexte de la Dernière Prédiction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Jour", ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'][latest_pred['jour']-1])
        
        with col2:
            st.metric("Volume", f"{latest_pred['volume']:,}")
        
        with col3:
            st.metric("Taux Final", f"{latest_pred['final_rework_rate']:.2f}%")
        
        if st.button("🚀 Configurer et Optimiser", type="primary"):
            with st.spinner("Configuration et optimisation en cours..."):
                try:
                    # Créer le planificateur
                    planner = StreamlitPlanningWithHistory(predictor)
                    
                    # Configurer avec l'historique
                    success = planner.configure_with_history(
                        S=S, T=T,
                        mean_capacity=mean_capacity,
                        penalite_penurie=penalite_penurie,
                        alpha_rework=0.8,
                        beta=1.2,
                        m=5
                    )
                    
                    if not success:
                        st.error("❌ Échec de la configuration")
                        return
                    
                    # Résoudre l'optimisation
                    success = planner.solve_optimization(time_limit=300)
                    
                    if success:
                        st.session_state.planner = planner
                        st.session_state.planning_configured = True
                        st.success("✅ Optimisation réussie!")
                        
                        # Afficher les résultats
                        st.markdown("### 📊 Résultats de l'Optimisation")
                        
                        results = planner.results
                        context = results['historical_context']
                        
                        # Métriques principales
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Coût Total", f"{results['cout_total']:.0f}")
                        
                        with col2:
                            st.metric("Taux Base", f"{context['base_rate']:.2f}%")
                        
                        with col3:
                            st.metric("Incertitude", f"{context['uncertainty']:.1%}")
                        
                        with col4:
                            st.metric("Scénarios", len(context['scenario_rates']))
                        
                        # Graphique des scénarios
                        st.markdown("### 📈 Scénarios Considérés")
                        
                        scenario_chart = create_planning_scenarios_chart(results)
                        if scenario_chart:
                            st.plotly_chart(scenario_chart, use_container_width=True)
                        
                        # Détails des scénarios
                        st.markdown("### 📋 Détails des Scénarios")
                        
                        scenario_data = []
                        scenario_names = ['Optimiste', 'Moyen', 'Pessimiste'] if S == 3 else [f'Scénario {i+1}' for i in range(S)]
                        
                        for i, (name, rate) in enumerate(zip(scenario_names, context['scenario_rates'])):
                            scenario_data.append({
                                'Scénario': name,
                                'Taux Rework (%)': f"{rate:.2f}",
                                'Probabilité': f"{100/S:.1f}%",
                                'Type': 'Historique + Incertitude'
                            })
                        
                        st.dataframe(pd.DataFrame(scenario_data), use_container_width=True)
                        
                    else:
                        st.error("❌ Échec de l'optimisation")
                        
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")

    # =================== RÉSULTATS INTÉGRÉS ===================
    elif page == "📋 Résultats Intégrés":
        st.header("📋 Résultats du Système Intégré")
        
        if not st.session_state.planning_configured or not st.session_state.planner:
            st.warning("⚠️ Effectuez d'abord une planification.")
            return
        
        planner = st.session_state.planner
        predictor = st.session_state.enhanced_predictor
        
        # Résumé exécutif
        st.markdown("### 📊 Résumé Exécutif")
        
        results = planner.results
        context = results['historical_context']
        stats = predictor.get_historical_statistics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔮 Prédiction:**
            - Historique: {} prédictions
            - Précision moyenne: {:.1f}%
            - Tendance: {}
            """.format(
                stats['total_predictions'],
                stats.get('avg_accuracy', 0),
                stats['recent_trend']
            ))
        
        with col2:
            st.markdown("""
            **📋 Planification:**
            - Coût optimal: {:.0f}
            - Incertitude: {:.1%}
            - Scénarios: {}
            """.format(
                results['cout_total'],
                context['uncertainty'],
                len(context['scenario_rates'])
            ))
        
        # Analyse de robustesse
        st.markdown("### 🛡️ Analyse de Robustesse")
        
        if context['uncertainty'] < 0.10:
            robustesse = "🟢 Forte - Historique stable"
        elif context['uncertainty'] < 0.20:
            robustesse = "🟡 Modérée - Variabilité contrôlée"
        else:
            robustesse = "🔴 Faible - Forte incertitude"
        
        st.info(f"**Niveau de robustesse:** {robustesse}")
        
        # Recommandations
        st.markdown("### 💡 Recommandations")
        
        recommendations = []
        
        if stats['validated_predictions'] < 5:
            recommendations.append("📊 Collecter plus de validations pour améliorer la précision")
        
        if context['uncertainty'] > 0.15:
            recommendations.append("⚖️ Considérer des marges de sécurité plus importantes")
        
        if stats.get('avg_accuracy', 0) < 80:
            recommendations.append("🔧 Réviser les paramètres des modèles de prédiction")
        
        if '📈' in stats['recent_trend']:
            recommendations.append("📈 Surveiller la tendance haussière des défauts")
        
        if not recommendations:
            recommendations.append("✅ Système performant - Continuer le monitoring")
        
        for rec in recommendations:
            st.write(f"• {rec}")

    # =================== EXPORT & STATISTIQUES ===================
    elif page == "💾 Export & Statistiques":
        st.header("💾 Export et Statistiques Avancées")
        
        predictor = st.session_state.enhanced_predictor
        
        if not predictor.predictions_history:
            st.info("📭 Aucun historique à exporter.")
            return
        
        # Statistiques détaillées
        st.markdown("### 📊 Statistiques Détaillées")
        
        stats = predictor.get_historical_statistics()
        
        # Tableau de statistiques
        if stats:
            stats_display = [
                ["📈 Total Prédictions", stats['total_predictions']],
                ["✅ Prédictions Validées", stats['validated_predictions']],
                ["🎯 Précision Moyenne", f"{stats.get('avg_accuracy', 0):.1f}%"],
                ["📊 Précision Min/Max", f"{stats.get('min_accuracy', 0):.1f}% / {stats.get('max_accuracy', 0):.1f}%"],
                ["📈 Tendance Récente", stats['recent_trend']],
                ["📏 Écart-Type Précision", f"{stats.get('std_accuracy', 0):.1f}%"]
            ]
            
            stats_df = pd.DataFrame(stats_display, columns=['Métrique', 'Valeur'])
            st.dataframe(stats_df, use_container_width=True)
        
        # Statistiques par jour
        if 'by_day' in stats and stats['by_day']:
            st.markdown("### 📅 Performance par Jour de la Semaine")
            
            day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            day_stats = []
            
            for jour in range(1, 8):
                if jour in stats['by_day']:
                    day_data = stats['by_day'][jour]
                    day_stats.append({
                        'Jour': day_names[jour-1],
                        'Nombre': day_data['count'],
                        'Précision Moyenne (%)': f"{day_data['avg_accuracy']:.1f}"
                    })
                else:
                    day_stats.append({
                        'Jour': day_names[jour-1],
                        'Nombre': 0,
                        'Précision Moyenne (%)': 'N/A'
                    })
            
            st.dataframe(pd.DataFrame(day_stats), use_container_width=True)
        
        # Export des données
        st.markdown("### 💾 Export des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Exporter Historique Excel"):
                excel_data = predictor.export_history_to_excel()
                if excel_data:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"historique_predictions_{timestamp}.xlsx"
                    
                    st.download_button(
                        label="💾 Télécharger Excel",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    st.success("✅ Fichier Excel généré!")
                else:
                    st.error("❌ Erreur lors de la génération")
        
        with col2:
            if st.button("📊 Exporter Rapport CSV"):
                # Créer un CSV simplifié
                csv_data = []
                for pred in predictor.predictions_history:
                    csv_data.append({
                        'Timestamp': pred['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                        'Jour': pred['jour'],
                        'Volume': pred['volume'],
                        'Taux_Final_%': pred['final_rework_rate'],
                        'Précision_%': pred.get('accuracy', ''),
                        'Validé': pred.get('actual_defects') is not None
                    })
                
                csv_df = pd.DataFrame(csv_data)
                csv_string = csv_df.to_csv(index=False)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rapport_predictions_{timestamp}.csv"
                
                st.download_button(
                    label="💾 Télécharger CSV",
                    data=csv_string,
                    file_name=filename,
                    mime="text/csv"
                )
        
        # Nettoyage de l'historique
        st.markdown("### 🧹 Gestion de l'Historique")
        
        st.warning("⚠️ Actions de nettoyage - Utilisez avec précaution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Supprimer Dernière Prédiction"):
                if predictor.predictions_history:
                    removed = predictor.predictions_history.pop()
                    predictor.save_history()
                    st.success(f"✅ Prédiction du {removed['timestamp'].strftime('%Y-%m-%d %H:%M')} supprimée")
                    st.experimental_rerun()
                else:
                    st.error("❌ Aucune prédiction à supprimer")
        
        with col2:
            if st.button("🧹 Vider Tout l'Historique"):
                if st.session_state.get('confirm_clear', False):
                    predictor.predictions_history = []
                    predictor.save_history()
                    st.session_state.confirm_clear = False
                    st.success("✅ Historique complètement vidé")
                    st.experimental_rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.error("⚠️ Cliquez à nouveau pour confirmer la suppression")

# =====================================================================
# FONCTIONS UTILITAIRES SUPPLÉMENTAIRES
# =====================================================================

def create_system_summary():
    """Crée un résumé du système pour la sidebar"""
    if 'enhanced_predictor' in st.session_state:
        predictor = st.session_state.enhanced_predictor
        history_count = len(predictor.predictions_history)
        
        if history_count > 0:
            stats = predictor.get_historical_statistics()
            
            st.sidebar.markdown("### 📊 Résumé Système")
            st.sidebar.metric("Historique", f"{history_count} prédictions")
            
            if 'avg_accuracy' in stats:
                st.sidebar.metric("Précision Moy.", f"{stats['avg_accuracy']:.1f}%")
            
            st.sidebar.write(f"**Tendance:** {stats['recent_trend']}")

def initialize_demo_history():
    """Initialise un historique de démonstration"""
    if st.sidebar.button("🎲 Créer Historique Demo"):
        if 'enhanced_predictor' in st.session_state:
            predictor = st.session_state.enhanced_predictor
            
            # Vider l'historique existant
            predictor.predictions_history = []
            
            # Créer quelques prédictions de démonstration
            demo_predictions = [
                (2, 1100, {'Poste1_defauts': 22, 'Poste2_defauts': 16, 'Poste3_defauts': 28}),
                (3, 1250, {'Poste1_defauts': 25, 'Poste2_defauts': 19, 'Poste3_defauts': 31}),
                (4, 1180, None),
                (5, 1350, {'Poste1_defauts': 27, 'Poste2_defauts': 20, 'Poste3_defauts': 34}),
                (1, 950, {'Poste1_defauts': 19, 'Poste2_defauts': 14, 'Poste3_defauts': 24}),
            ]
            
            for jour, volume, actual in demo_predictions:
                predictor.add_new_demand_prediction(jour, volume, actual)
            
            st.sidebar.success("✅ Historique demo créé!")
            st.experimental_rerun()

# =====================================================================
# SIDEBAR AVEC INFORMATIONS
# =====================================================================

def create_sidebar_info():
    """Crée les informations dans la sidebar"""
    st.sidebar.markdown("---")
    
    # Résumé du système
    create_system_summary()
    
    st.sidebar.markdown("---")
    
    # Actions rapides
    st.sidebar.markdown("### ⚡ Actions Rapides")
    
    # Bouton pour créer un historique de démo
    initialize_demo_history()
    
    # Informations sur la version
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Informations")
    st.sidebar.info("""
    **Version:** 2.0 Enhanced
    
    **Nouvelles fonctionnalités:**
    - 🧠 Apprentissage continu
    - 📊 Historique intelligent
    - 🔧 Ajustements automatiques
    - 📈 Analyse de tendances
    - 🛡️ Planification robuste
    """)
    
    # Contact/aide
    st.sidebar.markdown("### 📞 Support")
    st.sidebar.markdown("""
    Pour toute question:
    - 📧 [support@système.com](mailto:support@système.com)
    - 📖 [Documentation](https://docs.système.com)
    - 🐛 [Signaler un bug](https://github.com/système/issues)
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
# INSTRUCTIONS D'UTILISATION
# =====================================================================

"""
🚀 INSTRUCTIONS D'UTILISATION:

1. **Installation:**
   pip install streamlit pandas numpy plotly scikit-learn pulp openpyxl

2. **Lancement:**
   streamlit run nom_du_fichier.py

3. **Workflow Recommandé:**
   - Commencer par "Configuration Système"
   - Ajouter des "Nouvelles Demandes" avec validation
   - Analyser l'"Historique & Tendances"
   - Configurer la "Planification Intelligente"
   - Consulter les "Résultats Intégrés"
   - Exporter via "Export & Statistiques"

4. **Fonctionnalités Clés:**
   - ✅ Interface web Streamlit conservée
   - ✅ Historique complet des prédictions
   - ✅ Ajustements automatiques basés sur l'historique
   - ✅ Validation avec défauts réels
   - ✅ Planification avec scénarios intelligents
   - ✅ Visualisations interactives
   - ✅ Export Excel/CSV
   - ✅ Statistiques de performance

5. **Données de Démonstration:**
   - Utilisez le bouton "Créer Historique Demo" dans la sidebar
   - Ou configurez avec vos propres données Excel

6. **Amélioration Continue:**
   - Plus vous validez les prédictions, plus le système s'améliore
   - L'historique permet des corrections automatiques
   - Les tendances sont analysées pour optimiser les futures prédictions
"""
