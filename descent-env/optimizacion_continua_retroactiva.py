#!/usr/bin/env python3
"""
🎯 SISTEMA DE OPTIMIZACIÓN CONTINUA RETROACTIVA PARA PROYECTO FLAN
Analiza resultados previos y ajusta automáticamente hiperparámetros hasta conseguir +10 puntos mejora
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports (optional, fallback a métodos simples si no están disponibles)
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from scipy.optimize import minimize
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Scikit-learn no disponible. Usando métodos simples.")
    
    # Clases mock para compatibilidad
    class StandardScaler:
        def fit_transform(self, X):
            return X
        def transform(self, X):
            return X
import time
import os
import psutil
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

class OptimizadorContinuoRetroactivo:
    """
    Sistema de optimización continua que aprende de experimentos previos
    y ajusta hiperparámetros automáticamente hasta conseguir objetivos
    """
    
    def __init__(self, objetivo_mejora: float = 10.0, max_iteraciones: int = 100):
        self.objetivo_mejora = objetivo_mejora
        self.max_iteraciones = max_iteraciones
        self.historial_experimentos = []
        self.mejor_score_historico = -np.inf
        self.baseline_score = -np.inf
        self.predictor_performance = None
        self.scaler = StandardScaler()
        
        # Configuración de escalado dinámico de episodios
        self.episodios_base = 500          # Episodios iniciales (muy pocos)
        self.episodios_minimos = 200       # Mínimo absoluto para evaluación
        self.episodios_maximos = 8000      # Máximo para configuraciones muy prometedoras
        self.factor_escalado = 2.0         # Factor de aumento cuando encuentra mejoras
        self.umbral_mejora_minima = 2.0    # Mejora mínima para considerar escalar
        
        # Configuración de búsqueda adaptativa
        self.espacio_busqueda = {
            'learning_rate': (0.01, 0.99),
            'discount_factor': (0.90, 0.99999),
            'epsilon': (0.05, 0.98),
            'epsilon_decay': (0.9999, 0.999999),
            'use_double_q': [True, False],
            'use_reward_shaping': [True, False]
        }
        
        # Archivos de resultados a analizar
        self.archivos_resultados = [
            'flan_results.json',
            'flan_results_10k.json',
            'flan_results_express.json',
            'flan_results_196k_ultra.json',
            'mejoras_radicales_final_results.json',
            'mejoras_ultra_radicales_results.json',
            'results_optimized_target_30.json'
        ]
        
        print("🎯 SISTEMA DE OPTIMIZACIÓN CONTINUA RETROACTIVA")
        print("="*80)
        print(f"📊 Objetivo: +{objetivo_mejora} puntos de mejora")
        print(f"🔄 Máximo iteraciones: {max_iteraciones}")
        print(f"🧠 Modo: Aprendizaje automático + Optimización Bayesiana")
        print("="*80)
        
    def cargar_historial_completo(self):
        """Carga y analiza TODOS los resultados previos"""
        print("\n📚 CARGANDO HISTORIAL COMPLETO DE EXPERIMENTOS")
        print("="*60)
        
        datos_historicos = []
        
        for archivo in self.archivos_resultados:
            if os.path.exists(archivo):
                try:
                    with open(archivo, 'r') as f:
                        data = json.load(f)
                    
                    # Extraer datos del archivo
                    experimentos = self._extraer_experimentos_de_json(archivo, data)
                    datos_historicos.extend(experimentos)
                    
                    print(f"✅ {archivo}: {len(experimentos)} configuraciones")
                    
                except Exception as e:
                    print(f"❌ Error cargando {archivo}: {e}")
            else:
                print(f"⚠️  {archivo}: No encontrado")
        
        self.historial_experimentos = datos_historicos
        
        if datos_historicos:
            scores = [exp['score'] for exp in datos_historicos]
            self.mejor_score_historico = max(scores)
            self.baseline_score = np.mean(scores)
            
            print(f"\n📊 RESUMEN HISTÓRICO:")
            print(f"   • Experimentos cargados: {len(datos_historicos)}")
            print(f"   • Mejor score histórico: {self.mejor_score_historico:.2f}")
            print(f"   • Score promedio: {self.baseline_score:.2f}")
            print(f"   • Objetivo: {self.mejor_score_historico + self.objetivo_mejora:.2f}")
            
            return True
        else:
            print("⚠️  No se encontraron datos históricos")
            return False
    
    def _extraer_experimentos_de_json(self, archivo: str, data: Dict) -> List[Dict]:
        """Extrae experimentos individuales de un archivo JSON"""
        experimentos = []
        
        if 'experiment_info' in data:
            # Formato estándar FLAN
            for scheme_name, scheme_data in data.items():
                if scheme_name == 'experiment_info':
                    continue
                    
                if isinstance(scheme_data, dict):
                    for agent_type, agent_data in scheme_data.items():
                        if 'best_params' in agent_data and 'best_score' in agent_data:
                            experimento = {
                                'archivo': archivo,
                                'esquema': scheme_name,
                                'agente': agent_type,
                                'params': agent_data['best_params'],
                                'score': agent_data['best_score'],
                                'timestamp': os.path.getmtime(archivo)
                            }
                            
                            # Agregar métricas adicionales si están disponibles
                            if 'evaluation' in agent_data:
                                eval_data = agent_data['evaluation']
                                if 'total_rewards' in eval_data:
                                    rewards = eval_data['total_rewards']
                                    experimento.update({
                                        'std_score': np.std(rewards),
                                        'min_score': np.min(rewards),
                                        'max_score': np.max(rewards),
                                        'success_rate': np.mean([r > -50 for r in rewards])
                                    })
                            
                            experimentos.append(experimento)
        
        elif 'mejores_params' in data:
            # Formato mejoras radicales
            experimento = {
                'archivo': archivo,
                'esquema': 'mejoras_radicales',
                'agente': 'qlearning',
                'params': data['mejores_params'],
                'score': data.get('metricas_finales', {}).get('avg_reward', -999),
                'timestamp': os.path.getmtime(archivo)
            }
            experimentos.append(experimento)
        
        return experimentos
    
    def entrenar_predictor_performance(self):
        """Entrena un modelo para predecir performance basado en hiperparámetros"""
        print("\n🧠 ENTRENANDO PREDICTOR DE PERFORMANCE")
        print("="*50)
        
        if len(self.historial_experimentos) < 5:
            print("⚠️  Pocos datos históricos. Usando predictor básico.")
            return False
        
        # Preparar datos de entrenamiento
        X = []
        y = []
        
        for exp in self.historial_experimentos:
            params = exp['params']
            
            # Convertir parámetros a vector numérico
            vector_params = self._params_to_vector(params)
            if vector_params is not None:
                X.append(vector_params)
                y.append(exp['score'])
        
        if len(X) < 5:
            print("⚠️  Pocos parámetros válidos. Usando predictor básico.")
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar modelo conjunto (Random Forest + Gaussian Process)
        print(f"📊 Entrenando con {len(X)} ejemplos...")
        
        # Random Forest para capturar patrones no lineales
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        # Gaussian Process para incertidumbre
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp.fit(X_scaled, y)
        
        # Evaluar performance
        rf_score = np.mean(cross_val_score(rf, X_scaled, y, cv=min(5, len(X))))
        gp_score = np.mean(cross_val_score(gp, X_scaled, y, cv=min(5, len(X))))
        
        print(f"   Random Forest R²: {rf_score:.3f}")
        print(f"   Gaussian Process R²: {gp_score:.3f}")
        
        # Usar el mejor modelo
        if rf_score > gp_score:
            self.predictor_performance = rf
            print("   ✅ Usando Random Forest")
        else:
            self.predictor_performance = gp
            print("   ✅ Usando Gaussian Process")
        
        return True
    
    def _params_to_vector(self, params: Dict) -> Optional[np.ndarray]:
        """Convierte diccionario de parámetros a vector numérico"""
        try:
            vector = [
                params.get('learning_rate', 0.1),
                params.get('discount_factor', 0.99),
                params.get('epsilon', 0.1),
                params.get('epsilon_decay', 0.9999),
                float(params.get('use_double_q', True)),
                float(params.get('use_reward_shaping', True))
            ]
            return np.array(vector)
        except Exception:
            return None
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict:
        """Convierte vector numérico a diccionario de parámetros"""
        return {
            'learning_rate': float(np.clip(vector[0], 0.01, 0.99)),
            'discount_factor': float(np.clip(vector[1], 0.90, 0.99999)),
            'epsilon': float(np.clip(vector[2], 0.05, 0.98)),
            'epsilon_decay': float(np.clip(vector[3], 0.9999, 0.999999)),
            'use_double_q': bool(vector[4] > 0.5),
            'use_reward_shaping': bool(vector[5] > 0.5)
        }
    
    def sugerir_hiperparametros_inteligentes(self) -> Dict:
        """Sugiere hiperparámetros usando análisis retroactivo"""
        print("\n🔍 SUGIRIENDO HIPERPARÁMETROS INTELIGENTES")
        print("="*50)
        
        if not self.historial_experimentos:
            print("⚠️  Sin historial. Usando configuración conservadora.")
            return self._configuracion_conservadora()
        
        # Análisis de correlaciones
        correlaciones = self._analizar_correlaciones()
        
        # Sugerencias basadas en mejores resultados
        mejores_experimentos = sorted(
            self.historial_experimentos, 
            key=lambda x: x['score'], 
            reverse=True
        )[:5]
        
        print(f"📊 Analizando los {len(mejores_experimentos)} mejores experimentos:")
        for i, exp in enumerate(mejores_experimentos, 1):
            print(f"   {i}. Score: {exp['score']:.2f} - {exp['agente']} - {exp['archivo']}")
        
        # Si tenemos predictor entrenado, usar optimización bayesiana
        if self.predictor_performance is not None:
            return self._optimizacion_bayesiana()
        else:
            return self._interpolacion_mejores_params(mejores_experimentos)
    
    def _analizar_correlaciones(self) -> Dict:
        """Analiza correlaciones entre parámetros y performance"""
        print("   🔬 Analizando correlaciones parámetro-performance...")
        
        correlaciones = {}
        
        # Extraer datos para análisis
        data_for_analysis = []
        for exp in self.historial_experimentos:
            params = exp['params']
            score = exp['score']
            
            data_for_analysis.append({
                'learning_rate': params.get('learning_rate', 0.1),
                'discount_factor': params.get('discount_factor', 0.99),
                'epsilon': params.get('epsilon', 0.1),
                'epsilon_decay': params.get('epsilon_decay', 0.9999),
                'score': score
            })
        
        if len(data_for_analysis) > 3:
            df = pd.DataFrame(data_for_analysis)
            
            # Calcular correlaciones
            for param in ['learning_rate', 'discount_factor', 'epsilon', 'epsilon_decay']:
                if param in df.columns:
                    corr = df[param].corr(df['score'])
                    correlaciones[param] = corr
                    print(f"      • {param}: {corr:.3f}")
        
        return correlaciones
    
    def _optimizacion_bayesiana(self) -> Dict:
        """Optimización bayesiana usando predictor entrenado"""
        print("   🎯 Ejecutando optimización bayesiana...")
        
        def objective(x):
            x_scaled = self.scaler.transform([x])
            
            if hasattr(self.predictor_performance, 'predict'):
                # Random Forest
                pred = self.predictor_performance.predict(x_scaled)[0]
                return -pred  # Minimizar el negativo (maximizar predicción)
            else:
                # Gaussian Process
                pred, std = self.predictor_performance.predict(x_scaled, return_std=True)
                # Acquisition function: Upper Confidence Bound
                acquisition = pred[0] + 1.96 * std[0]
                return -acquisition
        
        # Definir límites de búsqueda
        bounds = [
            (0.01, 0.99),    # learning_rate
            (0.90, 0.99999), # discount_factor
            (0.05, 0.98),    # epsilon
            (0.9999, 0.999999), # epsilon_decay
            (0, 1),          # use_double_q (convertir a bool después)
            (0, 1)           # use_reward_shaping (convertir a bool después)
        ]
        
        # Optimización
        result = minimize(
            objective,
            x0=[0.5, 0.99, 0.3, 0.9999, 1, 1],
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # Convertir resultado a parámetros
        params = self._vector_to_params(result.x)
        
        print(f"   ✅ Parámetros optimizados: {params}")
        
        return params
    
    def _interpolacion_mejores_params(self, mejores_experimentos: List[Dict]) -> Dict:
        """Interpola entre los mejores parámetros históricos"""
        print("   📊 Interpolando entre mejores parámetros...")
        
        # Extraer parámetros de los mejores experimentos
        params_numericos = {
            'learning_rate': [],
            'discount_factor': [],
            'epsilon': [],
            'epsilon_decay': []
        }
        
        params_booleanos = {
            'use_double_q': [],
            'use_reward_shaping': []
        }
        
        for exp in mejores_experimentos:
            params = exp['params']
            
            for key in params_numericos:
                if key in params:
                    params_numericos[key].append(params[key])
            
            for key in params_booleanos:
                if key in params:
                    params_booleanos[key].append(params[key])
        
        # Calcular promedios ponderados (más peso a mejores scores)
        scores = [exp['score'] for exp in mejores_experimentos]
        weights = np.exp(np.array(scores) - max(scores))  # Softmax weights
        weights = weights / weights.sum()
        
        params_sugeridos = {}
        
        # Parámetros numéricos: promedio ponderado
        for key, values in params_numericos.items():
            if values:
                params_sugeridos[key] = np.average(values, weights=weights[:len(values)])
        
        # Parámetros booleanos: moda ponderada
        for key, values in params_booleanos.items():
            if values:
                params_sugeridos[key] = np.average(values, weights=weights[:len(values)]) > 0.5
        
        print(f"   ✅ Parámetros interpolados: {params_sugeridos}")
        
        return params_sugeridos
    
    def _configuracion_conservadora(self) -> Dict:
        """Configuración conservadora cuando no hay datos históricos"""
        return {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'epsilon': 0.3,
            'epsilon_decay': 0.9999,
            'use_double_q': True,
            'use_reward_shaping': True
        }
    
    def calcular_episodios_dinamicos(self, iteracion: int, mejor_score_reciente: float) -> int:
        """Calcula cuántos episodios usar basándose en el progreso"""
        # Episodios base para las primeras iteraciones
        if iteracion < 3:
            return self.episodios_minimos
        
        # Calcular mejora reciente
        mejora_reciente = mejor_score_reciente - self.baseline_score
        
        # Escalar episodios basándose en la mejora
        if mejora_reciente >= self.umbral_mejora_minima:
            # Buena mejora: usar más episodios
            factor = min(4.0, mejora_reciente / self.umbral_mejora_minima)
            episodios = int(self.episodios_base * factor)
        else:
            # Poca mejora: usar episodios mínimos
            episodios = self.episodios_minimos
        
        # Aplicar límites
        episodios = max(self.episodios_minimos, min(episodios, self.episodios_maximos))
        
        return episodios
    
    def ejecutar_experimento_con_params(self, params: Dict, episodios: int = None) -> Dict:
        """Ejecuta un experimento con parámetros específicos y episodios dinámicos"""
        if episodios is None:
            episodios = self.episodios_base
        
        print(f"\n🚀 EJECUTANDO EXPERIMENTO CON PARÁMETROS:")
        print(f"   {params}")
        print(f"   📊 Episodios: {episodios}")
        
        try:
            # Importar componentes necesarios
            from flan_qlearning_solution import (
                DescentEnv, MockDescentEnv, DiscretizationScheme, 
                QLearningAgent, QLearningTrainer, PerformanceEvaluator,
                BLUESKY_AVAILABLE, MOCK_AVAILABLE
            )
            
            # Crear entorno
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    if BLUESKY_AVAILABLE:
                        env = DescentEnv(render_mode=None)
                    elif MOCK_AVAILABLE:
                        env = MockDescentEnv(render_mode=None)
                    else:
                        raise ImportError("No hay entornos disponibles")
            
            # Configuración del experimento
            discretization = DiscretizationScheme("AutoOpt", 30, 25, 25, 25, 15)
            
            # Crear agente con parámetros sugeridos
            agent = QLearningAgent(discretization, **params)
            
            # Entrenamiento con episodios dinámicos
            trainer = QLearningTrainer(env, agent, discretization)
            training_rewards = trainer.train(episodes=episodios, verbose=False)
            
            # Evaluación proporcional al entrenamiento
            eval_episodes = max(50, episodios // 10)  # 10% de los episodios de entrenamiento
            evaluator = PerformanceEvaluator(env, agent, discretization)
            eval_results = evaluator.evaluate_multiple_episodes(num_episodes=eval_episodes)
            
            # Calcular métricas
            score = np.mean(eval_results['total_rewards'])
            std_score = np.std(eval_results['total_rewards'])
            
            resultado = {
                'params': params,
                'score': score,
                'std_score': std_score,
                'training_rewards': training_rewards,
                'eval_results': eval_results,
                'episodios_entrenamiento': episodios,
                'episodios_evaluacion': eval_episodes,
                'timestamp': time.time()
            }
            
            print(f"   ✅ Resultado: {score:.2f} ± {std_score:.2f}")
            
            env.close()
            
            return resultado
            
        except Exception as e:
            print(f"   ❌ Error en experimento: {e}")
            return {
                'params': params,
                'score': -999,
                'error': str(e),
                'episodios_entrenamiento': episodios,
                'timestamp': time.time()
            }
    
    def ejecutar_optimizacion_continua(self):
        """Ejecuta la optimización continua hasta alcanzar el objetivo"""
        print("\n🎯 INICIANDO OPTIMIZACIÓN CONTINUA")
        print("="*80)
        
        # Cargar historial
        if not self.cargar_historial_completo():
            print("⚠️  Sin historial previo. Iniciando desde cero.")
            self.mejor_score_historico = -999
            self.baseline_score = -999
        
        # Entrenar predictor
        self.entrenar_predictor_performance()
        
        # Objetivo dinámico
        objetivo_score = self.mejor_score_historico + self.objetivo_mejora
        
        print(f"\n📊 CONFIGURACIÓN DE OPTIMIZACIÓN:")
        print(f"   • Mejor score histórico: {self.mejor_score_historico:.2f}")
        print(f"   • Objetivo actual: {objetivo_score:.2f}")
        print(f"   • Mejora requerida: +{self.objetivo_mejora:.2f}")
        
        # Iteraciones de optimización con escalado dinámico
        mejores_resultados = []
        
        for iteracion in range(self.max_iteraciones):
            print(f"\n🔄 ITERACIÓN {iteracion + 1}/{self.max_iteraciones}")
            print("-" * 50)
            
            # Calcular episodios dinámicos basándose en el progreso
            mejor_score_actual = self.mejor_score_historico if self.mejor_score_historico > -np.inf else self.baseline_score
            episodios_dinamicos = self.calcular_episodios_dinamicos(iteracion, mejor_score_actual)
            
            print(f"   📊 Estrategia episodios: {episodios_dinamicos} episodios")
            if iteracion < 3:
                print(f"      • Modo: Exploración rápida (primeras iteraciones)")
            elif mejor_score_actual - self.baseline_score >= self.umbral_mejora_minima:
                print(f"      • Modo: Inversión intensiva (mejora detectada: +{mejor_score_actual - self.baseline_score:.1f})")
            else:
                print(f"      • Modo: Exploración eficiente (buscando mejoras)")
            
            # Sugerir nuevos hiperparámetros
            params_sugeridos = self.sugerir_hiperparametros_inteligentes()
            
            # Ejecutar experimento con episodios dinámicos
            resultado = self.ejecutar_experimento_con_params(params_sugeridos, episodios_dinamicos)
            
            # Agregar al historial
            self.historial_experimentos.append({
                'archivo': 'optimizacion_continua',
                'esquema': 'auto_optimizado',
                'agente': 'qlearning',
                'params': params_sugeridos,
                'score': resultado['score'],
                'episodios_usados': episodios_dinamicos,
                'timestamp': time.time(),
                'iteracion': iteracion + 1
            })
            
            # Actualizar mejor resultado
            if resultado['score'] > self.mejor_score_historico:
                mejora_anterior = self.mejor_score_historico - self.baseline_score
                self.mejor_score_historico = resultado['score']
                mejores_resultados.append(resultado)
                
                mejora_conseguida = resultado['score'] - self.baseline_score
                mejora_incremental = resultado['score'] - (self.mejor_score_historico if mejora_anterior > 0 else self.baseline_score)
                
                print(f"   🎉 ¡NUEVO MEJOR RESULTADO! {resultado['score']:.2f}")
                print(f"   📈 Mejora total conseguida: +{mejora_conseguida:.2f}")
                print(f"   ⚡ Mejora incremental: +{mejora_incremental:.2f}")
                print(f"   💪 Episodios invertidos: {episodios_dinamicos}")
                
                # Verificar si se alcanzó el objetivo
                if mejora_conseguida >= self.objetivo_mejora:
                    print(f"\n🏆 ¡OBJETIVO ALCANZADO!")
                    print(f"   • Mejora conseguida: +{mejora_conseguida:.2f}")
                    print(f"   • Objetivo era: +{self.objetivo_mejora:.2f}")
                    print(f"   • Iteraciones necesarias: {iteracion + 1}")
                    print(f"   • Episodios totales usados: {sum(r.get('episodios_usados', 500) for r in mejores_resultados)}")
                    
                    self._guardar_resultado_exitoso(resultado, iteracion + 1)
                    return resultado
            else:
                # No hay mejora: mostrar eficiencia
                print(f"   ⚡ Sin mejora. Episodios invertidos: {episodios_dinamicos} (eficiente)")
            
            # Re-entrenar predictor cada 10 iteraciones
            if (iteracion + 1) % 10 == 0 and SKLEARN_AVAILABLE:
                print("   🧠 Re-entrenando predictor con nuevos datos...")
                self.entrenar_predictor_performance()
        
        # Si no se alcanzó el objetivo
        print(f"\n⚠️  OBJETIVO NO ALCANZADO EN {self.max_iteraciones} ITERACIONES")
        if mejores_resultados:
            mejor_resultado = max(mejores_resultados, key=lambda x: x['score'])
            mejora_conseguida = mejor_resultado['score'] - self.baseline_score
            print(f"   • Mejor resultado: {mejor_resultado['score']:.2f}")
            print(f"   • Mejora conseguida: +{mejora_conseguida:.2f}")
            print(f"   • Mejora faltante: +{self.objetivo_mejora - mejora_conseguida:.2f}")
            
            self._guardar_mejor_intento(mejor_resultado)
            return mejor_resultado
        else:
            print("   ❌ No se consiguió ninguna mejora")
            return None
    
    def _guardar_resultado_exitoso(self, resultado: Dict, iteraciones: int):
        """Guarda el resultado exitoso"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        resultado_completo = {
            'exito': True,
            'objetivo_mejora': self.objetivo_mejora,
            'iteraciones_necesarias': iteraciones,
            'baseline_score': self.baseline_score,
            'mejor_score_historico': self.mejor_score_historico,
            'score_final': resultado['score'],
            'mejora_conseguida': resultado['score'] - self.baseline_score,
            'params_exitosos': resultado['params'],
            'timestamp': timestamp,
            'historial_completo': self.historial_experimentos[-iteraciones:]
        }
        
        # Guardar en JSON
        filename = f"optimizacion_exitosa_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(resultado_completo, f, indent=2)
        
        print(f"   💾 Resultado exitoso guardado en: {filename}")
    
    def _guardar_mejor_intento(self, mejor_resultado: Dict):
        """Guarda el mejor intento aunque no haya alcanzado el objetivo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        resultado_completo = {
            'exito': False,
            'objetivo_mejora': self.objetivo_mejora,
            'iteraciones_ejecutadas': self.max_iteraciones,
            'baseline_score': self.baseline_score,
            'mejor_score_historico_inicial': self.mejor_score_historico,
            'score_final': mejor_resultado['score'],
            'mejora_conseguida': mejor_resultado['score'] - self.baseline_score,
            'mejora_faltante': self.objetivo_mejora - (mejor_resultado['score'] - self.baseline_score),
            'params_mejor_intento': mejor_resultado['params'],
            'timestamp': timestamp,
            'recomendaciones': self._generar_recomendaciones_mejora()
        }
        
        filename = f"mejor_intento_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(resultado_completo, f, indent=2)
        
        print(f"   💾 Mejor intento guardado en: {filename}")
    
    def _generar_recomendaciones_mejora(self) -> List[str]:
        """Genera recomendaciones para mejorar el sistema"""
        recomendaciones = [
            "Aumentar número de episodios de entrenamiento",
            "Probar esquemas de discretización más finos",
            "Implementar reward shaping más agresivo",
            "Considerar algoritmos más avanzados (PPO, SAC)",
            "Aumentar diversidad en exploración de hiperparámetros"
        ]
        
        return recomendaciones

def main():
    """Función principal para ejecutar la optimización continua"""
    print("🚀 SISTEMA DE OPTIMIZACIÓN CONTINUA RETROACTIVA")
    print("="*80)
    
    # Configuración del usuario
    objetivo_mejora = float(input("📊 Mejora objetivo (puntos): ") or "10")
    max_iteraciones = int(input("🔄 Máximo iteraciones: ") or "50")
    
    # Crear optimizador
    optimizador = OptimizadorContinuoRetroactivo(
        objetivo_mejora=objetivo_mejora,
        max_iteraciones=max_iteraciones
    )
    
    # Ejecutar optimización
    resultado = optimizador.ejecutar_optimizacion_continua()
    
    if resultado:
        print(f"\n✅ OPTIMIZACIÓN COMPLETADA")
        print(f"   Score final: {resultado['score']:.2f}")
        print(f"   Parámetros: {resultado['params']}")
    else:
        print(f"\n❌ OPTIMIZACIÓN FALLIDA")
        print("   Revisar logs para más detalles")

if __name__ == "__main__":
    main() 