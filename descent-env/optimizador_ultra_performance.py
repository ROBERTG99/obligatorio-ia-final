#!/usr/bin/env python3
"""
🚀 OPTIMIZADOR ULTRA PERFORMANCE - MÁXIMO PROCESAMIENTO POSIBLE
Sistema que usa TODA la potencia de procesamiento disponible para optimización masiva
"""

import numpy as np
import time
import os
import psutil
import json
import pickle
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Paralelización masiva
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import asyncio
import threading
from queue import Queue
import gc

# Detectar recursos del sistema
def detectar_recursos_sistema():
    """Detecta TODOS los recursos disponibles del sistema"""
    info = {
        'cpu_cores_fisicos': psutil.cpu_count(logical=False) or 1,
        'cpu_cores_logicos': psutil.cpu_count(logical=True) or 1,
        'ram_total_gb': psutil.virtual_memory().total / (1024**3),
        'ram_disponible_gb': psutil.virtual_memory().available / (1024**3),
        'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else 3000,
    }
    
    # Detectar GPU si está disponible
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu_disponible'] = True
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            info['gpu_disponible'] = False
    except ImportError:
        info['gpu_disponible'] = False
    
    return info

class OptimizadorUltraPerformance:
    """
    Optimizador que usa TODO el procesamiento disponible
    - Paralelización masiva multinivel
    - Batch processing optimizado
    - Gestión inteligente de memoria
    - Paralelización anidada
    """
    
    def __init__(self, objetivo_mejora: float = 10.0):
        self.objetivo_mejora = objetivo_mejora
        self.recursos = detectar_recursos_sistema()
        self.configurar_paralelizacion_masiva()
        
        # Configuraciones dinámicas de episodios
        self.episodios_rapido = 200        # Para exploración inicial
        self.episodios_medio = 1000        # Para candidatos prometedores  
        self.episodios_intensivo = 4000    # Para validación final
        
        # Historial y tracking
        self.historial_experimentos = []
        self.mejor_score_historico = -np.inf
        self.baseline_score = -np.inf
        
        print("🚀 OPTIMIZADOR ULTRA PERFORMANCE INICIALIZADO")
        print("="*80)
        self._mostrar_recursos_sistema()
        
    def configurar_paralelizacion_masiva(self):
        """Configura paralelización masiva usando TODOS los recursos"""
        # CPU: Usar TODOS los cores lógicos disponibles
        self.max_workers_cpu = self.recursos['cpu_cores_logicos']
        
        # Para procesos intensivos: usar cores físicos
        self.max_workers_intensivos = self.recursos['cpu_cores_fisicos']
        
        # Para tareas I/O: usar más workers que cores (2x)
        self.max_workers_io = self.recursos['cpu_cores_logicos'] * 2
        
        # Batch sizes optimizados basándose en RAM
        ram_gb = self.recursos['ram_disponible_gb']
        if ram_gb > 16:
            self.batch_size_grande = 32
            self.batch_size_medio = 16
        elif ram_gb > 8:
            self.batch_size_grande = 16
            self.batch_size_medio = 8
        else:
            self.batch_size_grande = 8
            self.batch_size_medio = 4
            
        print(f"⚡ CONFIGURACIÓN PARALELIZACIÓN MASIVA:")
        print(f"   • Workers CPU intensivos: {self.max_workers_intensivos}")
        print(f"   • Workers CPU general: {self.max_workers_cpu}")
        print(f"   • Workers I/O: {self.max_workers_io}")
        print(f"   • Batch size grande: {self.batch_size_grande}")
        print(f"   • Batch size medio: {self.batch_size_medio}")
    
    def _mostrar_recursos_sistema(self):
        """Muestra información detallada de recursos del sistema"""
        print(f"🖥️  RECURSOS DEL SISTEMA:")
        print(f"   • CPU Físicos: {self.recursos['cpu_cores_fisicos']} cores")
        print(f"   • CPU Lógicos: {self.recursos['cpu_cores_logicos']} cores")
        print(f"   • RAM Total: {self.recursos['ram_total_gb']:.1f} GB")
        print(f"   • RAM Disponible: {self.recursos['ram_disponible_gb']:.1f} GB")
        print(f"   • CPU Freq Max: {self.recursos['cpu_freq_max']:.0f} MHz")
        
        if self.recursos['gpu_disponible']:
            print(f"   • GPU: ✅ {self.recursos['gpu_count']} dispositivos")
            print(f"   • GPU Memory: {self.recursos['gpu_memory_gb']:.1f} GB")
        else:
            print(f"   • GPU: ❌ No disponible")
            
        print(f"   🎯 ESTRATEGIA: Paralelización masiva CPU + optimización memoria")
    
    def cargar_historial_completo(self):
        """Carga historial de experimentos previos para análisis retroactivo"""
        print("\n📚 CARGANDO HISTORIAL COMPLETO...")
        
        archivos_resultados = [
            'flan_results.json',
            'flan_results_10k.json', 
            'flan_results_express.json',
            'flan_results_196k_ultra.json',
            'mejoras_radicales_final_results.json',
            'mejoras_ultra_radicales_results.json',
            'optimizacion_exitosa_*.json',
            'mejor_intento_*.json'
        ]
        
        experimentos_cargados = 0
        scores_historicos = []
        
        for patron in archivos_resultados:
            if '*' in patron:
                # Buscar archivos con patrón
                archivos = list(Path('.').glob(patron))
            else:
                archivos = [Path(patron)] if Path(patron).exists() else []
            
            for archivo in archivos:
                try:
                    with open(archivo, 'r') as f:
                        data = json.load(f)
                    
                    # Extraer experimentos del archivo
                    experimentos = self._extraer_experimentos_de_json(str(archivo), data)
                    self.historial_experimentos.extend(experimentos)
                    experimentos_cargados += len(experimentos)
                    
                    # Recopilar scores
                    for exp in experimentos:
                        if 'score' in exp and exp['score'] > -900:
                            scores_historicos.append(exp['score'])
                    
                    print(f"   ✅ {archivo.name}: {len(experimentos)} experimentos")
                    
                except Exception as e:
                    print(f"   ⚠️ Error cargando {archivo}: {e}")
        
        if scores_historicos:
            self.mejor_score_historico = max(scores_historicos)
            self.baseline_score = np.median(scores_historicos)  # Usar mediana como baseline
            
            print(f"\n📊 ANÁLISIS HISTÓRICO:")
            print(f"   • Experimentos totales: {experimentos_cargados}")
            print(f"   • Scores válidos: {len(scores_historicos)}")
            print(f"   • Mejor score histórico: {self.mejor_score_historico:.2f}")
            print(f"   • Baseline (mediana): {self.baseline_score:.2f}")
            print(f"   • Objetivo nuevo: {self.mejor_score_historico + self.objetivo_mejora:.2f}")
            
            return True
        else:
            print("   ⚠️ No se encontraron scores válidos. Iniciando desde cero.")
            self.mejor_score_historico = -999
            self.baseline_score = -999
            return False
    
    def _extraer_experimentos_de_json(self, archivo: str, data: Dict) -> List[Dict]:
        """Extrae experimentos de cualquier formato JSON"""
        experimentos = []
        
        try:
            # Formato estándar FLAN
            if 'experiment_info' in data:
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
                                experimentos.append(experimento)
            
            # Formato optimización exitosa
            elif 'params_exitosos' in data:
                experimento = {
                    'archivo': archivo,
                    'esquema': 'optimizacion_exitosa',
                    'agente': 'qlearning',
                    'params': data['params_exitosos'],
                    'score': data['score_final'],
                    'timestamp': data.get('timestamp', time.time())
                }
                experimentos.append(experimento)
            
            # Formato mejor intento
            elif 'params_mejor_intento' in data:
                experimento = {
                    'archivo': archivo,
                    'esquema': 'mejor_intento',
                    'agente': 'qlearning',
                    'params': data['params_mejor_intento'],
                    'score': data['score_final'],
                    'timestamp': data.get('timestamp', time.time())
                }
                experimentos.append(experimento)
                
        except Exception as e:
            print(f"   ⚠️ Error extrayendo de {archivo}: {e}")
        
        return experimentos
    
    def generar_grid_hiperparametros_inteligente(self) -> List[Dict]:
        """Genera grid de hiperparámetros inteligente basándose en historial"""
        print("\n🧠 GENERANDO GRID DE HIPERPARÁMETROS INTELIGENTE...")
        
        if not self.historial_experimentos:
            # Sin historial: usar grid exploratorio amplio
            grid = self._grid_exploratorio_amplio()
            print(f"   • Modo: Exploración amplia ({len(grid)} configuraciones)")
        else:
            # Con historial: usar análisis inteligente
            grid = self._grid_basado_en_historial()
            print(f"   • Modo: Basado en historial ({len(grid)} configuraciones)")
        
        return grid
    
    def _grid_exploratorio_amplio(self) -> List[Dict]:
        """Grid exploratorio amplio para cuando no hay historial"""
        import itertools
        
        # Grid amplio para exploración inicial
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
            'discount_factor': [0.95, 0.99, 0.995, 0.999],
            'epsilon': [0.1, 0.3, 0.5, 0.7, 0.9],
            'epsilon_decay': [0.9995, 0.9999, 0.99995, 0.99999],
            'use_double_q': [True, False],
            'use_reward_shaping': [True, False]
        }
        
        # Generar todas las combinaciones
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        grid = []
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            grid.append(params)
        
        return grid
    
    def _grid_basado_en_historial(self) -> List[Dict]:
        """Grid inteligente basándose en mejores resultados históricos"""
        # Obtener los mejores experimentos históricos
        experimentos_ordenados = sorted(
            self.historial_experimentos,
            key=lambda x: x.get('score', -999),
            reverse=True
        )
        
        mejores_experimentos = experimentos_ordenados[:10]  # Top 10
        
        print(f"   📊 Analizando top {len(mejores_experimentos)} experimentos históricos:")
        for i, exp in enumerate(mejores_experimentos[:5], 1):
            print(f"      {i}. Score: {exp['score']:.2f} - {exp.get('agente', 'unknown')}")
        
        # Extraer rangos de parámetros prometedores
        learning_rates = [exp['params'].get('learning_rate', 0.1) for exp in mejores_experimentos if 'params' in exp]
        discount_factors = [exp['params'].get('discount_factor', 0.99) for exp in mejores_experimentos if 'params' in exp]
        epsilons = [exp['params'].get('epsilon', 0.3) for exp in mejores_experimentos if 'params' in exp]
        
        # Generar variaciones inteligentes alrededor de los mejores
        grid = []
        
        # 1. Mejores configuraciones exactas
        for exp in mejores_experimentos[:5]:
            if 'params' in exp:
                grid.append(exp['params'].copy())
        
        # 2. Variaciones alrededor de los mejores learning rates
        if learning_rates:
            best_lr = np.mean(learning_rates)
            for variation in [-0.2, -0.1, 0, 0.1, 0.2]:
                new_lr = np.clip(best_lr + variation, 0.01, 0.99)
                
                config = {
                    'learning_rate': new_lr,
                    'discount_factor': np.mean(discount_factors) if discount_factors else 0.99,
                    'epsilon': np.mean(epsilons) if epsilons else 0.3,
                    'epsilon_decay': 0.9999,
                    'use_double_q': True,
                    'use_reward_shaping': True
                }
                grid.append(config)
        
        # 3. Configuraciones híbridas ultra-agresivas
        configuraciones_agresivas = [
            {'learning_rate': 0.95, 'discount_factor': 0.99999, 'epsilon': 0.98, 'epsilon_decay': 0.999995, 'use_double_q': True, 'use_reward_shaping': True},
            {'learning_rate': 0.8, 'discount_factor': 0.999, 'epsilon': 0.9, 'epsilon_decay': 0.9999, 'use_double_q': True, 'use_reward_shaping': True},
            {'learning_rate': 0.7, 'discount_factor': 0.9999, 'epsilon': 0.85, 'epsilon_decay': 0.99995, 'use_double_q': True, 'use_reward_shaping': True},
        ]
        grid.extend(configuraciones_agresivas)
        
        return grid
    
    def ejecutar_experimento_masivo_paralelo(self, grid_hiperparametros: List[Dict]) -> Dict:
        """Ejecuta experimentos masivos en paralelo usando TODA la potencia de procesamiento"""
        print(f"\n🚀 EJECUTANDO EXPERIMENTOS MASIVOS EN PARALELO")
        print(f"   📊 Configuraciones a probar: {len(grid_hiperparametros)}")
        print(f"   ⚡ Workers paralelos: {self.max_workers_cpu}")
        print("="*60)
        
        start_time = time.time()
        
        # Fase 1: Evaluación rápida masiva (200 episodios cada uno)
        print("🔥 FASE 1: EVALUACIÓN RÁPIDA MASIVA")
        candidatos_prometedores = self._fase_evaluacion_rapida(grid_hiperparametros)
        
        # Fase 2: Evaluación intermedia de candidatos prometedores (1000 episodios)
        print(f"\n🎯 FASE 2: EVALUACIÓN INTERMEDIA ({len(candidatos_prometedores)} candidatos)")
        candidatos_finales = self._fase_evaluacion_intermedia(candidatos_prometedores)
        
        # Fase 3: Evaluación intensiva final (4000 episodios)
        print(f"\n🏆 FASE 3: EVALUACIÓN INTENSIVA FINAL ({len(candidatos_finales)} candidatos)")
        resultado_final = self._fase_evaluacion_intensiva(candidatos_finales)
        
        tiempo_total = time.time() - start_time
        
        print(f"\n✅ EXPERIMENTO MASIVO COMPLETADO")
        print(f"   ⏱️ Tiempo total: {tiempo_total/60:.1f} minutos")
        print(f"   🎯 Mejor resultado: {resultado_final['score']:.2f}")
        print(f"   💪 Configuraciones probadas: {len(grid_hiperparametros)}")
        
        return resultado_final
    
    def _fase_evaluacion_rapida(self, grid_configs: List[Dict]) -> List[Dict]:
        """Fase 1: Evaluación rápida masiva con paralelización extrema"""
        print(f"   📊 Evaluando {len(grid_configs)} configuraciones con {self.episodios_rapido} episodios cada una")
        
        # Preparar tareas para paralelización masiva
        tareas = []
        for i, config in enumerate(grid_configs):
            tarea = {
                'id': i,
                'config': config,
                'episodios': self.episodios_rapido,
                'evaluacion_episodes': 50
            }
            tareas.append(tarea)
        
        # Ejecutar en lotes para optimizar memoria
        resultados = []
        batch_size = self.batch_size_grande
        
        for i in range(0, len(tareas), batch_size):
            batch = tareas[i:i + batch_size]
            print(f"      🔄 Batch {i//batch_size + 1}/{(len(tareas)-1)//batch_size + 1}: {len(batch)} configuraciones")
            
            # Paralelización masiva del batch
            batch_results = self._ejecutar_batch_paralelo(batch)
            resultados.extend(batch_results)
            
            # Limpiar memoria
            gc.collect()
        
        # Seleccionar top candidatos (30% mejores)
        resultados_validos = [r for r in resultados if r['score'] > -900]
        resultados_ordenados = sorted(resultados_validos, key=lambda x: x['score'], reverse=True)
        
        num_candidatos = max(5, len(resultados_ordenados) // 3)  # Top 30%
        candidatos_prometedores = resultados_ordenados[:num_candidatos]
        
        print(f"   ✅ Fase 1 completada. Candidatos prometedores: {len(candidatos_prometedores)}")
        for i, candidato in enumerate(candidatos_prometedores[:5], 1):
            print(f"      {i}. Score: {candidato['score']:.2f}")
        
        return candidatos_prometedores
    
    def _fase_evaluacion_intermedia(self, candidatos: List[Dict]) -> List[Dict]:
        """Fase 2: Evaluación intermedia de candidatos prometedores"""
        print(f"   📊 Re-evaluando candidatos con {self.episodios_medio} episodios cada uno")
        
        # Actualizar configuraciones para evaluación intermedia
        tareas = []
        for i, candidato in enumerate(candidatos):
            tarea = {
                'id': i,
                'config': candidato['config'],
                'episodios': self.episodios_medio,
                'evaluacion_episodes': 200
            }
            tareas.append(tarea)
        
        # Ejecutar con paralelización intensiva (menos workers para más recursos por tarea)
        resultados = self._ejecutar_batch_paralelo(tareas, workers=self.max_workers_intensivos)
        
        # Seleccionar top candidatos finales (50% mejores)
        resultados_validos = [r for r in resultados if r['score'] > -900]
        resultados_ordenados = sorted(resultados_validos, key=lambda x: x['score'], reverse=True)
        
        num_finales = max(2, len(resultados_ordenados) // 2)  # Top 50%
        candidatos_finales = resultados_ordenados[:num_finales]
        
        print(f"   ✅ Fase 2 completada. Candidatos finales: {len(candidatos_finales)}")
        for i, candidato in enumerate(candidatos_finales, 1):
            print(f"      {i}. Score: {candidato['score']:.2f}")
        
        return candidatos_finales
    
    def _fase_evaluacion_intensiva(self, candidatos: List[Dict]) -> Dict:
        """Fase 3: Evaluación intensiva final con máximos recursos"""
        print(f"   📊 Evaluación intensiva con {self.episodios_intensivo} episodios cada uno")
        
        # Configurar para máxima precisión
        tareas = []
        for i, candidato in enumerate(candidatos):
            tarea = {
                'id': i,
                'config': candidato['config'],
                'episodios': self.episodios_intensivo,
                'evaluacion_episodes': 800
            }
            tareas.append(tarea)
        
        # Ejecutar con recursos máximos
        resultados = self._ejecutar_batch_paralelo(tareas, workers=min(len(tareas), self.max_workers_intensivos))
        
        # Seleccionar el mejor absoluto
        mejor_resultado = max(resultados, key=lambda x: x['score'])
        
        print(f"   🏆 Mejor configuración final:")
        print(f"      • Score: {mejor_resultado['score']:.2f}")
        print(f"      • Config: {mejor_resultado['config']}")
        
        return mejor_resultado
    
    def _ejecutar_batch_paralelo(self, tareas: List[Dict], workers: int = None) -> List[Dict]:
        """Ejecuta un batch de tareas en paralelo masivo"""
        if workers is None:
            workers = self.max_workers_cpu
        
        resultados = []
        
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Enviar todas las tareas
                future_to_tarea = {
                    executor.submit(ejecutar_experimento_individual, tarea): tarea 
                    for tarea in tareas
                }
                
                # Recopilar resultados
                for future in as_completed(future_to_tarea):
                    tarea = future_to_tarea[future]
                    try:
                        resultado = future.result(timeout=300)  # 5 min timeout
                        resultados.append(resultado)
                        
                        print(f"         ✅ Config {tarea['id']}: {resultado['score']:.2f}")
                        
                    except Exception as e:
                        print(f"         ❌ Config {tarea['id']}: Error - {e}")
                        # Agregar resultado fallido
                        resultados.append({
                            'id': tarea['id'],
                            'config': tarea['config'],
                            'score': -999,
                            'error': str(e)
                        })
        
        except Exception as e:
            print(f"   ❌ Error en paralelización: {e}")
        
        return resultados
    
    def optimizar_hasta_objetivo(self) -> Optional[Dict]:
        """Optimiza continuamente hasta alcanzar el objetivo"""
        print(f"\n🎯 OPTIMIZACIÓN CONTINUA HASTA OBJETIVO +{self.objetivo_mejora}")
        print("="*80)
        
        # Cargar historial
        self.cargar_historial_completo()
        
        # Calcular objetivo
        objetivo_score = self.mejor_score_historico + self.objetivo_mejora
        print(f"   📊 Score objetivo: {objetivo_score:.2f}")
        print(f"   📈 Mejora requerida: +{self.objetivo_mejora:.2f}")
        
        # Iteraciones de optimización
        max_iteraciones = 10
        for iteracion in range(max_iteraciones):
            print(f"\n🔄 ITERACIÓN {iteracion + 1}/{max_iteraciones}")
            print("-" * 60)
            
            # Generar grid inteligente
            grid = self.generar_grid_hiperparametros_inteligente()
            
            # Ejecutar experimentos masivos
            mejor_resultado = self.ejecutar_experimento_masivo_paralelo(grid)
            
            # Verificar si se alcanzó el objetivo
            mejora_conseguida = mejor_resultado['score'] - self.baseline_score
            
            print(f"\n📊 RESULTADO ITERACIÓN {iteracion + 1}:")
            print(f"   • Score conseguido: {mejor_resultado['score']:.2f}")
            print(f"   • Mejora conseguida: +{mejora_conseguida:.2f}")
            print(f"   • Objetivo era: +{self.objetivo_mejora:.2f}")
            
            # Actualizar historial
            self.historial_experimentos.append({
                'archivo': 'optimizacion_ultra_performance',
                'esquema': 'ultra_performance',
                'agente': 'qlearning',
                'params': mejor_resultado['config'],
                'score': mejor_resultado['score'],
                'iteracion': iteracion + 1,
                'timestamp': time.time()
            })
            
            if mejor_resultado['score'] > self.mejor_score_historico:
                self.mejor_score_historico = mejor_resultado['score']
            
            # Verificar objetivo
            if mejora_conseguida >= self.objetivo_mejora:
                print(f"\n🏆 ¡OBJETIVO ALCANZADO!")
                print(f"   • Iteraciones necesarias: {iteracion + 1}")
                print(f"   • Mejora total: +{mejora_conseguida:.2f}")
                
                self._guardar_resultado_exitoso(mejor_resultado, iteracion + 1)
                return mejor_resultado
        
        print(f"\n⚠️ Objetivo no alcanzado en {max_iteraciones} iteraciones")
        print(f"   • Mejor resultado: {self.mejor_score_historico:.2f}")
        print(f"   • Mejora conseguida: +{self.mejor_score_historico - self.baseline_score:.2f}")
        
        return mejor_resultado if 'mejor_resultado' in locals() else None
    
    def _guardar_resultado_exitoso(self, resultado: Dict, iteraciones: int):
        """Guarda resultado exitoso con detalles completos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        resultado_completo = {
            'exito': True,
            'timestamp': timestamp,
            'objetivo_mejora': self.objetivo_mejora,
            'iteraciones_necesarias': iteraciones,
            'baseline_score': self.baseline_score,
            'score_final': resultado['score'],
            'mejora_conseguida': resultado['score'] - self.baseline_score,
            'config_exitosa': resultado['config'],
            'recursos_sistema': self.recursos,
            'paralelizacion_usada': {
                'workers_cpu': self.max_workers_cpu,
                'workers_intensivos': self.max_workers_intensivos,
                'batch_sizes': {
                    'grande': self.batch_size_grande,
                    'medio': self.batch_size_medio
                }
            }
        }
        
        filename = f"ultra_performance_exitoso_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(resultado_completo, f, indent=2)
        
        print(f"   💾 Resultado guardado en: {filename}")

def ejecutar_experimento_individual(tarea: Dict) -> Dict:
    """Función para ejecutar un experimento individual (paralelizable)"""
    try:
        # Importar dentro de la función para evitar problemas de paralelización
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from flan_qlearning_solution import (
            DescentEnv, MockDescentEnv, DiscretizationScheme,
            QLearningAgent, QLearningTrainer, PerformanceEvaluator,
            BLUESKY_AVAILABLE, MOCK_AVAILABLE
        )
        
        # Suprimir salida para paralelización limpia
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                # Crear entorno
                if BLUESKY_AVAILABLE:
                    env = DescentEnv(render_mode=None)
                elif MOCK_AVAILABLE:
                    env = MockDescentEnv(render_mode=None)
                else:
                    raise ImportError("No hay entornos disponibles")
                
                # Configuración optimizada
                discretization = DiscretizationScheme("UltraPerf", 25, 20, 20, 20, 12)
                
                # Crear y entrenar agente
                agent = QLearningAgent(discretization, **tarea['config'])
                trainer = QLearningTrainer(env, agent, discretization)
                
                # Entrenamiento
                training_rewards = trainer.train(episodes=tarea['episodios'], verbose=False)
                
                # Evaluación
                evaluator = PerformanceEvaluator(env, agent, discretization)
                eval_results = evaluator.evaluate_multiple_episodes(
                    num_episodes=tarea['evaluacion_episodes']
                )
                
                # Calcular score
                score = np.mean(eval_results['total_rewards'])
                
                env.close()
                
                return {
                    'id': tarea['id'],
                    'config': tarea['config'],
                    'score': score,
                    'episodios_entrenamiento': tarea['episodios'],
                    'episodios_evaluacion': tarea['evaluacion_episodes'],
                    'training_final_avg': np.mean(training_rewards[-100:]) if len(training_rewards) >= 100 else np.mean(training_rewards),
                    'eval_std': np.std(eval_results['total_rewards'])
                }
                
    except Exception as e:
        return {
            'id': tarea.get('id', -1),
            'config': tarea.get('config', {}),
            'score': -999,
            'error': str(e)
        }

def main():
    """Función principal para ejecutar optimización ultra performance"""
    print("🚀 OPTIMIZADOR ULTRA PERFORMANCE")
    print("="*80)
    print("💪 OBJETIVO: Usar TODA la potencia de procesamiento disponible")
    print("🎯 ESTRATEGIA: Paralelización masiva multinivel + optimización inteligente")
    print("="*80)
    
    # Configuración del usuario
    objetivo_mejora = float(input("📊 Mejora objetivo (puntos): ") or "10")
    
    # Crear optimizador
    optimizador = OptimizadorUltraPerformance(objetivo_mejora=objetivo_mejora)
    
    # Confirmar recursos
    print(f"\n🔥 ¿PROCEDER CON PARALELIZACIÓN MASIVA?")
    print(f"   • Se usarán {optimizador.max_workers_cpu} workers CPU")
    print(f"   • Esto puede sobrecargar el sistema temporalmente")
    
    confirm = input("¿Continuar? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Operación cancelada")
        return
    
    # Ejecutar optimización
    resultado = optimizador.optimizar_hasta_objetivo()
    
    if resultado:
        print(f"\n🎉 OPTIMIZACIÓN ULTRA PERFORMANCE COMPLETADA")
        print(f"   Score final: {resultado['score']:.2f}")
        print(f"   Configuración: {resultado['config']}")
    else:
        print(f"\n⚠️ No se alcanzó el objetivo, pero se obtuvieron mejoras")

if __name__ == "__main__":
    main() 