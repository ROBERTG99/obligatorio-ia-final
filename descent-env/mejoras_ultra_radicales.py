#!/usr/bin/env python3
"""
MEJORAS ULTRA-RADICALES PARA ALCANZAR -30

Implementa las mejoras críticas identificadas en análisis_avanzado_target_30.py:

PROBLEMAS IDENTIFICADOS:
- Q-Learning actual: -66.56 (mejor episodio: -32.85)
- Stochastic es PEOR (-3.74 puntos)
- Hiperparámetros extremos (LR: 0.7-0.8)
- Reward shaping contraproducente

MEJORAS IMPLEMENTADAS:
1. Discretización ultra-fina (50×50×50×50×50) vs (25×25×25×25×10)
2. Hiperparámetros balanceados (LR: 0.01-0.1) vs (0.7-0.9)
3. Reward shaping gradual vs masivo
4. Entrenamiento adaptativo con early stopping

EXPECTATIVA: De -66.56 a >= -30 (mejora de 36+ puntos)
"""

import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stderr, redirect_stdout
import os
import sys

# Importar entorno
try:
    from descent_env import DescentEnv
    BLUESKY_AVAILABLE = True
    print("✅ DescentEnv real cargado")
except ImportError:
    try:
        from mock_descent_env import MockDescentEnv
        DescentEnv = MockDescentEnv
        BLUESKY_AVAILABLE = False
        print("📄 Usando MockDescentEnv")
    except ImportError:
        raise ImportError("❌ Ningún entorno disponible")

import time
from typing import Dict, List, Tuple, Any, Optional
import json
import pickle

class DiscretizacionUltraFina:
    """Discretización 10x más fina que la anterior"""
    
    def __init__(self):
        # MEJORA RADICAL 1: 50×50×50×50×50 vs 25×25×25×25×10
        self.altitude_bins = 50      # vs 25 (2x más fino)
        self.velocity_bins = 50      # vs 25 (2x más fino)  
        self.target_alt_bins = 50    # vs 25 (2x más fino)
        self.runway_dist_bins = 50   # vs 25 (2x más fino)
        self.action_bins = 50        # vs 10 (5x más fino!)
        
        # Espacios más precisos
        self.altitude_space = np.linspace(-2.5, 2.5, self.altitude_bins)     # Expandido
        self.velocity_space = np.linspace(-3.5, 3.5, self.velocity_bins)     # Expandido
        self.target_alt_space = np.linspace(-2.5, 2.5, self.target_alt_bins) # Expandido
        self.runway_dist_space = np.linspace(-2.5, 2.5, self.runway_dist_bins) # Expandido
        self.action_space = np.linspace(-1, 1, self.action_bins)
        
        total_params = self.altitude_bins * self.velocity_bins * self.target_alt_bins * self.runway_dist_bins * self.action_bins
        memoria_gb = total_params * 8 / (1024**3)
        
        print(f"🔧 DISCRETIZACIÓN ULTRA-FINA:")
        print(f"   • Estados: {self.altitude_bins}×{self.velocity_bins}×{self.target_alt_bins}×{self.runway_dist_bins}")
        print(f"   • Acciones: {self.action_bins}")
        print(f"   • Parámetros Q: {total_params:,}")
        print(f"   • Memoria: {memoria_gb:.2f} GB")
        print(f"   • Mejora vs anterior: {total_params / (25*25*25*25*10):.1f}x más parámetros")
        
    def get_state(self, obs: Dict) -> Tuple[int, int, int, int]:
        """Estado con máxima precisión"""
        # Expandir rango para capturar más variabilidad
        alt = np.clip(obs['altitude'][0], -2.5, 2.5)
        vz = np.clip(obs['vz'][0], -3.5, 3.5) 
        target_alt = np.clip(obs['target_altitude'][0], -2.5, 2.5)
        runway_dist = np.clip(obs['runway_distance'][0], -2.5, 2.5)
        
        alt_idx = np.clip(np.digitize(alt, self.altitude_space) - 1, 0, self.altitude_bins - 1)
        vz_idx = np.clip(np.digitize(vz, self.velocity_space) - 1, 0, self.velocity_bins - 1)
        target_alt_idx = np.clip(np.digitize(target_alt, self.target_alt_space) - 1, 0, self.target_alt_bins - 1)
        runway_dist_idx = np.clip(np.digitize(runway_dist, self.runway_dist_space) - 1, 0, self.runway_dist_bins - 1)
        
        return alt_idx, vz_idx, target_alt_idx, runway_dist_idx
    
    def get_action_index(self, action: float) -> int:
        """Acción con máxima precisión"""
        action = np.clip(action, -1, 1)
        action_idx = np.digitize(action, self.action_space) - 1
        return int(np.clip(action_idx, 0, self.action_bins - 1))
    
    def get_action_from_index(self, action_idx: int) -> float:
        """Convertir índice a acción"""
        return self.action_space[action_idx]

class RewardShaperAdaptativo:
    """Reward shaping que SE ADAPTA al progreso del aprendizaje"""
    
    def __init__(self):
        self.episodio = 0
        self.mejor_recompensa_hasta_ahora = -np.inf
        self.episodios_sin_mejora = 0
        self.fase_aprendizaje = "inicial"  # inicial, intermedia, avanzada
        
        # Tracking para análisis
        self.prev_altitude_error = None
        self.steps = 0
        
    def determinar_fase_aprendizaje(self):
        """Determina la fase actual del aprendizaje"""
        if self.episodio < 5000:
            self.fase_aprendizaje = "inicial"
        elif self.episodio < 20000:
            self.fase_aprendizaje = "intermedia"
        else:
            self.fase_aprendizaje = "avanzada"
    
    def shape_reward(self, obs: Dict, action: float, reward: float, done: bool) -> float:
        """Reward shaping que SE ADAPTA según el progreso"""
        
        self.determinar_fase_aprendizaje()
        shaped_reward = reward
        
        # Obtener métricas
        current_alt = obs['altitude'][0]
        target_alt = obs['target_altitude'][0]
        runway_dist = obs['runway_distance'][0]
        vz = obs['vz'][0]
        altitude_error = abs(target_alt - current_alt)
        
        # CLAVE: Shaping adapta según fase de aprendizaje
        if self.fase_aprendizaje == "inicial":
            # Fase inicial: Reward shaping MUY conservador para estabilidad
            if altitude_error < 0.1:
                shaped_reward += 5.0
            elif altitude_error < 0.2:
                shaped_reward += 2.0
                
        elif self.fase_aprendizaje == "intermedia":
            # Fase intermedia: Aumentar incentivos gradualmente
            if altitude_error < 0.05:
                shaped_reward += 20.0
            elif altitude_error < 0.1:
                shaped_reward += 10.0
            elif altitude_error < 0.2:
                shaped_reward += 5.0
                
        else:  # avanzada
            # Fase avanzada: Incentivos altos para refinamiento final
            if altitude_error < 0.01:
                shaped_reward += 100.0
            elif altitude_error < 0.02:
                shaped_reward += 50.0
            elif altitude_error < 0.05:
                shaped_reward += 25.0
            elif altitude_error < 0.1:
                shaped_reward += 12.0
        
        # Bonificación por mejora consistente en todas las fases
        if self.prev_altitude_error is not None:
            improvement = self.prev_altitude_error - altitude_error
            if self.fase_aprendizaje == "inicial":
                shaped_reward += improvement * 2
            elif self.fase_aprendizaje == "intermedia":
                shaped_reward += improvement * 5
            else:
                shaped_reward += improvement * 10
        
        # Penalización por acciones extremas (todas las fases)
        action_penalty = abs(action) * 0.5
        shaped_reward -= action_penalty
        
        # Velocidad vertical apropiada (gradual)
        target_vz = -0.1 if current_alt > target_alt else 0.1
        vz_error = abs(vz - target_vz)
        
        if self.fase_aprendizaje == "inicial":
            shaped_reward += max(0, (0.5 - vz_error) * 2)
        else:
            shaped_reward += max(0, (0.3 - vz_error) * 5)
        
        # Bonus final (escalado por fase)
        if done and runway_dist <= 0:
            if self.fase_aprendizaje == "inicial":
                bonus_scale = 10
            elif self.fase_aprendizaje == "intermedia":
                bonus_scale = 30
            else:
                bonus_scale = 50
                
            if altitude_error < 0.01:
                shaped_reward += bonus_scale * 5
            elif altitude_error < 0.02:
                shaped_reward += bonus_scale * 3
            elif altitude_error < 0.05:
                shaped_reward += bonus_scale * 2
            elif altitude_error < 0.1:
                shaped_reward += bonus_scale
        
        self.prev_altitude_error = altitude_error
        self.steps += 1
        
        return shaped_reward
    
    def reset_episodio(self, recompensa_episodio: float):
        """Reset con tracking de progreso"""
        self.episodio += 1
        
        if recompensa_episodio > self.mejor_recompensa_hasta_ahora:
            self.mejor_recompensa_hasta_ahora = recompensa_episodio
            self.episodios_sin_mejora = 0
        else:
            self.episodios_sin_mejora += 1
        
        self.prev_altitude_error = None
        self.steps = 0

class QLearningUltraBalanceado:
    """Q-Learning con hiperparámetros científicamente balanceados"""
    
    def __init__(self, discretization,
                 learning_rate: float = 0.05,      # vs 0.8 anterior
                 discount_factor: float = 0.99,    # vs 0.999 anterior  
                 epsilon: float = 0.3,              # vs 0.05 anterior
                 epsilon_min: float = 0.02,         # Exploración mínima
                 epsilon_decay: float = 0.99995):   # Decay muy gradual
        
        self.discretization = discretization
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Tabla Q para discretización ultra-fina
        shape = (discretization.altitude_bins,
                discretization.velocity_bins,
                discretization.target_alt_bins,
                discretization.runway_dist_bins,
                discretization.action_bins)
        
        self.Q = np.zeros(shape)
        self.visits = np.zeros(shape)  # Para learning rate adaptativo
        
        # Reward shaper adaptativo
        self.reward_shaper = RewardShaperAdaptativo()
        
        # Métricas de entrenamiento
        self.episodios_entrenados = 0
        self.recompensas_recientes = []
        
        print(f"🤖 QLearningUltraBalanceado:")
        print(f"   • Learning rate: {learning_rate} (vs 0.8 anterior)")
        print(f"   • Discount: {discount_factor} (vs 0.999 anterior)")  
        print(f"   • Epsilon inicial: {epsilon} (vs 0.05 anterior)")
        print(f"   • Epsilon decay: {epsilon_decay}")
        print(f"   • Tabla Q: {shape}")
        
    def get_action(self, state: Tuple[int, int, int, int], training: bool = True) -> float:
        """Selección de acción epsilon-greedy balanceada"""
        if training and np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            action_idx = np.random.randint(0, self.discretization.action_bins)
        else:
            # Explotación: mejor acción conocida
            q_values = self.Q[state]
            action_idx = int(np.argmax(q_values))
        
        return self.discretization.get_action_from_index(action_idx)
    
    def update(self, state, action, reward, next_state, done, obs):
        """Actualización Q-Learning con reward shaping adaptativo"""
        
        action_idx = self.discretization.get_action_index(action)
        
        # Aplicar reward shaping adaptativo
        reward = self.reward_shaper.shape_reward(obs, action, reward, done)
        
        # Learning rate adaptativo MUY gradual
        self.visits[state][action_idx] += 1
        # Adaptación más conservadora que antes
        alpha = self.learning_rate / (1 + 0.0001 * self.visits[state][action_idx])
        
        # Actualización Q-Learning clásica
        current_q = self.Q[state][action_idx]
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.Q[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # Actualización conservadora
        self.Q[state][action_idx] = current_q + alpha * (target_q - current_q)
        
        # Epsilon decay muy gradual
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def finish_episode(self, total_reward):
        """Terminar episodio con métricas"""
        self.episodios_entrenados += 1
        self.recompensas_recientes.append(total_reward)
        
        # Mantener ventana de 1000 episodios
        if len(self.recompensas_recientes) > 1000:
            self.recompensas_recientes.pop(0)
        
        # Reset reward shaper
        self.reward_shaper.reset_episodio(total_reward)
        
    def get_progress_metrics(self):
        """Obtener métricas de progreso"""
        if len(self.recompensas_recientes) < 10:
            return None
            
        return {
            'avg_recent': np.mean(self.recompensas_recientes[-100:]) if len(self.recompensas_recientes) >= 100 else np.mean(self.recompensas_recientes),
            'best_recent': np.max(self.recompensas_recientes[-100:]) if len(self.recompensas_recientes) >= 100 else np.max(self.recompensas_recientes),
            'epsilon': self.epsilon,
            'episodios': self.episodios_entrenados,
            'fase_aprendizaje': self.reward_shaper.fase_aprendizaje
        }

def buscar_hiperparametros_balanceados():
    """Búsqueda sistemática de hiperparámetros balanceados"""
    
    print("\n🔍 BÚSQUEDA DE HIPERPARÁMETROS BALANCEADOS")
    print("="*55)
    print("🎯 Objetivo: Encontrar configuración estable para alcanzar -30")
    
    # Grid balanceado (no extremo)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],           # vs [0.7, 0.8, 0.9] anterior
        'discount_factor': [0.95, 0.99],              # vs [0.999] anterior
        'epsilon': [0.2, 0.3, 0.4],                   # vs [0.05] anterior  
        'epsilon_decay': [0.9999, 0.99995, 0.999995]  # Muy gradual
    }
    
    print("📊 Grid de búsqueda (BALANCEADO vs EXTREMO anterior):")
    for param, values in param_grid.items():
        print(f"   • {param}: {values}")
    
    import itertools
    combinaciones = list(itertools.product(*param_grid.values()))
    print(f"   • Total combinaciones: {len(combinaciones)}")
    print(f"   • Episodios por combinación: 3,000 (entrenamiento rápido)")
    print(f"   • Evaluación: 200 episodios")
    
    # Crear entorno
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            env = DescentEnv(render_mode=None)
    
    discretization = DiscretizacionUltraFina()
    
    best_params = None
    best_score = -np.inf
    results = []
    
    param_names = list(param_grid.keys())
    
    for i, combination in enumerate(combinaciones):
        params = dict(zip(param_names, combination))
        
        print(f"\n🧪 Combinación {i+1}/{len(combinaciones)}: {params}")
        
        # Crear y entrenar agente
        agent = QLearningUltraBalanceado(discretization, **params)
        
        # Entrenamiento rápido
        for episode in range(3000):
            obs, _ = env.reset()
            state = discretization.get_state(obs)
            
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < 500:
                action = agent.get_action(state, training=True)
                next_obs, reward, done, _, _ = env.step(np.array([action]))
                next_state = discretization.get_state(next_obs)
                
                agent.update(state, action, reward, next_state, done, next_obs)
                
                total_reward += reward
                state = next_state
                step += 1
            
            agent.finish_episode(total_reward)
        
        # Evaluación
        eval_rewards = []
        for episode in range(200):
            obs, _ = env.reset()
            state = discretization.get_state(obs)
            
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < 500:
                action = agent.get_action(state, training=False)
                obs, reward, done, _, _ = env.step(np.array([action]))
                next_state = discretization.get_state(obs)
                
                total_reward += reward
                state = next_state
                step += 1
            
            eval_rewards.append(total_reward)
        
        score = np.mean(eval_rewards)
        best_eval = np.max(eval_rewards)
        std_eval = np.std(eval_rewards)
        
        results.append({
            'params': params,
            'score': score,
            'best_eval': best_eval,
            'std_eval': std_eval,
            'final_epsilon': agent.epsilon
        })
        
        print(f"   📊 Score: {score:.2f}, Mejor: {best_eval:.2f}, Std: {std_eval:.2f}")
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"   🏆 ¡NUEVO MEJOR!")
    
    print(f"\n🏆 MEJORES HIPERPARÁMETROS BALANCEADOS:")
    print(f"   • Parámetros: {best_params}")
    print(f"   • Score: {best_score:.2f}")
    print(f"   • Mejora vs anterior (-66.56): {best_score - (-66.56):.2f} puntos")
    
    return best_params, results

def entrenar_con_early_stopping(best_params, max_episodes=50000):
    """Entrenamiento con early stopping inteligente"""
    
    print(f"\n🚀 ENTRENAMIENTO CON EARLY STOPPING")
    print(f"="*45)
    print(f"📊 Configuración:")
    print(f"   • Hiperparámetros: {best_params}")
    print(f"   • Máximo episodios: {max_episodes:,}")
    print(f"   • Early stopping si promedio >= -30 por 1000 episodios")
    
    # Crear entorno y agente
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            env = DescentEnv(render_mode=None)
    
    discretization = DiscretizacionUltraFina()
    agent = QLearningUltraBalanceado(discretization, **best_params)
    
    # Variables de seguimiento
    episode_rewards = []
    episodios_exitosos = []  # >= -30
    check_convergencia = 1000
    
    print(f"\n🎯 Iniciando entrenamiento inteligente...")
    start_time = time.time()
    
    for episode in range(max_episodes):
        obs, _ = env.reset()
        state = discretization.get_state(obs)
        
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < 500:
            action = agent.get_action(state, training=True)
            next_obs, reward, done, _, _ = env.step(np.array([action]))
            next_state = discretization.get_state(next_obs)
            
            agent.update(state, action, reward, next_state, done, next_obs)
            
            total_reward += reward
            state = next_state
            step += 1
        
        episode_rewards.append(total_reward)
        agent.finish_episode(total_reward)
        
        if total_reward >= -30:
            episodios_exitosos.append(episode)
        
        # Reportar progreso cada 2000 episodios
        if (episode + 1) % 2000 == 0:
            metrics = agent.get_progress_metrics()
            exitosos_recientes = sum(1 for r in episode_rewards[-check_convergencia:] if r >= -30)
            success_rate = exitosos_recientes / min(len(episode_rewards), check_convergencia) * 100
            
            elapsed = time.time() - start_time
            
            print(f"📈 Episodio {episode + 1:,}:")
            print(f"   • Promedio reciente: {metrics['avg_recent']:.2f}")
            print(f"   • Mejor reciente: {metrics['best_recent']:.2f}")
            print(f"   • Épsilon: {metrics['epsilon']:.5f}")
            print(f"   • Fase: {metrics['fase_aprendizaje']}")
            print(f"   • Éxito >= -30: {exitosos_recientes}/{min(len(episode_rewards), check_convergencia)} ({success_rate:.1f}%)")
            print(f"   • Tiempo: {elapsed/3600:.1f}h")
        
        # EARLY STOPPING: Si conseguimos objetivo consistente
        if len(episode_rewards) >= check_convergencia:
            avg_recent = np.mean(episode_rewards[-check_convergencia:])
            exitosos_recientes = sum(1 for r in episode_rewards[-check_convergencia:] if r >= -30)
            success_rate = exitosos_recientes / check_convergencia
            
            # Condiciones de éxito estrictas
            if avg_recent >= -30 and success_rate >= 0.3:  # 30% de éxito mínimo
                print(f"\n🎉 ¡EARLY STOPPING EXITOSO EN EPISODIO {episode + 1}!")
                print(f"   • Promedio últimos {check_convergencia}: {avg_recent:.2f} >= -30")
                print(f"   • Tasa de éxito: {success_rate*100:.1f}% >= 30%")
                print(f"   • Criterio alcanzado: OBJETIVO LOGRADO")
                break
            
            # Early stopping por convergencia (sin mejora significativa)
            elif episode >= 30000:  # Después de 30k episodios
                recent_std = np.std(episode_rewards[-2000:])
                if recent_std < 5.0:  # Poca variabilidad = convergencia
                    print(f"\n⚠️  Early stopping por convergencia en episodio {episode + 1}")
                    print(f"   • Promedio: {avg_recent:.2f}")
                    print(f"   • Std reciente: {recent_std:.2f} < 5.0")
                    print(f"   • Criterio: Convergencia sin mejora")
                    break
    
    elapsed_total = time.time() - start_time
    print(f"\n✅ Entrenamiento completado")
    print(f"   • Tiempo total: {elapsed_total/3600:.2f} horas")
    print(f"   • Episodios totales: {len(episode_rewards):,}")
    print(f"   • Episodios exitosos: {len(episodios_exitosos)}")
    
    return agent, episode_rewards, episodios_exitosos

def evaluar_resultado_final(agent, discretization, episodes=1000):
    """Evaluación final exhaustiva"""
    
    print(f"\n🔍 EVALUACIÓN FINAL EXHAUSTIVA ({episodes} episodios)")
    print("="*55)
    
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            env = DescentEnv(render_mode=None)
    
    eval_rewards = []
    episodios_exitosos = 0
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretization.get_state(obs)
        
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < 500:
            action = agent.get_action(state, training=False)
            obs, reward, done, _, _ = env.step(np.array([action]))
            next_state = discretization.get_state(obs)
            
            total_reward += reward
            state = next_state
            step += 1
        
        eval_rewards.append(total_reward)
        
        if total_reward >= -30:
            episodios_exitosos += 1
        
        if (episode + 1) % 200 == 0:
            current_avg = np.mean(eval_rewards)
            current_success = episodios_exitosos / (episode + 1) * 100
            print(f"   📊 Episodio {episode + 1}: Promedio {current_avg:.2f}, Éxito {current_success:.1f}%")
    
    return eval_rewards, episodios_exitosos

def main():
    """Función principal de mejoras ultra-radicales"""
    
    print("🎯 MEJORAS ULTRA-RADICALES PARA ALCANZAR -30")
    print("="*50)
    print("🚨 PROBLEMA: Q-Learning actual -66.56, Objetivo: -30")
    print("🎯 ESTRATEGIA: Discretización ultra-fina + hiperparámetros balanceados")
    print("📈 EXPECTATIVA: Mejora de 36+ puntos para alcanzar objetivo")
    
    # FASE 1: Búsqueda de hiperparámetros balanceados
    print(f"\n{'='*60}")
    print("FASE 1: BÚSQUEDA DE HIPERPARÁMETROS BALANCEADOS")
    print('='*60)
    
    best_params, search_results = buscar_hiperparametros_balanceados()
    
    # FASE 2: Entrenamiento con early stopping
    print(f"\n{'='*60}")
    print("FASE 2: ENTRENAMIENTO CON EARLY STOPPING")
    print('='*60)
    
    agent, training_rewards, episodios_exitosos = entrenar_con_early_stopping(best_params)
    
    # FASE 3: Evaluación final
    print(f"\n{'='*60}")
    print("FASE 3: EVALUACIÓN FINAL")
    print('='*60)
    
    discretization = DiscretizacionUltraFina()
    eval_rewards, eval_exitosos = evaluar_resultado_final(agent, discretization)
    
    # FASE 4: Análisis de resultados
    print(f"\n{'='*70}")
    print("📊 RESULTADOS FINALES DE MEJORAS ULTRA-RADICALES")
    print('='*70)
    
    avg_reward = np.mean(eval_rewards)
    success_rate = eval_exitosos / len(eval_rewards) * 100
    objetivo_alcanzado = avg_reward >= -30
    
    # Comparación con situación anterior
    situacion_anterior = -66.56
    mejora_absoluta = avg_reward - situacion_anterior
    mejora_porcentual = (mejora_absoluta / abs(situacion_anterior)) * 100
    
    print(f"🎯 MÉTRICAS PRINCIPALES:")
    print(f"   • Situación anterior: {situacion_anterior:.2f}")
    print(f"   • Resultado actual: {avg_reward:.2f}")
    print(f"   • Mejora absoluta: +{mejora_absoluta:.2f} puntos")
    print(f"   • Mejora porcentual: +{mejora_porcentual:.1f}%")
    print(f"   • Mejor episodio: {np.max(eval_rewards):.2f}")
    print(f"   • Percentil 90: {np.percentile(eval_rewards, 90):.2f}")
    print(f"   • Percentil 75: {np.percentile(eval_rewards, 75):.2f}")
    print(f"   • Desviación estándar: {np.std(eval_rewards):.2f}")
    
    print(f"\n🏆 RENDIMIENTO:")
    print(f"   • Episodios >= -30: {eval_exitosos}/1000")
    print(f"   • Tasa de éxito: {success_rate:.1f}%")
    print(f"   • Episodios exitosos durante entrenamiento: {len(episodios_exitosos)}")
    
    print(f"\n🎯 OBJETIVO: {'✅ ALCANZADO' if objetivo_alcanzado else '❌ NO ALCANZADO'}")
    
    if objetivo_alcanzado:
        print(f"🎉 ¡ÉXITO TOTAL! Recompensa promedio {avg_reward:.2f} >= -30")
        superacion = avg_reward + 30
        print(f"💪 Superó el objetivo por {superacion:.2f} puntos")
        print(f"🚀 Las mejoras ultra-radicales FUNCIONARON!")
    else:
        falta = -30 - avg_reward
        print(f"⚠️  Falta {falta:.2f} puntos para alcanzar -30")
        print(f"📈 Pero mejoró significativamente: +{mejora_absoluta:.2f} puntos")
        
        if mejora_absoluta > 20:
            print(f"💡 Mejora SIGNIFICATIVA - continuar con más entrenamiento")
        elif mejora_absoluta > 10:
            print(f"💡 Mejora moderada - ajustar hiperparámetros o probar algoritmos continuos")
        else:
            print(f"⚠️  Mejora pequeña - considerar cambio de paradigma (DDPG/TD3/SAC)")
    
    # Guardar resultados completos
    results = {
        'mejores_params': best_params,
        'busqueda_resultados': search_results,
        'entrenamiento_rewards': training_rewards,
        'episodios_exitosos_entrenamiento': episodios_exitosos,
        'evaluacion_rewards': eval_rewards,
        'evaluacion_exitosos': eval_exitosos,
        'metricas_finales': {
            'avg_reward': float(avg_reward),
            'success_rate': float(success_rate),
            'objetivo_alcanzado': objetivo_alcanzado,
            'mejora_absoluta': float(mejora_absoluta),
            'mejora_porcentual': float(mejora_porcentual),
            'mejor_episodio': float(np.max(eval_rewards)),
            'percentil_90': float(np.percentile(eval_rewards, 90)),
            'std_dev': float(np.std(eval_rewards))
        }
    }
    
    # Serializar para JSON
    results_json = {}
    for key, value in results.items():
        if isinstance(value, (list, np.ndarray)):
            if len(value) > 0 and isinstance(value[0], (np.float64, np.float32, np.int64)):
                results_json[key] = [float(x) for x in value]
            else:
                results_json[key] = list(value)
        else:
            results_json[key] = value
    
    with open('mejoras_ultra_radicales_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n💾 Resultados guardados en 'mejoras_ultra_radicales_results.json'")
    
    # Recomendaciones finales
    print(f"\n📋 PRÓXIMOS PASOS RECOMENDADOS:")
    if objetivo_alcanzado:
        print(f"   1. ✅ ¡Objetivo alcanzado! Guardar modelo final")
        print(f"   2. 📊 Generar reporte detallado de éxito")
        print(f"   3. 🔬 Analizar qué mejoras fueron más efectivas")
    else:
        if mejora_absoluta > 20:
            print(f"   1. 🔄 Continuar entrenamiento por más episodios")
            print(f"   2. 🎯 Ajustar reward shaping para fase final")
            print(f"   3. 📊 Probar ensemble de múltiples agentes")
        else:
            print(f"   1. 🧠 Migrar a algoritmos deep learning (DDPG/TD3/SAC)")
            print(f"   2. 🔄 Cambio de paradigma a control continuo")
            print(f"   3. 📚 Q-Learning tabular puede haber alcanzado su límite")
    
    return results

if __name__ == "__main__":
    results = main() 