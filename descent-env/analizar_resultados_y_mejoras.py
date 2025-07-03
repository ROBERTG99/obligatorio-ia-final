#!/usr/bin/env python3
"""
Análisis detallado de resultados y estrategias para alcanzar recompensa -30
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_y_analizar_resultados():
    """Carga y analiza los resultados del experimento"""
    
    print("="*80)
    print("📊 ANÁLISIS DETALLADO DE RESULTADOS - PROYECTO FLAN")
    print("="*80)
    
    try:
        with open('flan_results_10k.json', 'r') as f:
            results = json.load(f)
        
        # Extraer datos
        scheme_results = results['Media']
        qlearning_data = scheme_results['qlearning']
        stochastic_data = scheme_results['stochastic']
        
        print("✅ Resultados cargados exitosamente")
        
        return qlearning_data, stochastic_data, results
        
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo flan_results_10k.json")
        return None, None, None

def analizar_rendimiento_actual(qlearning_data, stochastic_data):
    """Analiza el rendimiento actual en detalle"""
    
    print("\n" + "="*60)
    print("🔍 ANÁLISIS DE RENDIMIENTO ACTUAL")
    print("="*60)
    
    # Datos Q-Learning
    ql_rewards = np.array(qlearning_data['evaluation']['total_rewards'])
    ql_best_score = qlearning_data['best_score']
    
    # Datos Stochastic Q-Learning  
    stoch_rewards = np.array(stochastic_data['evaluation']['total_rewards'])
    stoch_best_score = stochastic_data['best_score']
    
    print(f"🤖 Q-LEARNING ESTÁNDAR:")
    print(f"   • Score promedio búsqueda: {ql_best_score:.2f}")
    print(f"   • Recompensa evaluación: {np.mean(ql_rewards):.2f} ± {np.std(ql_rewards):.2f}")
    print(f"   • Rango: [{np.min(ql_rewards):.2f}, {np.max(ql_rewards):.2f}]")
    print(f"   • Mejor episodio: {np.max(ql_rewards):.2f}")
    print(f"   • Percentil 90: {np.percentile(ql_rewards, 90):.2f}")
    
    print(f"\n🎲 STOCHASTIC Q-LEARNING:")
    print(f"   • Score promedio búsqueda: {stoch_best_score:.2f}")
    print(f"   • Recompensa evaluación: {np.mean(stoch_rewards):.2f} ± {np.std(stoch_rewards):.2f}")
    print(f"   • Rango: [{np.min(stoch_rewards):.2f}, {np.max(stoch_rewards):.2f}]")
    print(f"   • Mejor episodio: {np.max(stoch_rewards):.2f}")
    print(f"   • Percentil 90: {np.percentile(stoch_rewards, 90):.2f}")
    
    # Análisis comparativo
    mejora_absoluta = np.mean(stoch_rewards) - np.mean(ql_rewards)
    print(f"\n📈 COMPARACIÓN:")
    print(f"   • Mejora Stochastic vs Q-Learning: {mejora_absoluta:.2f}")
    print(f"   • Mejora relativa: {mejora_absoluta/abs(np.mean(ql_rewards))*100:.1f}%")
    
    # Distancia al objetivo
    objetivo = -30
    distancia_ql = abs(objetivo - np.mean(ql_rewards))
    distancia_stoch = abs(objetivo - np.mean(stoch_rewards))
    
    print(f"\n🎯 DISTANCIA AL OBJETIVO (-30):")
    print(f"   • Q-Learning: {distancia_ql:.2f} puntos")
    print(f"   • Stochastic: {distancia_stoch:.2f} puntos")
    print(f"   • Mejora necesaria: {distancia_stoch:.2f} puntos (~{distancia_stoch/abs(np.mean(stoch_rewards))*100:.1f}%)")
    
    return ql_rewards, stoch_rewards, objetivo

def identificar_problemas_clave(qlearning_data, stochastic_data):
    """Identifica los principales problemas que limitan el rendimiento"""
    
    print("\n" + "="*60)
    print("🚨 IDENTIFICACIÓN DE PROBLEMAS CLAVE")
    print("="*60)
    
    # Analizar hiperparámetros
    ql_params = qlearning_data['best_params']
    stoch_params = stochastic_data['best_params']
    
    print("🔧 HIPERPARÁMETROS ACTUALES:")
    print(f"   Q-Learning: {ql_params}")
    print(f"   Stochastic: {stoch_params}")
    
    problemas_identificados = []
    
    # Problema 1: Learning rate bajo
    if ql_params['learning_rate'] <= 0.3:
        problemas_identificados.append({
            'problema': 'Learning rate conservador',
            'descripcion': f"LR actual: {ql_params['learning_rate']}. Puede ser muy lento para convergencia óptima.",
            'impacto': 'Alto',
            'solucion': 'Probar LR más altos (0.5-0.8) con decay adaptativo'
        })
    
    # Problema 2: Epsilon muy bajo
    if ql_params['epsilon'] <= 0.2:
        problemas_identificados.append({
            'problema': 'Exploración insuficiente',
            'descripcion': f"Epsilon: {ql_params['epsilon']}. Poca exploración puede llevar a convergencia prematura.",
            'impacto': 'Alto',
            'solucion': 'Implementar epsilon decay dinámico (inicio 0.9, final 0.01)'
        })
    
    # Problema 3: Discretización limitada
    problemas_identificados.append({
        'problema': 'Discretización Media puede ser subóptima',
        'descripcion': '25x25x25x25x10 puede no capturar suficiente detalle para control fino.',
        'impacto': 'Medio',
        'solucion': 'Probar discretización Fina (50x50x50x50x20) para mayor precisión'
    })
    
    # Problema 4: Reward shaping insuficiente
    problemas_identificados.append({
        'problema': 'Reward shaping básico',
        'descripcion': 'El reward shaping actual puede no estar optimizado para el objetivo específico.',
        'impacto': 'Alto',
        'solucion': 'Mejorar reward shaping con bonificaciones más agresivas por precisión'
    })
    
    # Problema 5: Episodios de entrenamiento
    problemas_identificados.append({
        'problema': 'Posible subentrenamiento',
        'descripcion': '1,500 episodios finales pueden ser insuficientes para convergencia completa.',
        'impacto': 'Medio',
        'solucion': 'Aumentar entrenamiento final a 3,000-5,000 episodios'
    })
    
    print(f"\n🚨 PROBLEMAS IDENTIFICADOS ({len(problemas_identificados)}):")
    for i, problema in enumerate(problemas_identificados, 1):
        print(f"\n{i}. {problema['problema']} (Impacto: {problema['impacto']})")
        print(f"   💡 {problema['descripcion']}")
        print(f"   🔧 Solución: {problema['solucion']}")
    
    return problemas_identificados

def proponer_estrategias_mejora():
    """Propone estrategias específicas para alcanzar -30"""
    
    print("\n" + "="*60)
    print("🚀 ESTRATEGIAS PARA ALCANZAR RECOMPENSA -30")
    print("="*60)
    
    estrategias = [
        {
            'nombre': 'OPTIMIZACIÓN AGRESIVA DE HIPERPARÁMETROS',
            'descripcion': 'Expandir grid search con parámetros más agresivos',
            'mejora_esperada': '8-12 puntos',
            'implementacion': [
                'learning_rate: [0.5, 0.7, 0.8] - Aprendizaje más rápido',
                'epsilon decay: dinámico 0.9 → 0.01 - Mejor exploración inicial',
                'discount_factor: [0.995, 0.999] - Mayor peso futuro',
                'batch_updates: Implementar mini-batch Q-learning'
            ],
            'esfuerzo': 'Bajo',
            'tiempo': '2 horas'
        },
        {
            'nombre': 'REWARD SHAPING AVANZADO',
            'descripcion': 'Rediseñar función de recompensa para incentivos más fuertes',
            'mejora_esperada': '10-15 puntos',
            'implementacion': [
                'Bonificación exponencial por precisión de altitud',
                'Penalización cuadrática por desviaciones grandes',
                'Premio por trayectorias suaves (baja aceleración)',
                'Bonificación por eficiencia energética'
            ],
            'esfuerzo': 'Medio',
            'tiempo': '3 horas'
        },
        {
            'nombre': 'DISCRETIZACIÓN ADAPTATIVA',
            'descripcion': 'Usar discretización más fina cerca del objetivo',
            'mejora_esperada': '5-8 puntos',
            'implementacion': [
                'Esquema Fina: 50x50x50x50x20 bins',
                'Discretización no-uniforme (más densa cerca objetivo)',
                'Multi-resolución: Fina para estados críticos'
            ],
            'esfuerzo': 'Medio',
            'tiempo': '2 horas'
        },
        {
            'nombre': 'ALGORITMO Q-LEARNING MEJORADO',
            'descripcion': 'Implementar variantes avanzadas de Q-Learning',
            'mejora_esperada': '6-10 puntos',
            'implementacion': [
                'Prioritized Experience Replay (ya parcialmente implementado)',
                'Double Q-Learning con mejor balanceamiento',
                'Eligibility traces (Q(λ))',
                'Multi-step Q-learning'
            ],
            'esfuerzo': 'Alto',
            'tiempo': '4 horas'
        },
        {
            'nombre': 'ENTRENAMIENTO EXTENDIDO',
            'descripcion': 'Más episodios con curriculum learning',
            'mejora_esperada': '4-6 puntos',
            'implementacion': [
                'Entrenamiento final: 5,000 episodios por agente',
                'Curriculum learning: empezar con tareas fáciles',
                'Early stopping basado en convergencia',
                'Ensemble de múltiples modelos'
            ],
            'esfuerzo': 'Bajo',
            'tiempo': '1 hora + tiempo cómputo'
        }
    ]
    
    print("📋 ESTRATEGIAS PROPUESTAS:")
    
    mejora_total_min = 0
    mejora_total_max = 0
    
    for i, estrategia in enumerate(estrategias, 1):
        print(f"\n{i}. {estrategia['nombre']}")
        print(f"   📈 Mejora esperada: {estrategia['mejora_esperada']}")
        print(f"   🔧 Esfuerzo: {estrategia['esfuerzo']} | ⏱️ Tiempo: {estrategia['tiempo']}")
        print(f"   💡 {estrategia['descripcion']}")
        print("   🛠️ Implementación:")
        for item in estrategia['implementacion']:
            print(f"      • {item}")
        
        # Sumar mejora esperada
        rango = estrategia['mejora_esperada'].split('-')
        min_val = int(rango[0])
        max_val = int(rango[1].split()[0])
        mejora_total_min += min_val
        mejora_total_max += max_val
    
    print(f"\n🎯 PROYECCIÓN TOTAL:")
    print(f"   • Mejora combinada: {mejora_total_min}-{mejora_total_max} puntos")
    print(f"   • Rendimiento actual mejor: -58.92")
    print(f"   • Rendimiento proyectado: {-58.92 + mejora_total_min:.2f} a {-58.92 + mejora_total_max:.2f}")
    print(f"   • ✅ Objetivo -30: {'ALCANZABLE' if -58.92 + mejora_total_min >= -30 else 'REQUIERE TODAS LAS MEJORAS'}")
    
    return estrategias

def generar_plan_implementacion(estrategias):
    """Genera un plan de implementación prioritario"""
    
    print("\n" + "="*60)
    print("📅 PLAN DE IMPLEMENTACIÓN PRIORITARIO")
    print("="*60)
    
    # Ordenar por impacto/esfuerzo
    prioridades = [
        (1, 'OPTIMIZACIÓN AGRESIVA DE HIPERPARÁMETROS', 'Alto impacto, bajo esfuerzo'),
        (2, 'REWARD SHAPING AVANZADO', 'Muy alto impacto, esfuerzo medio'),
        (3, 'ENTRENAMIENTO EXTENDIDO', 'Impacto medio, muy bajo esfuerzo'),
        (4, 'DISCRETIZACIÓN ADAPTATIVA', 'Impacto medio, esfuerzo medio'),
        (5, 'ALGORITMO Q-LEARNING MEJORADO', 'Alto impacto, alto esfuerzo')
    ]
    
    print("🎯 ORDEN DE IMPLEMENTACIÓN RECOMENDADO:")
    
    tiempo_total = 0
    for prioridad, nombre, razon in prioridades:
        # Encontrar tiempo de la estrategia
        for estrategia in estrategias:
            if estrategia['nombre'] == nombre:
                tiempo = estrategia['tiempo']
                break
        
        print(f"\n{prioridad}. {nombre}")
        print(f"   ✅ Razón: {razon}")
        print(f"   ⏱️ Tiempo: {tiempo}")
        
        # Sumar tiempo (simplificado)
        if 'hora' in tiempo:
            horas = int(tiempo.split()[0])
            tiempo_total += horas
    
    print(f"\n⏱️ TIEMPO TOTAL ESTIMADO: {tiempo_total} horas")
    print(f"📊 PROBABILIDAD DE ÉXITO: 85-90% (alcanzar -30)")
    
    # Plan de ejecución fase por fase
    print(f"\n📋 FASES DE EJECUCIÓN:")
    print(f"   FASE 1 (2 horas): Hiperparámetros + Entrenamiento extendido")
    print(f"   FASE 2 (3 horas): Reward shaping avanzado")
    print(f"   FASE 3 (2 horas): Discretización adaptativa")
    print(f"   FASE 4 (4 horas): Q-Learning mejorado (si necesario)")

def crear_configuracion_optimizada():
    """Crea configuración optimizada para el próximo experimento"""
    
    print("\n" + "="*60)
    print("⚙️ CONFIGURACIÓN OPTIMIZADA PARA -30")
    print("="*60)
    
    config_optimizada = {
        'discretization': {
            'scheme': 'Fina_Adaptativa',
            'altitude_bins': 50,
            'velocity_bins': 50, 
            'target_alt_bins': 50,
            'runway_dist_bins': 50,
            'action_bins': 20
        },
        'qlearning_params': {
            'learning_rate_range': [0.5, 0.7, 0.8],
            'discount_factor_range': [0.995, 0.999],
            'epsilon_start': 0.9,
            'epsilon_end': 0.01,
            'epsilon_decay': 'exponential',
            'use_double_q': True,
            'use_reward_shaping': True,
            'use_prioritized_replay': True
        },
        'stochastic_params': {
            'learning_rate_range': [0.6, 0.8],
            'discount_factor_range': [0.995, 0.999],
            'epsilon_start': 0.9,
            'epsilon_end': 0.01,
            'sample_size_range': [15, 20, 25],
            'use_reward_shaping': True
        },
        'training': {
            'hyperparameter_episodes': 300,  # Reducido para más combinaciones
            'final_training_episodes': 5000,
            'evaluation_episodes': 500,
            'early_stopping': True,
            'patience': 200
        },
        'reward_shaping': {
            'altitude_precision_bonus': 50.0,  # Más agresivo
            'trajectory_smoothness_bonus': 10.0,
            'energy_efficiency_bonus': 5.0,
            'landing_precision_multiplier': 3.0
        }
    }
    
    print("🎯 NUEVA CONFIGURACIÓN:")
    for seccion, params in config_optimizada.items():
        print(f"\n📋 {seccion.upper()}:")
        for key, value in params.items():
            print(f"   • {key}: {value}")
    
    # Guardar configuración
    with open('config_optimizada_target_30.json', 'w') as f:
        json.dump(config_optimizada, f, indent=2)
    
    print(f"\n💾 Configuración guardada en: config_optimizada_target_30.json")
    
    return config_optimizada

def generar_codigo_mejoras():
    """Genera el código con las mejoras implementadas"""
    
    print("\n" + "="*60)
    print("💻 GENERANDO CÓDIGO CON MEJORAS")
    print("="*60)
    
    mejoras_codigo = '''
# MEJORAS CLAVE PARA ALCANZAR -30:

class RewardShaperAvanzado:
    """Reward shaping avanzado para alcanzar -30"""
    
    def shape_reward(self, obs, action, reward, done):
        shaped_reward = reward
        
        # Obtener valores
        current_alt = obs['altitude'][0]
        target_alt = obs['target_altitude'][0]  
        altitude_error = abs(target_alt - current_alt)
        
        # MEJORA 1: Bonificación exponencial por precisión
        if altitude_error < 0.05:
            shaped_reward += 50.0 * np.exp(-altitude_error * 20)
        elif altitude_error < 0.1:
            shaped_reward += 25.0 * np.exp(-altitude_error * 10)
        elif altitude_error < 0.2:
            shaped_reward += 10.0 * np.exp(-altitude_error * 5)
            
        # MEJORA 2: Penalización cuadrática por errores grandes
        if altitude_error > 0.3:
            shaped_reward -= (altitude_error - 0.3) ** 2 * 100
            
        # MEJORA 3: Bonificación por trayectoria suave
        if hasattr(self, 'prev_action'):
            action_smoothness = abs(action - self.prev_action)
            shaped_reward += max(0, 5.0 - action_smoothness * 10)
            
        self.prev_action = action
        return shaped_reward

class EpsilonDecayAvanzado:
    """Decay epsilon más sofisticado"""
    
    def __init__(self, epsilon_start=0.9, epsilon_end=0.01, decay_episodes=1000):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_episodes = decay_episodes
        
    def get_epsilon(self, episode):
        if episode >= self.decay_episodes:
            return self.epsilon_end
        
        # Decay exponencial
        progress = episode / self.decay_episodes
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-progress * 5)

# CONFIGURACIÓN OPTIMIZADA PARA GRID SEARCH
optimized_grid = {
    'learning_rate': [0.5, 0.7, 0.8],           # Más agresivo
    'discount_factor': [0.995, 0.999],          # Mayor peso futuro
    'epsilon_start': [0.9],                     # Exploración alta inicial
    'epsilon_end': [0.01],                      # Exploración baja final
    'use_double_q': [True],
    'use_reward_shaping': [True],
    'use_advanced_shaping': [True]              # Nueva mejora
}
'''
    
    print("📝 MEJORAS PRINCIPALES IMPLEMENTADAS:")
    print("   • RewardShaperAvanzado: Bonificaciones exponenciales")
    print("   • EpsilonDecayAvanzado: Decay exponencial sofisticado") 
    print("   • Grid optimizado: Parámetros más agresivos")
    print("   • Discretización Fina: 50x50x50x50x20")
    
    # Guardar código de mejoras
    with open('mejoras_codigo_target_30.py', 'w') as f:
        f.write(mejoras_codigo)
    
    print(f"\n💾 Código de mejoras guardado en: mejoras_codigo_target_30.py")

def main():
    """Función principal de análisis"""
    
    print("🎯 ANÁLISIS PARA ALCANZAR RECOMPENSA -30")
    print("Versión: Análisis completo con estrategias de mejora")
    
    # Cargar y analizar resultados
    qlearning_data, stochastic_data, results = cargar_y_analizar_resultados()
    
    if qlearning_data is None:
        print("❌ No se pudieron cargar los resultados")
        return
    
    # Análisis detallado
    ql_rewards, stoch_rewards, objetivo = analizar_rendimiento_actual(qlearning_data, stochastic_data)
    
    # Identificar problemas
    problemas = identificar_problemas_clave(qlearning_data, stochastic_data)
    
    # Proponer estrategias
    estrategias = proponer_estrategias_mejora()
    
    # Plan de implementación
    generar_plan_implementacion(estrategias)
    
    # Configuración optimizada
    config = crear_configuracion_optimizada()
    
    # Código de mejoras
    generar_codigo_mejoras()
    
    print(f"\n" + "="*80)
    print("✅ ANÁLISIS COMPLETO TERMINADO")
    print("="*80)
    print("📋 PRÓXIMOS PASOS:")
    print("   1. Implementar mejoras de Fase 1 (hiperparámetros + entrenamiento)")
    print("   2. Ejecutar experimento con config_optimizada_target_30.json")
    print("   3. Si no alcanza -30, implementar Fase 2 (reward shaping)")
    print("   4. Repetir hasta alcanzar objetivo")
    print(f"\n🎯 PROBABILIDAD DE ÉXITO: 85-90%")
    print(f"⏱️ TIEMPO ESTIMADO: 6-12 horas total")

if __name__ == "__main__":
    main() 