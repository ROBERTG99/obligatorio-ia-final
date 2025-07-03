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
                'discount_factor: [0.995, 0.999] - Mayor peso futuro'
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
                'Premio por trayectorias suaves'
            ],
            'esfuerzo': 'Medio',
            'tiempo': '3 horas'
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
    
    return estrategias

def main():
    """Función principal de análisis"""
    
    print("🎯 ANÁLISIS PARA ALCANZAR RECOMPENSA -30")
    
    # Cargar y analizar resultados
    qlearning_data, stochastic_data, results = cargar_y_analizar_resultados()
    
    if qlearning_data is None:
        print("❌ No se pudieron cargar los resultados")
        return
    
    # Análisis detallado
    analizar_rendimiento_actual(qlearning_data, stochastic_data)
    
    # Identificar problemas
    identificar_problemas_clave(qlearning_data, stochastic_data)
    
    # Proponer estrategias
    proponer_estrategias_mejora()
    
    print(f"\n" + "="*80)
    print("✅ ANÁLISIS COMPLETO TERMINADO")
    print("="*80)

if __name__ == "__main__":
    main() 