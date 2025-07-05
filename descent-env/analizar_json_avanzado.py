#!/usr/bin/env python3
"""
🎯 ANALIZADOR JSON AVANZADO PARA PROYECTO FLAN
Análisis específico del JSON de resultados con mejoras automáticas inteligentes
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

class AnalizadorJSONAvanzado:
    """Analizador específico para JSON de resultados FLAN"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.resultados = self.cargar_resultados()
        self.objetivo_target = -25
        self.umbral_critico = -35
        self.umbral_excelente = -20
        
    def cargar_resultados(self) -> Dict:
        """Carga los resultados del JSON con manejo de errores"""
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
                print(f"✅ JSON cargado exitosamente: {self.json_path}")
                return data
        except FileNotFoundError:
            print(f"⚠️ Archivo {self.json_path} no encontrado")
            return {}
        except json.JSONDecodeError as e:
            print(f"⚠️ Error al leer JSON (puede estar incompleto): {e}")
            return {}
    
    def analizar_estructura_json(self) -> Dict:
        """Analiza la estructura del JSON para entender qué datos hay"""
        estructura = {
            'esquemas_disponibles': [],
            'agentes_disponibles': [],
            'metricas_disponibles': [],
            'info_experimento': {},
            'completitud': {}
        }
        
        if not self.resultados:
            return estructura
        
        # Analizar esquemas y agentes
        for key, value in self.resultados.items():
            if key == 'experiment_info':
                estructura['info_experimento'] = value
                continue
            
            estructura['esquemas_disponibles'].append(key)
            
            if isinstance(value, dict):
                for agent_type, agent_data in value.items():
                    if agent_type not in estructura['agentes_disponibles']:
                        estructura['agentes_disponibles'].append(agent_type)
                    
                    # Analizar completitud
                    completitud_key = f"{key}_{agent_type}"
                    estructura['completitud'][completitud_key] = {
                        'tiene_evaluation': 'evaluation' in agent_data,
                        'tiene_training': 'training_rewards' in agent_data,
                        'tiene_params': 'best_params' in agent_data,
                        'tiene_score': 'best_score' in agent_data
                    }
                    
                    if 'evaluation' in agent_data:
                        metricas = list(agent_data['evaluation'].keys())
                        for metrica in metricas:
                            if metrica not in estructura['metricas_disponibles']:
                                estructura['metricas_disponibles'].append(metrica)
        
        return estructura
    
    def diagnosticar_problemas(self) -> Dict:
        """Diagnóstico inteligente de problemas específicos"""
        diagnostico = {
            'problemas_criticos': [],
            'problemas_moderados': [],
            'oportunidades_mejora': [],
            'analisis_detallado': {}
        }
        
        if not self.resultados:
            diagnostico['problemas_criticos'].append("JSON vacío o no disponible")
            return diagnostico
        
        # Analizar cada agente
        for scheme_name, scheme_data in self.resultados.items():
            if scheme_name == 'experiment_info':
                continue
                
            for agent_type, agent_data in scheme_data.items():
                key = f"{scheme_name}_{agent_type}"
                
                if 'evaluation' not in agent_data:
                    diagnostico['problemas_criticos'].append(
                        f"❌ {key}: No tiene datos de evaluación"
                    )
                    continue
                
                # Análisis detallado
                eval_data = agent_data['evaluation']
                rewards = eval_data.get('total_rewards', [])
                steps = eval_data.get('survival_times', [])
                errors = eval_data.get('altitude_errors', [])
                
                if not rewards:
                    diagnostico['problemas_criticos'].append(
                        f"❌ {key}: Sin datos de recompensas"
                    )
                    continue
                
                # Calcular métricas
                media_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                min_reward = np.min(rewards)
                max_reward = np.max(rewards)
                
                media_steps = np.mean(steps) if steps else 0
                media_error = np.mean(errors) if errors else float('inf')
                
                diagnostico['analisis_detallado'][key] = {
                    'media_reward': media_reward,
                    'std_reward': std_reward,
                    'min_reward': min_reward,
                    'max_reward': max_reward,
                    'media_steps': media_steps,
                    'media_error': media_error,
                    'num_evaluaciones': len(rewards),
                    'distancia_objetivo': abs(media_reward - self.objetivo_target)
                }
                
                # Diagnóstico específico
                if media_reward < -70:
                    diagnostico['problemas_criticos'].append(
                        f"🔴 {key}: Recompensa CRÍTICA ({media_reward:.1f}) - "
                        f"Necesita +{self.objetivo_target - media_reward:.1f} puntos"
                    )
                elif media_reward < -50:
                    diagnostico['problemas_moderados'].append(
                        f"🟡 {key}: Recompensa baja ({media_reward:.1f}) - "
                        f"Necesita +{self.objetivo_target - media_reward:.1f} puntos"
                    )
                elif media_reward < -25:
                    diagnostico['oportunidades_mejora'].append(
                        f"🟢 {key}: Cerca del objetivo ({media_reward:.1f}) - "
                        f"Solo +{self.objetivo_target - media_reward:.1f} puntos más"
                    )
                
                # Análisis de variabilidad
                if std_reward > 20:
                    diagnostico['problemas_criticos'].append(
                        f"🔴 {key}: Variabilidad EXTREMA (±{std_reward:.1f}) - Muy inconsistente"
                    )
                elif std_reward > 15:
                    diagnostico['problemas_moderados'].append(
                        f"🟡 {key}: Variabilidad alta (±{std_reward:.1f}) - Inconsistente"
                    )
                
                # Análisis de supervivencia
                if media_steps < 50:
                    diagnostico['problemas_criticos'].append(
                        f"🔴 {key}: Supervivencia CRÍTICA ({media_steps:.1f} pasos) - Muere muy rápido"
                    )
                elif media_steps < 150:
                    diagnostico['problemas_moderados'].append(
                        f"🟡 {key}: Supervivencia baja ({media_steps:.1f} pasos) - Mejorable"
                    )
                
                # Análisis de precisión
                if media_error > 0.2:
                    diagnostico['problemas_criticos'].append(
                        f"🔴 {key}: Error altitud CRÍTICO ({media_error:.3f}) - Muy impreciso"
                    )
                elif media_error > 0.1:
                    diagnostico['problemas_moderados'].append(
                        f"🟡 {key}: Error altitud alto ({media_error:.3f}) - Impreciso"
                    )
        
        return diagnostico
    
    def generar_mejoras_automaticas(self, diagnostico: Dict) -> Dict:
        """Genera mejoras automáticas basadas en el diagnóstico"""
        mejoras = {
            'reward_shaping': [],
            'hiperparametros': [],
            'arquitectura': [],
            'entrenamiento': [],
            'codigo_implementacion': ""
        }
        
        # Análisis de problemas predominantes
        problemas_supervivencia = sum(1 for p in diagnostico['problemas_criticos'] 
                                    if 'Supervivencia' in p)
        problemas_precision = sum(1 for p in diagnostico['problemas_criticos'] 
                                if 'altitud' in p)
        problemas_variabilidad = sum(1 for p in diagnostico['problemas_criticos'] 
                                   if 'Variabilidad' in p)
        
        # Mejoras de reward shaping
        if problemas_supervivencia > 0:
            mejoras['reward_shaping'].extend([
                "🔴 SUPERVIVENCIA CRÍTICA - Reward Shaping Ultra Agresivo:",
                "  • Bonificación supervivencia: +50 por cada paso sobrevivido",
                "  • Bonificación exponencial: +step^1.2 por longevidad",
                "  • Penalización muerte temprana: -1000 si steps < 100",
                "  • Bonus hitos: +500 cada 50 pasos, +2000 cada 200 pasos"
            ])
        
        if problemas_precision > 0:
            mejoras['reward_shaping'].extend([
                "🔴 PRECISIÓN CRÍTICA - Reward Shaping para Altitud:",
                "  • Bonificación exponencial: +5000 * (0.1 - error)^2",
                "  • Penalización cúbica: -error^3 * 10000",
                "  • Bonus mejora progresiva: +100 si error < prev_error"
            ])
        
        if problemas_variabilidad > 0:
            mejoras['reward_shaping'].extend([
                "🔴 VARIABILIDAD CRÍTICA - Reward Shaping para Consistencia:",
                "  • Smoothing reward: shaped = 0.6*shaped + 0.4*prev_shaped",
                "  • Penalización acciones erráticas: -200*abs(action-prev_action)",
                "  • Bonus consistencia: +100 si acción similar a promedio"
            ])
        
        # Mejoras de hiperparámetros
        recompensas_muy_bajas = any(d['media_reward'] < -60 
                                   for d in diagnostico['analisis_detallado'].values())
        
        if recompensas_muy_bajas:
            mejoras['hiperparametros'].extend([
                "🔴 HIPERPARÁMETROS BREAKTHROUGH para Recompensas < -60:",
                "  • Learning Rate ULTRA agresivo: 0.95",
                "  • Epsilon inicial MÁXIMO: 0.98 (exploración extrema)",
                "  • Epsilon decay ULTRA lento: 0.999995",
                "  • Discount factor MÁXIMO: 0.99999",
                "  • Episodios aumentados: +50% más entrenamiento"
            ])
        
        # Mejoras de arquitectura
        if problemas_supervivencia > 1:
            mejoras['arquitectura'].extend([
                "🔴 ARQUITECTURA BREAKTHROUGH para Supervivencia:",
                "  • Límite pasos: 1000 → 5000 (permitir más exploración)",
                "  • Discretización más fina: bins +50%",
                "  • Curriculum learning: entrenar primero supervivencia, luego precisión",
                "  • Pre-entrenamiento con reward solo de supervivencia"
            ])
        
        # Mejoras de entrenamiento
        mejoras['entrenamiento'].extend([
            "🎯 ESTRATEGIA DE ENTRENAMIENTO OPTIMIZADA:",
            "  • Fase 1: Solo supervivencia (50k episodios)",
            "  • Fase 2: Supervivencia + precisión básica (100k episodios)", 
            "  • Fase 3: Precisión fina (50k episodios)",
            "  • Early stopping adaptativo por fase"
        ])
        
        # Generar código de implementación
        mejoras['codigo_implementacion'] = self.generar_codigo_mejoras(diagnostico)
        
        return mejoras
    
    def generar_codigo_mejoras(self, diagnostico: Dict) -> str:
        """Genera código Python implementando las mejoras"""
        
        # Encontrar el peor caso para calibrar mejoras
        peor_reward = -100
        if diagnostico['analisis_detallado']:
            peor_reward = min(d['media_reward'] for d in diagnostico['analisis_detallado'].values())
        
        codigo = f'''
# 🚀 MEJORAS AUTOMÁTICAS GENERADAS - OBJETIVO: {self.objetivo_target}
# Peor caso detectado: {peor_reward:.1f} (mejora necesaria: +{self.objetivo_target - peor_reward:.1f})

class RewardShaperAutomatico:
    """Reward shaper generado automáticamente basado en análisis JSON"""
    
    def __init__(self):
        self.step_count = 0
        self.prev_reward = 0
        self.prev_action = 0
        self.best_error = float('inf')
        self.consecutive_improvements = 0
        
    def shape_reward(self, obs, action, reward, done):
        shaped = reward
        self.step_count += 1
        
        # Métricas actuales
        altitude_error = abs(obs.get('target_altitude', [0])[0] - obs.get('altitude', [0])[0])
        
        # SUPERVIVENCIA MASIVA (detectado problema crítico)
        if not done:
            # Bonificación base por supervivencia
            shaped += 50.0
            
            # Bonificación exponencial por longevidad
            if self.step_count > 100:
                shaped += (self.step_count - 100) ** 1.2 * 0.1
            
            # Bonus por hitos de supervivencia
            if self.step_count % 50 == 0:
                shaped += 500.0
            if self.step_count % 200 == 0:
                shaped += 2000.0
        else:
            # Penalización por muerte temprana
            if self.step_count < 100:
                shaped -= 1000.0
        
        # PRECISIÓN ULTRA AGRESIVA (detectado problema crítico)
        if altitude_error < 0.01:
            shaped += 10000.0  # JACKPOT
        elif altitude_error < 0.05:
            shaped += 5000.0
        elif altitude_error < 0.1:
            shaped += 2000.0
        elif altitude_error < 0.15:
            shaped += 1000.0
        
        # Penalización cúbica por errores grandes
        if altitude_error > 0.1:
            shaped -= (altitude_error ** 3) * 10000
        
        # Bonus por mejora progresiva
        if altitude_error < self.best_error:
            self.best_error = altitude_error
            self.consecutive_improvements += 1
            shaped += 100.0 + (self.consecutive_improvements * 50)
        else:
            self.consecutive_improvements = 0
        
        # CONSISTENCIA (detectado problema variabilidad)
        action_consistency = abs(action - self.prev_action)
        if action_consistency < 0.1:
            shaped += 100.0
        else:
            shaped -= action_consistency * 200
        
        # Smoothing para reducir variabilidad
        shaped = 0.6 * shaped + 0.4 * self.prev_reward
        
        # Actualizar estado
        self.prev_reward = shaped
        self.prev_action = action
        
        return shaped
    
    def reset(self):
        self.step_count = 0
        self.prev_reward = 0
        self.prev_action = 0
        self.best_error = float('inf')
        self.consecutive_improvements = 0

# HIPERPARÁMETROS AUTOMÁTICOS OPTIMIZADOS
CONFIG_AUTOMATICO = {{
    'learning_rate': 0.95,           # Ultra agresivo para convergencia rápida
    'epsilon': 0.98,                 # Exploración máxima
    'epsilon_decay': 0.999995,       # Decay ultra lento
    'epsilon_min': 0.15,             # Mantener exploración mínima
    'discount_factor': 0.99999,      # Peso máximo al futuro
    'episodes_fase1': 50000,         # Fase supervivencia
    'episodes_fase2': 100000,        # Fase supervivencia + precisión
    'episodes_fase3': 50000,         # Fase precisión fina
    'step_limit': 5000,              # Límite muy alto para exploración
    'early_stopping_threshold': {self.objetivo_target + 10},  # Más permisivo inicialmente
}}

# FUNCIÓN DE ENTRENAMIENTO AUTOMÁTICO
def entrenar_con_mejoras_automaticas(env, agent, discretization):
    """Entrenamiento automático con fases optimizadas"""
    
    # Fase 1: Solo supervivencia
    print("🎯 FASE 1: Entrenamiento de supervivencia")
    agent.reward_shaper = RewardShaperSupervivencia()  # Solo supervivencia
    trainer = QLearningTrainer(env, agent, discretization)
    trainer.train(episodes=50000)
    
    # Fase 2: Supervivencia + precisión básica
    print("🎯 FASE 2: Supervivencia + precisión básica")
    agent.reward_shaper = RewardShaperAutomatico()
    trainer.train(episodes=100000)
    
    # Fase 3: Precisión fina
    print("🎯 FASE 3: Precisión fina")
    agent.reward_shaper = RewardShaperPrecisionFina()
    trainer.train(episodes=50000)
    
    return agent

class RewardShaperSupervivencia:
    """Reward shaper solo para supervivencia (Fase 1)"""
    def __init__(self):
        self.step_count = 0
    
    def shape_reward(self, obs, action, reward, done):
        self.step_count += 1
        if not done:
            return reward + 100.0 + (self.step_count * 0.5)
        return reward - 500.0 if self.step_count < 200 else reward
    
    def reset(self):
        self.step_count = 0

class RewardShaperPrecisionFina:
    """Reward shaper para precisión fina (Fase 3)"""
    def shape_reward(self, obs, action, reward, done):
        altitude_error = abs(obs.get('target_altitude', [0])[0] - obs.get('altitude', [0])[0])
        if altitude_error < 0.05:
            return reward + 5000.0
        elif altitude_error < 0.1:
            return reward + 1000.0
        else:
            return reward - (altitude_error ** 2) * 500
    
    def reset(self):
        pass
'''
        
        return codigo
    
    def crear_visualizacion_diagnostico(self):
        """Crea visualización del diagnóstico"""
        diagnostico = self.diagnosticar_problemas()
        
        if not diagnostico['analisis_detallado']:
            print("❌ No hay datos para visualizar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🎯 DIAGNÓSTICO AUTOMÁTICO JSON FLAN', fontsize=16, fontweight='bold')
        
        # Extraer datos
        agents = list(diagnostico['analisis_detallado'].keys())
        rewards = [diagnostico['analisis_detallado'][a]['media_reward'] for a in agents]
        errors = [diagnostico['analisis_detallado'][a]['media_error'] for a in agents]
        steps = [diagnostico['analisis_detallado'][a]['media_steps'] for a in agents]
        std_rewards = [diagnostico['analisis_detallado'][a]['std_reward'] for a in agents]
        
        # Gráfico 1: Distancia al objetivo
        distancias = [abs(r - self.objetivo_target) for r in rewards]
        colors = ['red' if d > 40 else 'orange' if d > 15 else 'green' for d in distancias]
        
        bars = axes[0,0].bar(range(len(agents)), distancias, color=colors, alpha=0.7)
        axes[0,0].set_title('Distancia al Objetivo (-25)')
        axes[0,0].set_ylabel('Puntos de Distancia')
        axes[0,0].set_xticks(range(len(agents)))
        axes[0,0].set_xticklabels(agents, rotation=45)
        
        # Añadir valores
        for i, (bar, dist) in enumerate(zip(bars, distancias)):
            axes[0,0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                          f'{dist:.1f}', ha='center', va='bottom')
        
        # Gráfico 2: Recompensas vs Supervivencia
        scatter = axes[0,1].scatter(steps, rewards, c=errors, s=100, alpha=0.7, cmap='viridis')
        axes[0,1].axhline(y=self.objetivo_target, color='green', linestyle='--', alpha=0.7, label=f'Objetivo {self.objetivo_target}')
        axes[0,1].axhline(y=self.umbral_critico, color='red', linestyle='--', alpha=0.7, label=f'Crítico {self.umbral_critico}')
        axes[0,1].set_xlabel('Pasos de Supervivencia')
        axes[0,1].set_ylabel('Recompensa Media')
        axes[0,1].set_title('Supervivencia vs Recompensa (color = error)')
        axes[0,1].legend()
        plt.colorbar(scatter, ax=axes[0,1], label='Error Altitud')
        
        # Gráfico 3: Variabilidad
        axes[1,0].bar(range(len(agents)), std_rewards, alpha=0.7)
        axes[1,0].set_title('Variabilidad (Desviación Estándar)')
        axes[1,0].set_ylabel('Std Dev Recompensa')
        axes[1,0].set_xticks(range(len(agents)))
        axes[1,0].set_xticklabels(agents, rotation=45)
        axes[1,0].axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Crítico')
        axes[1,0].axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Moderado')
        axes[1,0].legend()
        
        # Gráfico 4: Resumen problemas
        problemas_counts = {
            'Críticos': len(diagnostico['problemas_criticos']),
            'Moderados': len(diagnostico['problemas_moderados']),
            'Oportunidades': len(diagnostico['oportunidades_mejora'])
        }
        
        colors_pie = ['red', 'orange', 'green']
        axes[1,1].pie(problemas_counts.values(), labels=problemas_counts.keys(), 
                      colors=colors_pie, autopct='%1.0f', startangle=90)
        axes[1,1].set_title('Distribución de Problemas')
        
        plt.tight_layout()
        plt.savefig('diagnostico_json_flan.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generar_reporte_completo(self):
        """Genera reporte completo con análisis y mejoras"""
        print("🎯 ANALIZADOR JSON AVANZADO - REPORTE COMPLETO")
        print("="*80)
        
        # Estructura del JSON
        estructura = self.analizar_estructura_json()
        print(f"📊 ESTRUCTURA DEL JSON:")
        print(f"  • Esquemas: {len(estructura['esquemas_disponibles'])}")
        print(f"  • Agentes: {len(estructura['agentes_disponibles'])}")
        print(f"  • Métricas: {len(estructura['metricas_disponibles'])}")
        
        # Diagnóstico
        diagnostico = self.diagnosticar_problemas()
        print(f"\n�� DIAGNÓSTICO AUTOMÁTICO:")
        print(f"  • Problemas críticos: {len(diagnostico['problemas_criticos'])}")
        print(f"  • Problemas moderados: {len(diagnostico['problemas_moderados'])}")
        print(f"  • Oportunidades: {len(diagnostico['oportunidades_mejora'])}")
        
        # Mostrar problemas críticos
        if diagnostico['problemas_criticos']:
            print(f"\n🔴 PROBLEMAS CRÍTICOS DETECTADOS:")
            for problema in diagnostico['problemas_criticos']:
                print(f"  {problema}")
        
        # Mejoras automáticas
        mejoras = self.generar_mejoras_automaticas(diagnostico)
        print(f"\n�� MEJORAS AUTOMÁTICAS GENERADAS:")
        
        for categoria, items in mejoras.items():
            if categoria == 'codigo_implementacion':
                continue
            if items:
                print(f"\n📈 {categoria.upper()}:")
                for item in items:
                    print(f"  {item}")
        
        # Guardar código
        with open('mejoras_automaticas.py', 'w') as f:
            f.write(mejoras['codigo_implementacion'])
        
        print(f"\n✅ Código de mejoras guardado en: mejoras_automaticas.py")
        
        # Crear visualización
        self.crear_visualizacion_diagnostico()

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizador JSON Avanzado FLAN')
    parser.add_argument('--json', '-j', default='flan_results_196k_ultra.json',
                       help='Archivo JSON de resultados')
    
    args = parser.parse_args()
    
    # Buscar archivos JSON disponibles
    json_files = list(Path('.').glob('flan_results*.json'))
    if not Path(args.json).exists() and json_files:
        print(f"⚠️ {args.json} no encontrado. Archivos disponibles:")
        for f in json_files:
            print(f"  • {f}")
        
        # Usar el más reciente
        args.json = str(sorted(json_files, key=lambda x: x.stat().st_mtime)[-1])
        print(f"📊 Usando archivo más reciente: {args.json}")
    
    # Ejecutar análisis
    analizador = AnalizadorJSONAvanzado(args.json)
    analizador.generar_reporte_completo()

if __name__ == "__main__":
    main()
