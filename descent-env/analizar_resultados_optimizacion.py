#!/usr/bin/env python3
"""
🎯 ANALIZADOR AVANZADO DE RESULTADOS FLAN
Analiza el JSON de resultados y sugiere mejoras específicas para alcanzar -25
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class AnalizadorResultadosFLAN:
    """Analizador avanzado para optimizar hacia recompensa -25"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.resultados = self.cargar_resultados()
        self.objetivo_target = -25
        self.umbral_critico = -35  # Umbral mínimo aceptable
        
    def cargar_resultados(self) -> Dict:
        """Carga los resultados del JSON"""
        try:
            with open(self.json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Archivo {self.json_path} no encontrado. Creando análisis vacío.")
            return {}
        except json.JSONDecodeError:
            print(f"⚠️ Error al leer {self.json_path}. Puede estar incompleto.")
            return {}
    
    def analizar_rendimiento_actual(self) -> Dict[str, Any]:
        """Analiza el rendimiento actual y identifica problemas críticos"""
        analisis = {
            'problemas_criticos': [],
            'problemas_moderados': [],
            'puntos_fuertes': [],
            'metricas_clave': {},
            'distancia_objetivo': {}
        }
        
        if not self.resultados:
            analisis['problemas_criticos'].append("No hay resultados disponibles para analizar")
            return analisis
        
        # Analizar cada esquema y agente
        for scheme_name, scheme_data in self.resultados.items():
            if scheme_name == 'experiment_info':
                continue
                
            for agent_type, agent_data in scheme_data.items():
                key = f"{scheme_name}_{agent_type}"
                
                # Obtener métricas
                if 'evaluation' in agent_data:
                    rewards = agent_data['evaluation']['total_rewards']
                    steps = agent_data['evaluation']['survival_times']
                    errors = agent_data['evaluation']['altitude_errors']
                    
                    media_reward = np.mean(rewards)
                    std_reward = np.std(rewards)
                    media_steps = np.mean(steps)
                    media_error = np.mean(errors)
                    
                    analisis['metricas_clave'][key] = {
                        'media_reward': media_reward,
                        'std_reward': std_reward,
                        'media_steps': media_steps,
                        'media_error': media_error,
                        'min_reward': np.min(rewards),
                        'max_reward': np.max(rewards),
                        'q25_reward': np.percentile(rewards, 25),
                        'q75_reward': np.percentile(rewards, 75)
                    }
                    
                    # Calcular distancia al objetivo
                    distancia = abs(media_reward - self.objetivo_target)
                    analisis['distancia_objetivo'][key] = {
                        'distancia_absoluta': distancia,
                        'mejora_necesaria': self.objetivo_target - media_reward,
                        'porcentaje_mejora': (distancia / abs(media_reward)) * 100 if media_reward != 0 else float('inf')
                    }
                    
                    # Identificar problemas críticos
                    if media_reward < -60:
                        analisis['problemas_criticos'].append(
                            f"{key}: Recompensa muy baja ({media_reward:.1f}), necesita +{self.objetivo_target - media_reward:.1f} puntos"
                        )
                    elif media_reward < -40:
                        analisis['problemas_moderados'].append(
                            f"{key}: Recompensa mejorable ({media_reward:.1f}), necesita +{self.objetivo_target - media_reward:.1f} puntos"
                        )
                    else:
                        analisis['puntos_fuertes'].append(
                            f"{key}: Recompensa prometedora ({media_reward:.1f})"
                        )
                    
                    if std_reward > 15:
                        analisis['problemas_criticos'].append(
                            f"{key}: Alta variabilidad (±{std_reward:.1f}), inconsistente"
                        )
                    elif std_reward > 10:
                        analisis['problemas_moderados'].append(
                            f"{key}: Variabilidad moderada (±{std_reward:.1f})"
                        )
                    
                    if media_steps < 100:
                        analisis['problemas_criticos'].append(
                            f"{key}: Supervivencia muy baja ({media_steps:.1f} pasos)"
                        )
                    elif media_steps < 200:
                        analisis['problemas_moderados'].append(
                            f"{key}: Supervivencia mejorable ({media_steps:.1f} pasos)"
                        )
                    else:
                        analisis['puntos_fuertes'].append(
                            f"{key}: Buena supervivencia ({media_steps:.1f} pasos)"
                        )
        
        return analisis
    
    def sugerir_mejoras_reward_shaping(self, analisis: Dict) -> List[str]:
        """Sugiere mejoras específicas en reward shaping"""
        mejoras = []
        
        # Analizar problemas de supervivencia
        supervivencia_baja = any('Supervivencia' in p for p in analisis['problemas_criticos'])
        if supervivencia_baja:
            mejoras.extend([
                "🔴 CRÍTICO - Reward Shaping para Supervivencia:",
                "  • Aumentar bonificación por paso: +5 → +10 por supervivencia",
                "  • Añadir bonificación exponencial: +1*step^1.1 por longevidad",
                "  • Penalización menor por muerte temprana: -50*max(0, 200-steps)",
                "  • Bonificación por hitos: +50 puntos cada 100 pasos"
            ])
        
        # Analizar problemas de precisión
        for key, metricas in analisis['metricas_clave'].items():
            if metricas['media_error'] > 0.1:
                mejoras.extend([
                    f"🔴 CRÍTICO - Reward Shaping para Precisión ({key}):",
                    "  • Bonificación exponencial por precisión: +1000*(0.1-error)^2",
                    "  • Penalización suave por error: -100*error^1.5",
                    "  • Bonificación por mejora: +20 si error < prev_error"
                ])
        
        # Analizar problemas de variabilidad
        alta_variabilidad = any('variabilidad' in p.lower() for p in analisis['problemas_criticos'])
        if alta_variabilidad:
            mejoras.extend([
                "🔴 CRÍTICO - Reward Shaping para Consistencia:",
                "  • Reward smoothing: shaped_reward = 0.7*shaped + 0.3*prev_shaped",
                "  • Penalización por acciones erráticas: -10*abs(action-prev_action)",
                "  • Bonificación por trayectoria suave: +5 si smooth_trajectory"
            ])
        
        return mejoras
    
    def sugerir_mejoras_hiperparametros(self, analisis: Dict) -> List[str]:
        """Sugiere ajustes de hiperparámetros"""
        mejoras = []
        
        # Analizar convergencia
        for key, metricas in analisis['metricas_clave'].items():
            if metricas['media_reward'] < -50:
                mejoras.extend([
                    f"🔴 CRÍTICO - Hiperparámetros para Convergencia ({key}):",
                    "  • Learning Rate más agresivo: 0.5 → 0.8",
                    "  • Epsilon inicial MUY alto: 0.9 (exploración masiva)",
                    "  • Epsilon decay ULTRA lento: 0.99995",
                    "  • Discount factor máximo: 0.9999"
                ])
            elif metricas['media_reward'] < -35:
                mejoras.extend([
                    f"🟡 MODERADO - Hiperparámetros para Mejora ({key}):",
                    "  • Learning Rate aumentado: actual → +0.2",
                    "  • Epsilon más exploratorio: +0.2 inicial",
                    "  • Decay más lento: actual*0.5"
                ])
        
        # Analizar estabilidad
        for key, metricas in analisis['metricas_clave'].items():
            if metricas['std_reward'] > 15:
                mejoras.extend([
                    f"🔴 CRÍTICO - Hiperparámetros para Estabilidad ({key}):",
                    "  • Learning Rate más conservador para convergencia final",
                    "  • Epsilon mínimo mayor: 0.05 → 0.1 (mantener exploración)",
                    "  • Usar experience replay con priorización"
                ])
        
        return mejoras
    
    def sugerir_mejoras_arquitectura(self, analisis: Dict) -> List[str]:
        """Sugiere cambios arquitecturales"""
        mejoras = []
        
        # Problemas severos requieren cambios drásticos
        problemas_severos = len(analisis['problemas_criticos']) > 2
        if problemas_severos:
            mejoras.extend([
                "🔴 CRÍTICO - Cambios Arquitecturales Drásticos:",
                "  • Aumentar discretización: 40×30×30×30×20 → 50×40×40×40×25",
                "  • Implementar curriculum learning: empezar con target -50, luego -35, luego -25",
                "  • Usar ensemble de agentes: combinar mejores políticas",
                "  • Implementar prioritized experience replay",
                "  • Añadir meta-learning: aprender a aprender más rápido"
            ])
        
        # Problemas de supervivencia
        supervivencia_critica = any('muy baja' in p for p in analisis['problemas_criticos'])
        if supervivencia_critica:
            mejoras.extend([
                "🔴 CRÍTICO - Arquitectura para Supervivencia:",
                "  • Aumentar límite de pasos: 1000 → 2000",
                "  • Implementar early stopping más agresivo por supervivencia",
                "  • Usar reward shaping adaptativo basado en performance",
                "  • Pre-entrenar con target de supervivencia antes que precisión"
            ])
        
        return mejoras
    
    def analizar_convergencia(self) -> Dict[str, Any]:
        """Analiza la convergencia del entrenamiento"""
        convergencia = {}
        
        for scheme_name, scheme_data in self.resultados.items():
            if scheme_name == 'experiment_info':
                continue
                
            for agent_type, agent_data in scheme_data.items():
                if 'training_rewards' not in agent_data:
                    continue
                    
                rewards = agent_data['training_rewards']
                key = f"{scheme_name}_{agent_type}"
                
                # Analizar tendencia
                if len(rewards) > 100:
                    inicio = np.mean(rewards[:100])
                    final = np.mean(rewards[-100:])
                    mejora = final - inicio
                    
                    # Analizar ventanas de convergencia
                    ventanas = [rewards[i:i+1000] for i in range(0, len(rewards)-1000, 1000)]
                    medias_ventanas = [np.mean(v) for v in ventanas]
                    
                    convergencia[key] = {
                        'mejora_total': mejora,
                        'mejora_porcentual': (mejora / abs(inicio)) * 100 if inicio != 0 else 0,
                        'convergio': mejora > 0,
                        'plateando': len(medias_ventanas) > 2 and np.std(medias_ventanas[-3:]) < 2,
                        'medias_ventanas': medias_ventanas,
                        'reward_inicial': inicio,
                        'reward_final': final
                    }
        
        return convergencia
    
    def generar_recomendaciones_prioritarias(self) -> List[str]:
        """Genera recomendaciones priorizadas por impacto"""
        analisis = self.analizar_rendimiento_actual()
        convergencia = self.analizar_convergencia()
        
        recomendaciones = ["🎯 RECOMENDACIONES PRIORITARIAS PARA ALCANZAR -25:"]
        recomendaciones.append("="*60)
        
        # Prioridad 1: Problemas críticos de reward
        distancias = analisis['distancia_objetivo']
        if distancias:
            peor_caso = max(distancias.items(), key=lambda x: x[1]['distancia_absoluta'])
            mejor_caso = min(distancias.items(), key=lambda x: x[1]['distancia_absoluta'])
            
            recomendaciones.extend([
                f"\n🔴 PRIORIDAD 1 - BREAKTHROUGH CRÍTICO:",
                f"  • Peor caso: {peor_caso[0]} necesita +{peor_caso[1]['mejora_necesaria']:.1f} puntos",
                f"  • Mejor caso: {mejor_caso[0]} necesita +{mejor_caso[1]['mejora_necesaria']:.1f} puntos",
                f"  • ACCIÓN: Implementar RewardShaperBreakthrough con bonificaciones 5x más altas"
            ])
        
        # Prioridad 2: Problemas de convergencia
        no_converge = [k for k, v in convergencia.items() if not v.get('convergio', False)]
        if no_converge:
            recomendaciones.extend([
                f"\n🟡 PRIORIDAD 2 - CONVERGENCIA:",
                f"  • Agentes sin convergencia: {', '.join(no_converge)}",
                f"  • ACCIÓN: Aumentar learning rate a 0.9, episodios a 100k"
            ])
        
        # Prioridad 3: Problemas de supervivencia
        supervivencia = any('Supervivencia muy baja' in p for p in analisis['problemas_criticos'])
        if supervivencia:
            recomendaciones.extend([
                f"\n🔴 PRIORIDAD 3 - SUPERVIVENCIA CRÍTICA:",
                f"  • ACCIÓN INMEDIATA: +20 puntos por paso de supervivencia",
                f"  • Límite de pasos: 1000 → 3000",
                f"  • Pre-entrenamiento solo para supervivencia (ignora precisión)"
            ])
        
        # Sugerencias específicas
        mejoras_rs = self.sugerir_mejoras_reward_shaping(analisis)
        mejoras_hp = self.sugerir_mejoras_hiperparametros(analisis)
        mejoras_arch = self.sugerir_mejoras_arquitectura(analisis)
        
        if mejoras_rs:
            recomendaciones.extend(["\n📈 MEJORAS REWARD SHAPING:"] + mejoras_rs)
        if mejoras_hp:
            recomendaciones.extend(["\n⚙️ MEJORAS HIPERPARÁMETROS:"] + mejoras_hp)
        if mejoras_arch:
            recomendaciones.extend(["\n🏗️ MEJORAS ARQUITECTURA:"] + mejoras_arch)
        
        return recomendaciones
    
    def generar_codigo_mejoras(self) -> str:
        """Genera código Python con las mejoras sugeridas"""
        analisis = self.analizar_rendimiento_actual()
        
        codigo = '''
# 🚀 MEJORAS AUTOMÁTICAS SUGERIDAS PARA ALCANZAR -25

class RewardShaperBreakthrough:
    """Reward shaper ULTRA AGRESIVO para breakthrough -70 → -25"""
    
    def __init__(self):
        self.step_count = 0
        self.best_error = float('inf')
        self.prev_action = 0
        
    def shape_reward(self, obs, action, reward, done):
        shaped = reward
        
        altitude_error = abs(obs['target_altitude'][0] - obs['altitude'][0])
        self.step_count += 1
        
        # SUPERVIVENCIA MASIVA (problema crítico identificado)
        if not done:
            shaped += 20.0  # +20 por cada paso (vs +5 anterior)
            
        # BONIFICACIÓN EXPONENCIAL POR SUPERVIVENCIA
        if self.step_count > 200:
            shaped += self.step_count * 0.1  # Crece con el tiempo
            
        # JACKPOT MEGA por precisión
        if altitude_error < 0.01:
            shaped += 10000.0  # 10x más que antes
        elif altitude_error < 0.05:
            shaped += 5000.0
        elif altitude_error < 0.1:
            shaped += 2000.0
            
        # PENALIZACIÓN CÚBICA por errores grandes
        if altitude_error > 0.2:
            shaped -= (altitude_error ** 3) * 5000
            
        # CONSISTENCIA (problema variabilidad identificado)
        action_consistency = abs(action - self.prev_action)
        if action_consistency < 0.1:
            shaped += 50.0
        else:
            shaped -= action_consistency * 100
            
        self.prev_action = action
        return shaped
    
    def reset(self):
        self.step_count = 0
        self.best_error = float('inf')
        self.prev_action = 0

# HIPERPARÁMETROS BREAKTHROUGH
BREAKTHROUGH_CONFIG = {
    'learning_rate': 0.9,        # Ultra agresivo
    'epsilon': 0.95,             # Exploración máxima
    'epsilon_decay': 0.99999,    # Decay ultra lento
    'epsilon_min': 0.1,          # Nunca dejar de explorar
    'discount_factor': 0.9999,   # Máximo peso al futuro
    'episodes': 150000,          # 1.5x más episodios
    'step_limit': 3000,          # Triple límite de pasos
}

# EARLY STOPPING ADAPTATIVO
def early_stopping_breakthrough(rewards, min_episodes=2000):
    if len(rewards) < min_episodes:
        return False
    
    ventana = 1000
    media_reciente = np.mean(rewards[-ventana:])
    
    # Objetivo progresivo: -50 → -35 → -25
    if len(rewards) < 50000 and media_reciente > -50:
        return True
    elif len(rewards) < 100000 and media_reciente > -35:
        return True
    elif media_reciente > -25:
        return True
    
    return False
'''
        
        return codigo
    
    def crear_visualizaciones(self):
        """Crea visualizaciones avanzadas del análisis"""
        if not self.resultados:
            print("❌ No hay datos para visualizar")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('🎯 ANÁLISIS AVANZADO FLAN - OPTIMIZACIÓN HACIA -25', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Distancia al objetivo
        distancias = self.analizar_rendimiento_actual()['distancia_objetivo']
        if distancias:
            keys = list(distancias.keys())
            mejoras_necesarias = [distancias[k]['mejora_necesaria'] for k in keys]
            
            colors = ['red' if m > 30 else 'orange' if m > 15 else 'green' for m in mejoras_necesarias]
            bars = axes[0,0].bar(range(len(keys)), mejoras_necesarias, color=colors, alpha=0.7)
            axes[0,0].axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Objetivo -25')
            axes[0,0].set_title('Mejora Necesaria para Alcanzar -25')
            axes[0,0].set_ylabel('Puntos de Mejora Necesarios')
            axes[0,0].set_xticks(range(len(keys)))
            axes[0,0].set_xticklabels(keys, rotation=45)
            
            # Añadir valores en las barras
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                             f'{height:.1f}', ha='center', va='bottom')
        
        # Gráfico 2: Convergencia temporal
        convergencia = self.analizar_convergencia()
        for i, (key, data) in enumerate(convergencia.items()):
            if 'medias_ventanas' in data and data['medias_ventanas']:
                axes[0,1].plot(data['medias_ventanas'], label=key, marker='o', alpha=0.7)
        
        axes[0,1].axhline(y=-25, color='green', linestyle='--', alpha=0.7, label='Objetivo -25')
        axes[0,1].axhline(y=-35, color='orange', linestyle='--', alpha=0.7, label='Umbral Crítico -35')
        axes[0,1].set_title('Convergencia por Ventanas de 1000 Episodios')
        axes[0,1].set_ylabel('Recompensa Media')
        axes[0,1].set_xlabel('Ventana de Entrenamiento')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Gráfico 3: Distribución de recompensas
        all_rewards = []
        labels = []
        for scheme_name, scheme_data in self.resultados.items():
            if scheme_name == 'experiment_info':
                continue
            for agent_type, agent_data in scheme_data.items():
                if 'evaluation' in agent_data:
                    all_rewards.append(agent_data['evaluation']['total_rewards'])
                    labels.append(f"{scheme_name}_{agent_type}")
        
        if all_rewards:
            axes[0,2].boxplot(all_rewards, labels=labels)
            axes[0,2].axhline(y=-25, color='green', linestyle='--', alpha=0.7, label='Objetivo -25')
            axes[0,2].axhline(y=-35, color='orange', linestyle='--', alpha=0.7, label='Crítico -35')
            axes[0,2].set_title('Distribución de Recompensas Finales')
            axes[0,2].set_ylabel('Recompensa')
            axes[0,2].tick_params(axis='x', rotation=45)
            axes[0,2].legend()
        
        # Gráfico 4: Supervivencia vs Recompensa
        analisis = self.analizar_rendimiento_actual()
        if analisis['metricas_clave']:
            survival_times = [m['media_steps'] for m in analisis['metricas_clave'].values()]
            rewards = [m['media_reward'] for m in analisis['metricas_clave'].values()]
            keys = list(analisis['metricas_clave'].keys())
            
            scatter = axes[1,0].scatter(survival_times, rewards, c=range(len(keys)), 
                                      s=100, alpha=0.7, cmap='viridis')
            axes[1,0].axhline(y=-25, color='green', linestyle='--', alpha=0.7)
            axes[1,0].set_xlabel('Tiempo de Supervivencia Promedio')
            axes[1,0].set_ylabel('Recompensa Promedio')
            axes[1,0].set_title('Correlación Supervivencia-Recompensa')
            
            # Añadir etiquetas
            for i, key in enumerate(keys):
                axes[1,0].annotate(key, (survival_times[i], rewards[i]), 
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Gráfico 5: Heatmap de performance
        if analisis['metricas_clave']:
            metrics_matrix = []
            metric_names = ['media_reward', 'std_reward', 'media_steps', 'media_error']
            agent_names = list(analisis['metricas_clave'].keys())
            
            for agent in agent_names:
                metrics_matrix.append([
                    analisis['metricas_clave'][agent]['media_reward'],
                    -analisis['metricas_clave'][agent]['std_reward'],  # Negativo porque menor es mejor
                    analisis['metricas_clave'][agent]['media_steps'],
                    -analisis['metricas_clave'][agent]['media_error']   # Negativo porque menor es mejor
                ])
            
            if metrics_matrix:
                im = axes[1,1].imshow(metrics_matrix, cmap='RdYlGn', aspect='auto')
                axes[1,1].set_xticks(range(len(metric_names)))
                axes[1,1].set_xticklabels(['Recompensa', 'Consistencia', 'Supervivencia', 'Precisión'])
                axes[1,1].set_yticks(range(len(agent_names)))
                axes[1,1].set_yticklabels(agent_names)
                axes[1,1].set_title('Mapa de Calor de Performance')
                plt.colorbar(im, ax=axes[1,1])
        
        # Gráfico 6: Recomendaciones prioritarias (texto)
        recomendaciones = self.generar_recomendaciones_prioritarias()
        texto_rec = '\n'.join(recomendaciones[:15])  # Mostrar solo las primeras 15 líneas
        axes[1,2].text(0.05, 0.95, texto_rec, transform=axes[1,2].transAxes, 
                      fontsize=8, verticalalignment='top', fontfamily='monospace')
        axes[1,2].set_title('Recomendaciones Prioritarias')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig('analisis_optimizacion_flan.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generar_reporte_completo(self):
        """Genera un reporte completo del análisis"""
        print("🎯 ANALIZADOR AVANZADO FLAN - REPORTE COMPLETO")
        print("="*80)
        
        # Información básica
        if 'experiment_info' in self.resultados:
            info = self.resultados['experiment_info']
            print(f"📊 Experimento: {info.get('optimization', 'N/A')}")
            print(f"🎯 Objetivo: {info.get('objective', 'N/A')}")
            print(f"📈 Total Episodios: {info.get('total_episodes', 'N/A'):,}")
            print(f"⏱️  Tiempo Estimado: {info.get('estimated_time_hours', 'N/A')} horas")
        
        # Análisis de rendimiento
        analisis = self.analizar_rendimiento_actual()
        print(f"\n🔍 ANÁLISIS DE RENDIMIENTO ACTUAL:")
        print(f"  • Problemas críticos: {len(analisis['problemas_criticos'])}")
        print(f"  • Problemas moderados: {len(analisis['problemas_moderados'])}")
        print(f"  • Puntos fuertes: {len(analisis['puntos_fuertes'])}")
        
        if analisis['problemas_criticos']:
            print("\n🔴 PROBLEMAS CRÍTICOS:")
            for problema in analisis['problemas_criticos']:
                print(f"  • {problema}")
        
        # Análisis de convergencia
        convergencia = self.analizar_convergencia()
        print(f"\n📈 ANÁLISIS DE CONVERGENCIA:")
        for key, data in convergencia.items():
            convergio = "✅" if data.get('convergio', False) else "❌"
            plateando = "🔄" if data.get('plateando', False) else "📈"
            print(f"  • {key}: {convergio} Mejora: {data.get('mejora_total', 0):.1f} {plateando}")
        
        # Recomendaciones
        recomendaciones = self.generar_recomendaciones_prioritarias()
        print("\n" + "\n".join(recomendaciones))
        
        # Código de mejoras
        print(f"\n💻 CÓDIGO DE MEJORAS SUGERIDAS:")
        print(self.generar_codigo_mejoras())
        
        # Crear visualizaciones
        self.crear_visualizaciones()

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizador de resultados FLAN')
    parser.add_argument('--json', '-j', default='flan_results_196k_ultra.json',
                       help='Archivo JSON de resultados')
    parser.add_argument('--output', '-o', default='mejoras_sugeridas.py',
                       help='Archivo de salida para código de mejoras')
    
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
    analizador = AnalizadorResultadosFLAN(args.json)
    analizador.generar_reporte_completo()
    
    # Guardar código de mejoras
    with open(args.output, 'w') as f:
        f.write(analizador.generar_codigo_mejoras())
    
    print(f"\n✅ Análisis completo. Código de mejoras guardado en: {args.output}")
    print(f"📊 Visualizaciones guardadas en: analisis_optimizacion_flan.png")

if __name__ == "__main__":
    main() 