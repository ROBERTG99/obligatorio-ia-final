#!/usr/bin/env python3
"""
Script para generar gráficos y reportes del experimento BORED
usando resultados ya calculados
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

def load_results(filename: str = 'bored_results.json') -> Dict[str, Any]:
    """Carga los resultados del experimento"""
    with open(filename, 'r') as f:
        return json.load(f)

def generate_plots_corrected(results: Dict, alpha_beta_results: Optional[Dict] = None):
    """Genera gráficos corregidos de los resultados"""
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Resultados del torneo
    agent_names = list(results['agent_stats'].keys())
    win_rates = []
    
    for name in agent_names:
        stats = results['agent_stats'][name]
        if stats['total_games'] > 0:
            win_rate = stats['wins'] / stats['total_games']
        else:
            win_rate = 0
        win_rates.append(win_rate)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico de tasas de victoria
    bars = axes[0, 0].bar(agent_names, win_rates, color='skyblue')
    axes[0, 0].set_ylabel('Tasa de Victoria')
    axes[0, 0].set_title('Tasas de Victoria por Agente')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Anotar valores en las barras
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{rate:.2f}', ha='center', va='bottom')
    
    # 2. Comparación Minimax vs Expectimax (corregido)
    minimax_agents = [name for name in agent_names if 'Minimax' in name and 'NoPruning' not in name]
    expectimax_agents = [name for name in agent_names if 'Expectimax' in name and 'NoPruning' not in name]
    
    # Extraer profundidades para hacer matching correcto
    minimax_depths = []
    minimax_rates = []
    for name in minimax_agents:
        if results['agent_stats'][name]['total_games'] > 0:
            depth = name.split('_')[1]
            minimax_depths.append(depth)
            rate = results['agent_stats'][name]['wins'] / results['agent_stats'][name]['total_games']
            minimax_rates.append(rate)
    
    expectimax_depths = []
    expectimax_rates = []
    for name in expectimax_agents:
        if results['agent_stats'][name]['total_games'] > 0:
            depth = name.split('_')[1]
            expectimax_depths.append(depth)
            rate = results['agent_stats'][name]['wins'] / results['agent_stats'][name]['total_games']
            expectimax_rates.append(rate)
    
    # Encontrar profundidades comunes
    common_depths = list(set(minimax_depths) & set(expectimax_depths))
    common_depths.sort()
    
    # Filtrar datos para profundidades comunes
    common_minimax_rates = []
    common_expectimax_rates = []
    
    for depth in common_depths:
        if depth in minimax_depths:
            idx = minimax_depths.index(depth)
            common_minimax_rates.append(minimax_rates[idx])
        
        if depth in expectimax_depths:
            idx = expectimax_depths.index(depth)
            common_expectimax_rates.append(expectimax_rates[idx])
    
    if common_depths:
        x = np.arange(len(common_depths))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, common_minimax_rates, width, label='Minimax', alpha=0.8)
        axes[0, 1].bar(x + width/2, common_expectimax_rates, width, label='Expectimax', alpha=0.8)
        axes[0, 1].set_xlabel('Profundidad')
        axes[0, 1].set_ylabel('Tasa de Victoria')
        axes[0, 1].set_title('Minimax vs Expectimax por Profundidad')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(common_depths)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No hay profundidades comunes', 
                       transform=axes[0, 1].transAxes, ha='center', va='center')
    
    # 3. Análisis de rendimiento por tipo de agente
    agent_types = {}
    for name in agent_names:
        if 'Minimax' in name:
            agent_type = 'Minimax'
        elif 'Expectimax' in name:
            agent_type = 'Expectimax'
        elif 'Trainer' in name:
            agent_type = 'Trainer'
        elif 'Random' in name:
            agent_type = 'Random'
        else:
            agent_type = 'Otros'
        
        if agent_type not in agent_types:
            agent_types[agent_type] = []
        
        stats = results['agent_stats'][name]
        if stats['total_games'] > 0:
            win_rate = stats['wins'] / stats['total_games']
            agent_types[agent_type].append(win_rate)
    
    # Box plot por tipo de agente
    data_for_boxplot = []
    labels_for_boxplot = []
    for agent_type, rates in agent_types.items():
        if rates:  # Solo si hay datos
            data_for_boxplot.append(rates)
            labels_for_boxplot.append(agent_type)
    
    if data_for_boxplot:
        axes[1, 0].boxplot(data_for_boxplot, tick_labels=labels_for_boxplot)
        axes[1, 0].set_ylabel('Tasa de Victoria')
        axes[1, 0].set_title('Distribución de Rendimiento por Tipo de Agente')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Ranking de mejores agentes
    sorted_agents = sorted(results['agent_stats'].items(), 
                          key=lambda x: x[1]['wins'] / x[1]['total_games'] if x[1]['total_games'] > 0 else 0,
                          reverse=True)
    
    top_agents = sorted_agents[:10]  # Top 10
    top_names = [name for name, _ in top_agents]
    top_rates = [stats['wins'] / stats['total_games'] if stats['total_games'] > 0 else 0 
                for _, stats in top_agents]
    
    y_pos = np.arange(len(top_names))
    axes[1, 1].barh(y_pos, top_rates, color='lightgreen')
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(top_names)
    axes[1, 1].set_xlabel('Tasa de Victoria')
    axes[1, 1].set_title('Top 10 Agentes')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Anotar valores
    for i, rate in enumerate(top_rates):
        axes[1, 1].text(rate + 0.01, i, f'{rate:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig('bored_results_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_report(results: Dict):
    """Genera un reporte detallado de los resultados"""
    
    print("\n" + "="*80)
    print("REPORTE DETALLADO - EXPERIMENTO BORED")
    print("="*80)
    
    # Estadísticas generales
    total_agents = len(results['agent_stats'])
    total_matchups = len(results['matchups'])
    total_games = sum(stats['total_games'] for stats in results['agent_stats'].values()) // 2
    
    print(f"\n📊 ESTADÍSTICAS GENERALES")
    print(f"   • Total de agentes: {total_agents}")
    print(f"   • Total de matchups: {total_matchups}")
    print(f"   • Total de partidas: {total_games}")
    
    # Mejores agentes
    print(f"\n🏆 TOP 10 AGENTES")
    print("-" * 50)
    
    agent_stats = results['agent_stats']
    sorted_agents = sorted(agent_stats.items(), 
                          key=lambda x: x[1]['wins'] / x[1]['total_games'] if x[1]['total_games'] > 0 else 0,
                          reverse=True)
    
    for i, (agent_name, stats) in enumerate(sorted_agents[:10]):
        if stats['total_games'] > 0:
            win_rate = stats['wins'] / stats['total_games']
            print(f"{i+1:2d}. {agent_name:<25} {win_rate:.3f} ({stats['wins']}W/{stats['losses']}L)")
    
    # Análisis por tipo
    print(f"\n🎯 ANÁLISIS POR TIPO DE AGENTE")
    print("-" * 50)
    
    agent_types = {}
    for agent_name, stats in agent_stats.items():
        if 'Minimax' in agent_name:
            agent_type = 'Minimax'
        elif 'Expectimax' in agent_name:
            agent_type = 'Expectimax'
        elif 'Trainer' in agent_name:
            agent_type = 'Trainer'
        elif 'Random' in agent_name:
            agent_type = 'Random'
        else:
            agent_type = 'Otros'
        
        if agent_type not in agent_types:
            agent_types[agent_type] = []
        
        if stats['total_games'] > 0:
            win_rate = stats['wins'] / stats['total_games']
            agent_types[agent_type].append(win_rate)
    
    for agent_type, rates in agent_types.items():
        if rates:
            avg_rate = np.mean(rates)
            std_rate = np.std(rates)
            print(f"{agent_type:<12}: {avg_rate:.3f} ± {std_rate:.3f} (n={len(rates)})")
    
    # Análisis de profundidad
    print(f"\n📈 ANÁLISIS POR PROFUNDIDAD")
    print("-" * 50)
    
    depth_performance = {}
    for agent_name, stats in agent_stats.items():
        if ('Minimax' in agent_name or 'Expectimax' in agent_name) and 'NoPruning' not in agent_name:
            try:
                depth = agent_name.split('_')[1]  # Ej: 'Minimax_D4' -> 'D4'
                if depth not in depth_performance:
                    depth_performance[depth] = []
                
                if stats['total_games'] > 0:
                    win_rate = stats['wins'] / stats['total_games']
                    depth_performance[depth].append(win_rate)
            except:
                continue
    
    for depth in sorted(depth_performance.keys()):
        rates = depth_performance[depth]
        if rates:
            avg_rate = np.mean(rates)
            std_rate = np.std(rates)
            print(f"{depth:<4}: {avg_rate:.3f} ± {std_rate:.3f} (n={len(rates)})")
    
    # Matchups más interesantes
    print(f"\n⚔️  MATCHUPS MÁS COMPETITIVOS")
    print("-" * 50)
    
    competitive_matchups = []
    for matchup_name, matchup_data in results['matchups'].items():
        total_games = matchup_data['agent1_wins'] + matchup_data['agent2_wins']
        if total_games > 0:
            win_rate_diff = abs(matchup_data['agent1_wins'] - matchup_data['agent2_wins']) / total_games
            competitive_matchups.append((matchup_name, win_rate_diff, matchup_data))
    
    # Ordenar por menor diferencia (más competitivo)
    competitive_matchups.sort(key=lambda x: x[1])
    
    for i, (matchup_name, diff, data) in enumerate(competitive_matchups[:5]):
        agent1, agent2 = matchup_name.split('_vs_')
        print(f"{i+1}. {agent1} vs {agent2}")
        print(f"   Resultado: {data['agent1_wins']}-{data['agent2_wins']} (diferencia: {diff:.3f})")
    
    print(f"\n✅ CONCLUSIONES")
    print("-" * 50)
    print("• Experimento completado exitosamente")
    print("• Datos guardados y analizados")
    print("• Gráficos generados correctamente")
    print("• Reporte disponible para análisis detallado")

def main():
    """Función principal"""
    print("Cargando resultados del experimento BORED...")
    
    try:
        results = load_results()
        print("✅ Resultados cargados exitosamente")
        
        # Generar gráficos corregidos
        print("\nGenerando gráficos corregidos...")
        generate_plots_corrected(results, None)
        print("✅ Gráficos generados: bored_results_corrected.png")
        
        # Generar reporte detallado
        generate_detailed_report(results)
        
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo bored_results.json")
        print("   Asegúrate de estar en el directorio correcto")
    except Exception as e:
        print(f"❌ Error al procesar los resultados: {e}")

if __name__ == "__main__":
    main() 