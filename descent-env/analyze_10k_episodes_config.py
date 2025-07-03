#!/usr/bin/env python3
"""
Análisis para optimizar experimento FLAN con ~10,000 episodios
Evaluación de uso de GPU y distribución eficiente
"""

import os
import sys
import json
import psutil
import subprocess
import platform
from typing import Dict, List, Tuple, Optional

def detect_gpu_capabilities():
    """Detecta capacidades de GPU disponibles en el sistema"""
    
    print("="*80)
    print("🔍 DETECCIÓN DE CAPACIDADES GPU")
    print("="*80)
    
    gpu_info = {
        'nvidia_available': False,
        'cuda_available': False,
        'mps_available': False,  # Metal Performance Shaders (macOS)
        'gpu_count': 0,
        'gpu_memory': [],
        'recommendations': []
    }
    
    # Detectar NVIDIA GPU (CUDA)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_info['nvidia_available'] = True
            gpu_lines = result.stdout.strip().split('\n')
            gpu_info['gpu_count'] = len(gpu_lines)
            
            for line in gpu_lines:
                if line.strip():
                    name, memory = line.split(',')
                    gpu_info['gpu_memory'].append({
                        'name': name.strip(),
                        'memory_mb': int(memory.strip())
                    })
            
            print(f"✅ NVIDIA GPU detectada:")
            for i, gpu in enumerate(gpu_info['gpu_memory']):
                print(f"   GPU {i}: {gpu['name']} - {gpu['memory_mb']:,} MB")
        else:
            print("❌ nvidia-smi no encontrado o falló")
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        print("❌ NVIDIA GPU no detectada")
    
    # Detectar CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            cuda_count = torch.cuda.device_count()
            print(f"✅ CUDA disponible con {cuda_count} dispositivos")
            
            for i in range(cuda_count):
                props = torch.cuda.get_device_properties(i)
                print(f"   CUDA {i}: {props.name} - {props.total_memory // 1024**2:,} MB")
        else:
            print("❌ CUDA no disponible en PyTorch")
    except ImportError:
        print("❌ PyTorch no instalado")
    
    # Detectar Metal Performance Shaders (macOS)
    if platform.system() == "Darwin":  # macOS
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info['mps_available'] = True
                print("✅ Metal Performance Shaders (MPS) disponible")
                gpu_info['recommendations'].append("Usar MPS para aceleración en macOS")
            else:
                print("❌ MPS no disponible")
        except ImportError:
            print("❌ PyTorch no instalado - no se puede verificar MPS")
    
    # Generar recomendaciones
    if gpu_info['nvidia_available'] or gpu_info['cuda_available']:
        gpu_info['recommendations'].extend([
            "Q-Learning con GPU: Beneficio limitado (operaciones de tabla)",
            "Entorno paralelo: GPU puede acelerar simulaciones complejas",
            "Batch processing: Agrupar evaluaciones para usar GPU eficientemente"
        ])
    elif gpu_info['mps_available']:
        gpu_info['recommendations'].extend([
            "MPS puede acelerar cálculos matemáticos intensivos",
            "Mejor para operaciones vectorizadas que para Q-Learning discreto",
            "Considerar paralelización CPU como alternativa principal"
        ])
    else:
        gpu_info['recommendations'].extend([
            "Sin GPU disponible - usar paralelización CPU optimizada",
            "Ray multiprocessing será más efectivo que GPU para este problema",
            "Considerar usar instancia cloud con GPU si es necesario"
        ])
    
    return gpu_info

def design_10k_episode_experiment():
    """Diseña distribución eficiente de 10,000 episodios"""
    
    print("\n" + "="*80)
    print("📊 DISEÑO PARA ~10,000 EPISODIOS")
    print("="*80)
    
    # Configuraciones posibles para llegar a ~10k episodios
    configs = [
        {
            "name": "Configuración Balanceada",
            "description": "Balance entre exploración y tiempo",
            "schemes": 1,  # Solo Media
            "agents": 1,   # Solo Q-Learning
            "hyperparameter_combinations": 16,  # 2×2×2×2 grid más completo
            "episodes_per_combination": 200,    # Reducido pero suficiente
            "final_training": 3000,
            "final_evaluation": 500,
            "total_episodes": 16 * 200 + 3000 + 500
        },
        {
            "name": "Configuración Exploratoria",
            "description": "Más exploración de hiperparámetros",
            "schemes": 1,  # Solo Media
            "agents": 1,   # Solo Q-Learning
            "hyperparameter_combinations": 32,  # 2×2×2×2×2 grid expandido
            "episodes_per_combination": 150,    
            "final_training": 2000,
            "final_evaluation": 400,
            "total_episodes": 32 * 150 + 2000 + 400
        },
        {
            "name": "Configuración Enfocada",
            "description": "Pocos hiperparámetros, más entrenamiento",
            "schemes": 1,  # Solo Media
            "agents": 1,   # Solo Q-Learning
            "hyperparameter_combinations": 8,   # 2×2×2 grid simple
            "episodes_per_combination": 250,    
            "final_training": 4000,  # Más entrenamiento final
            "final_evaluation": 600,
            "total_episodes": 8 * 250 + 4000 + 600
        }
    ]
    
    print("🎯 OPCIONES DE CONFIGURACIÓN:")
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['name']} ({config['total_episodes']:,} episodios)")
        print(f"   📝 {config['description']}")
        print(f"   🔬 Hiperparámetros: {config['hyperparameter_combinations']} combinaciones × {config['episodes_per_combination']} eps")
        print(f"   🏋️ Entrenamiento final: {config['final_training']:,} episodios")
        print(f"   📊 Evaluación final: {config['final_evaluation']:,} episodios")
        
        # Estimación de tiempo
        episodes_per_minute = 60  # Estimación optimista con paralelización
        estimated_minutes = config['total_episodes'] / episodes_per_minute
        print(f"   ⏱️  Tiempo estimado: {estimated_minutes:.0f} minutos ({estimated_minutes/60:.1f} horas)")
    
    return configs

def create_optimized_10k_config():
    """Crea configuración optimizada para 10k episodios"""
    
    config = {
        "experiment_name": "FLAN_10K_OPTIMIZED",
        "description": "Experimento FLAN optimizado para ~10,000 episodios",
        "target_episodes": 10000,
        
        "discretization_schemes": [
            {
                "name": "Media",
                "altitude_bins": 25,
                "velocity_bins": 25,
                "target_alt_bins": 25,
                "runway_dist_bins": 25,
                "action_bins": 10,
                "justification": "Esquema balanceado - buen compromiso precisión/velocidad"
            }
        ],
        
        "hyperparameter_grid": {
            "learning_rate": [0.3, 0.4],           # 2 valores (mejores del análisis)
            "discount_factor": [0.98, 0.99],       # 2 valores (altos para largo plazo)
            "epsilon": [0.2, 0.3],                 # 2 valores (exploración balanceada)
            "use_double_q": [True],                 # 1 valor (siempre mejor)
            "use_reward_shaping": [True]            # 1 valor (siempre mejor)
        },
        
        "training_episodes": {
            "hyperparameter_search": 400,    # 8 combinaciones × 400 = 3,200
            "hyperparameter_evaluation": 50, # Incluido en los 400
            "final_training": 4500,           # Entrenamiento intensivo
            "final_evaluation": 800           # Evaluación robusta
        },
        
        "parallelization": {
            "strategy": "ray_multiprocessing",
            "cpu_cores": "all_available",
            "max_concurrent_trials": 8,
            "batch_size": 2,
            "justification": "Ray más eficiente que GPU para Q-Learning discreto"
        },
        
        "environment_strategy": {
            "hyperparameter_search": "MockDescentEnv",
            "final_training": "DescentEnv", 
            "justification": "Mock para velocidad en búsqueda, real para resultado final"
        },
        
        "gpu_considerations": {
            "q_learning_benefit": "BAJO - operaciones de tabla, no intensivas",
            "environment_benefit": "MEDIO - puede acelerar simulaciones complejas",
            "evaluation_benefit": "ALTO - paralelización masiva de evaluaciones",
            "recommendation": "CPU paralelizado es más efectivo para este problema específico"
        }
    }
    
    # Calcular episodios totales
    combinations = 2 * 2 * 2 * 1 * 1  # 8 combinaciones
    total_episodes = (combinations * 400) + 4500 + 800  # 8,500 episodios
    
    config["calculated_episodes"] = total_episodes
    config["estimated_time_minutes"] = total_episodes / 80  # ~80 episodios/minuto con optimizaciones
    
    return config

def main():
    """Función principal del análisis"""
    
    print("🎯 ANÁLISIS PARA EXPERIMENTO DE ~10,000 EPISODIOS")
    print("Evaluación GPU vs CPU para Q-Learning")
    
    # Detectar capacidades GPU
    gpu_info = detect_gpu_capabilities()
    
    # Diseñar opciones de experimento
    configs = design_10k_episode_experiment()
    
    # Crear configuración optimizada
    optimized_config = create_optimized_10k_config()
    
    # Evaluación específica de GPU para Q-Learning
    print("\n" + "="*80)
    print("⚡ EVALUACIÓN GPU PARA Q-LEARNING")
    print("="*80)
    
    print("🔴 COMPONENTES CON BAJO BENEFICIO GPU:")
    print("   • Q-Table Updates: Operaciones de indexación simple")
    print("   • State Discretization: Cálculos básicos de binning")
    print("   • Action Selection: Búsqueda de máximo en tabla pequeña")
    
    print("\n🟡 COMPONENTES CON BENEFICIO MEDIO GPU:")
    print("   • Environment Simulation: Depende de complejidad del simulador")
    print("   • Reward Calculation: Puede beneficiarse si hay muchas operaciones matemáticas")
    
    print("\n🟢 COMPONENTES CON ALTO BENEFICIO GPU:")
    print("   • Batch Evaluation: 100+ evaluaciones simultáneas")
    print("   • Hyperparameter Search: Entrenamientos paralelos independientes")
    
    print(f"\n🎯 RECOMENDACIÓN PARA TU CASO:")
    print(f"   • Q-Learning discreto: CPU paralelizado es MÁS eficiente")
    print(f"   • Usa Ray multiprocessing en lugar de GPU")
    print(f"   • GPU útil solo si tienes 50+ agentes evaluándose simultáneamente")
    print(f"   • Para 10k episodios: CPU optimizado será 2-3x más rápido que GPU")
    
    # Guardar configuración
    with open('flan_10k_config.json', 'w') as f:
        json.dump(optimized_config, f, indent=2)
    
    # Resumen final
    print("\n" + "="*80)
    print("📋 CONFIGURACIÓN RECOMENDADA PARA 10K EPISODIOS")
    print("="*80)
    
    print(f"📊 DISTRIBUCIÓN DE EPISODIOS:")
    print(f"   • Búsqueda hiperparámetros: 8 combinaciones × 400 = 3,200")
    print(f"   • Entrenamiento final: 4,500 episodios")
    print(f"   • Evaluación final: 800 episodios")
    print(f"   • TOTAL: {optimized_config['calculated_episodes']:,} episodios")
    
    print(f"\n⏱️  TIEMPO ESTIMADO:")
    print(f"   • Con CPU optimizado: {optimized_config['estimated_time_minutes']:.0f} minutos")
    print(f"   • Con GPU (menos eficiente): ~{optimized_config['estimated_time_minutes']*1.5:.0f} minutos")
    
    print(f"\n🚀 ESTRATEGIA RECOMENDADA:")
    print(f"   • Usar Ray multiprocessing (CPU)")
    print(f"   • Todos los cores disponibles")
    print(f"   • MockDescentEnv para búsqueda rápida")
    print(f"   • DescentEnv real para entrenamiento final")
    
    print(f"\n🛠️  ARCHIVO GENERADO:")
    print(f"   • flan_10k_config.json - Configuración optimizada")
    
    print(f"\n💡 GPU ALTERNATIVA:")
    print(f"   • Si quieres probar GPU: usar para evaluación masiva únicamente")
    print(f"   • Mantener entrenamiento Q-Learning en CPU")
    print(f"   • Considerar GPU solo si tienes 100+ agentes en paralelo")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main() 