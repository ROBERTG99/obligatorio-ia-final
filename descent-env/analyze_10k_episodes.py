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

def evaluate_gpu_benefits():
    """Evalúa beneficios específicos de GPU para Q-Learning"""
    
    print("\n" + "="*80)
    print("⚡ EVALUACIÓN DE BENEFICIOS GPU PARA Q-LEARNING")
    print("="*80)
    
    components = [
        {
            "component": "Q-Table Updates",
            "description": "Actualización de valores Q discretos",
            "gpu_benefit": "BAJO",
            "reason": "Operaciones de indexación y lookup, no intensivas computacionalmente",
            "cpu_better": True
        },
        {
            "component": "State Discretization", 
            "description": "Conversión de estados continuos a discretos",
            "gpu_benefit": "BAJO",
            "reason": "Operaciones simples de mapping y clipping",
            "cpu_better": True
        },
        {
            "component": "Environment Simulation",
            "description": "Simulación del entorno de descenso",
            "gpu_benefit": "MEDIO",
            "reason": "Depende de la complejidad de la física del simulador",
            "cpu_better": False
        },
        {
            "component": "Batch Evaluation",
            "description": "Evaluación de múltiples episodios",
            "gpu_benefit": "MEDIO-ALTO",
            "reason": "Paralelización masiva de evaluaciones independientes",
            "cpu_better": False
        },
        {
            "component": "Hyperparameter Search",
            "description": "Búsqueda paralela de hiperparámetros",
            "gpu_benefit": "ALTO",
            "reason": "Múltiples entrenamientos independientes simultáneos",
            "cpu_better": False
        }
    ]
    
    print("📋 ANÁLISIS POR COMPONENTE:")
    for comp in components:
        benefit_emoji = {
            "BAJO": "🔴",
            "MEDIO": "🟡", 
            "MEDIO-ALTO": "🟠",
            "ALTO": "🟢"
        }
        
        print(f"\n{benefit_emoji[comp['gpu_benefit']]} {comp['component']}")
        print(f"   📄 {comp['description']}")
        print(f"   ⚡ Beneficio GPU: {comp['gpu_benefit']}")
        print(f"   💭 Razón: {comp['reason']}")
        print(f"   🏆 CPU mejor: {'Sí' if comp['cpu_better'] else 'No'}")
    
    print(f"\n🎯 RECOMENDACIÓN GENERAL:")
    print(f"   • Q-Learning discreto se beneficia MÁS de paralelización CPU")
    print(f"   • GPU útil principalmente para evaluación masiva en batch")
    print(f"   • Ray multiprocessing probablemente más efectivo que GPU")
    print(f"   • GPU justificado solo si hay >100 evaluaciones simultáneas")

def create_gpu_optimized_config():
    """Crea configuración que aprovecha GPU cuando está disponible"""
    
    gpu_info = detect_gpu_capabilities()
    
    # Configuración base para 10k episodios
    base_config = {
        "experiment_name": "FLAN_10K_OPTIMIZED",
        "description": "Experimento FLAN optimizado para ~10,000 episodios con consideraciones GPU",
        "target_episodes": 10000,
        
        "discretization_schemes": [
            {
                "name": "Media",
                "altitude_bins": 25,
                "velocity_bins": 25,
                "target_alt_bins": 25,
                "runway_dist_bins": 25,
                "action_bins": 10
            }
        ],
        
        "hyperparameter_grid": {
            "learning_rate": [0.2, 0.3, 0.4, 0.5],      # 4 valores
            "discount_factor": [0.95, 0.98, 0.99],       # 3 valores
            "epsilon": [0.1, 0.2, 0.3],                  # 3 valores
            "use_double_q": [True, False],                # 2 valores
            "use_reward_shaping": [True]                  # 1 valor
        },
        
        "training_episodes": {
            "hyperparameter_search": 200,    # 72 combinaciones × 200 = 14,400
            "hyperparameter_evaluation": 50, # Incluido en los 200
            "final_training": 2000,           # 2,000 episodios finales
            "final_evaluation": 500           # 500 evaluaciones
        }
    }
    
    # Calcular episodios totales
    combinations = 4 * 3 * 3 * 2 * 1  # 72 combinaciones
    total_episodes = (combinations * 200) + 2000 + 500  # 16,900 episodios
    
    # Ajustar si supera 10k significativamente
    if total_episodes > 12000:
        # Reducir grid search
        base_config["hyperparameter_grid"] = {
            "learning_rate": [0.3, 0.4],           # 2 valores
            "discount_factor": [0.98, 0.99],       # 2 valores  
            "epsilon": [0.2, 0.3],                 # 2 valores
            "use_double_q": [True],                 # 1 valor
            "use_reward_shaping": [True]            # 1 valor
        }
        combinations = 2 * 2 * 2 * 1 * 1  # 8 combinaciones
        base_config["training_episodes"]["hyperparameter_search"] = 400  # Más episodios por combinación
        total_episodes = (8 * 400) + 2000 + 500  # 5,700 episodios
        
        # Aumentar entrenamiento final para llegar a ~10k
        base_config["training_episodes"]["final_training"] = 4000
        base_config["training_episodes"]["final_evaluation"] = 800
        total_episodes = (8 * 400) + 4000 + 800  # 8,000 episodios
    
    # Configuración específica según GPU disponible
    if gpu_info['cuda_available'] or gpu_info['nvidia_available']:
        base_config["gpu_strategy"] = {
            "enabled": True,
            "type": "CUDA",
            "batch_evaluation": True,
            "batch_size": 32,
            "parallel_envs": 16,
            "justification": "CUDA disponible - usar para evaluación en batch y entornos paralelos"
        }
    elif gpu_info['mps_available']:
        base_config["gpu_strategy"] = {
            "enabled": True,
            "type": "MPS", 
            "batch_evaluation": True,
            "batch_size": 16,
            "parallel_envs": 8,
            "justification": "MPS disponible - usar para operaciones vectorizadas"
        }
    else:
        base_config["gpu_strategy"] = {
            "enabled": False,
            "type": "CPU_ONLY",
            "cpu_cores": "all_available",
            "justification": "Sin GPU - usar paralelización CPU optimizada con Ray"
        }
    
    # Configuración de paralelización
    cpu_count = psutil.cpu_count(logical=True)
    base_config["parallelization"] = {
        "strategy": "hybrid" if base_config["gpu_strategy"]["enabled"] else "cpu_only",
        "cpu_cores": cpu_count,
        "max_concurrent_trials": min(cpu_count, 16),
        "environment_parallelization": base_config["gpu_strategy"]["enabled"]
    }
    
    base_config["estimated_episodes"] = total_episodes
    base_config["estimated_time_minutes"] = total_episodes / 100  # Estimación optimista
    
    return base_config

def generate_implementation_code():
    """Genera código de implementación optimizado"""
    
    implementation_code = '''
# Implementación optimizada para 10K episodios con soporte GPU

import os
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import ray
from ray.util.multiprocessing import Pool

class GPUOptimizedEvaluator:
    """Evaluador que aprovecha GPU cuando está disponible"""
    
    def __init__(self, use_gpu=False, device='cuda'):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = device if self.use_gpu else 'cpu'
        
    def batch_evaluate(self, agents, environments, num_episodes_per_agent=100):
        """Evaluación en batch aprovechando paralelización GPU/CPU"""
        
        if self.use_gpu:
            return self._gpu_batch_evaluate(agents, environments, num_episodes_per_agent)
        else:
            return self._cpu_batch_evaluate(agents, environments, num_episodes_per_agent)
    
    def _gpu_batch_evaluate(self, agents, environments, num_episodes):
        """Evaluación usando GPU para operaciones vectorizadas"""
        results = []
        
        # Agrupar evaluaciones en batches para GPU
        batch_size = 32
        for i in range(0, len(agents), batch_size):
            batch_agents = agents[i:i+batch_size]
            batch_results = []
            
            # Crear múltiples entornos en paralelo
            with ThreadPoolExecutor(max_workers=len(batch_agents)) as executor:
                futures = []
                for agent in batch_agents:
                    future = executor.submit(self._evaluate_single_agent_gpu, 
                                           agent, environments[0], num_episodes)
                    futures.append(future)
                
                for future in futures:
                    batch_results.append(future.result())
            
            results.extend(batch_results)
        
        return results
    
    def _cpu_batch_evaluate(self, agents, environments, num_episodes):
        """Evaluación usando paralelización CPU optimizada"""
        
        # Usar Ray para paralelización eficiente
        @ray.remote
        def evaluate_agent_remote(agent, env, episodes):
            return self._evaluate_single_agent_cpu(agent, env, episodes)
        
        # Ejecutar evaluaciones en paralelo
        futures = []
        for agent in agents:
            future = evaluate_agent_remote.remote(agent, environments[0], num_episodes)
            futures.append(future)
        
        results = ray.get(futures)
        return results
    
    def _evaluate_single_agent_gpu(self, agent, env, num_episodes):
        """Evaluación de un agente con optimizaciones GPU"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Si el agente puede usar GPU para decisiones
                if hasattr(agent, 'get_action_gpu'):
                    action = agent.get_action_gpu(obs, device=self.device)
                else:
                    action = agent.get_action(obs, training=False)
                
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
            
            episode_rewards.append(total_reward)
        
        return {
            'rewards': episode_rewards,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards)
        }
    
    def _evaluate_single_agent_cpu(self, agent, env, num_episodes):
        """Evaluación de un agente optimizada para CPU"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.get_action(obs, training=False)
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
            
            episode_rewards.append(total_reward)
        
        return {
            'rewards': episode_rewards,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards)
        }

class HybridHyperparameterOptimizer:
    """Optimizador que combina CPU y GPU eficientemente"""
    
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.evaluator = GPUOptimizedEvaluator(use_gpu)
    
    def optimize(self, param_grid, agent_class, env, discretization):
        """Optimización híbrida de hiperparámetros"""
        
        if self.use_gpu:
            # GPU: Usar batch processing para evaluaciones
            return self._gpu_optimize(param_grid, agent_class, env, discretization)
        else:
            # CPU: Usar Ray para paralelización masiva
            return self._cpu_optimize(param_grid, agent_class, env, discretization)
    
    def _gpu_optimize(self, param_grid, agent_class, env, discretization):
        """Optimización aprovechando GPU para evaluaciones"""
        
        # Entrenar agentes en CPU (más eficiente para Q-Learning)
        # Evaluar en GPU (paralelización masiva)
        
        import itertools
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        best_score = -np.inf
        best_params = None
        
        # Procesar en batches para aprovechar GPU
        batch_size = 8  # Número de agentes a evaluar simultáneamente
        
        for i in range(0, len(param_combinations), batch_size):
            batch_combinations = param_combinations[i:i+batch_size]
            
            # Entrenar batch de agentes en CPU
            agents = []
            for combination in batch_combinations:
                params = dict(zip(param_names, combination))
                agent = agent_class(discretization, **params)
                
                # Entrenamiento rápido en CPU
                self._train_agent_cpu(agent, env, episodes=200)
                agents.append(agent)
            
            # Evaluar batch en GPU
            batch_results = self.evaluator.batch_evaluate(
                agents, [env], num_episodes_per_agent=50
            )
            
            # Actualizar mejor resultado
            for j, result in enumerate(batch_results):
                if result['mean_reward'] > best_score:
                    best_score = result['mean_reward']
                    best_params = dict(zip(param_names, batch_combinations[j]))
        
        return {'best_params': best_params, 'best_score': best_score}
    
    def _cpu_optimize(self, param_grid, agent_class, env, discretization):
        """Optimización usando solo CPU con Ray"""
        
        @ray.remote
        def train_and_evaluate(params):
            agent = agent_class(discretization, **params)
            self._train_agent_cpu(agent, env, episodes=200)
            result = self.evaluator._evaluate_single_agent_cpu(agent, env, 50)
            return params, result['mean_reward']
        
        import itertools
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        # Ejecutar todas las combinaciones en paralelo
        futures = []
        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            future = train_and_evaluate.remote(params)
            futures.append(future)
        
        # Recoger resultados
        results = ray.get(futures)
        
        # Encontrar mejor resultado
        best_score = -np.inf
        best_params = None
        
        for params, score in results:
            if score > best_score:
                best_score = score
                best_params = params
        
        return {'best_params': best_params, 'best_score': best_score}
    
    def _train_agent_cpu(self, agent, env, episodes):
        """Entrenamiento rápido en CPU"""
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                state = agent.discretization.get_state(obs)
                action = agent.get_action(state, training=True)
                next_obs, reward, done, _, _ = env.step(np.array([action]))
                next_state = agent.discretization.get_state(next_obs)
                
                agent.update(state, action, reward, next_state, done, next_obs)
                obs = next_obs

# Ejemplo de uso
def main_10k_optimized():
    """Función principal optimizada para 10K episodios"""
    
    # Detectar capacidades GPU
    use_gpu = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    
    print(f"🚀 Iniciando experimento 10K episodios")
    print(f"⚡ GPU habilitada: {use_gpu}")
    
    # Inicializar Ray
    ray.init()
    
    # Configuración optimizada
    discretization = DiscretizationScheme("Media", 25, 25, 25, 25, 10)
    
    # Optimizador híbrido
    optimizer = HybridHyperparameterOptimizer(use_gpu=use_gpu)
    
    # Grid de hiperparámetros reducido pero efectivo
    param_grid = {
        'learning_rate': [0.3, 0.4],
        'discount_factor': [0.98, 0.99], 
        'epsilon': [0.2, 0.3],
        'use_double_q': [True],
        'use_reward_shaping': [True]
    }
    
    # Optimización
    results = optimizer.optimize(param_grid, QLearningAgent, env, discretization)
    
    print(f"✅ Mejores parámetros: {results['best_params']}")
    print(f"📊 Mejor score: {results['best_score']:.2f}")
    
    # Entrenamiento final con mejores parámetros
    best_agent = QLearningAgent(discretization, **results['best_params'])
    
    # Entrenamiento intensivo final
    trainer = QLearningTrainer(env, best_agent, discretization)
    trainer.train(episodes=4000, verbose=True)
    
    # Evaluación final
    evaluator = GPUOptimizedEvaluator(use_gpu=use_gpu)
    final_results = evaluator.batch_evaluate([best_agent], [env], 800)
    
    print(f"🏆 Resultado final: {final_results[0]['mean_reward']:.2f} ± {final_results[0]['std_reward']:.2f}")
    
    ray.shutdown()

if __name__ == "__main__":
    main_10k_optimized()
'''
    
    return implementation_code

def main():
    """Función principal del análisis"""
    
    print("🎯 ANÁLISIS PARA EXPERIMENTO DE ~10,000 EPISODIOS")
    print("Optimización con consideraciones GPU")
    
    # Detectar GPU
    gpu_info = detect_gpu_capabilities()
    
    # Diseñar experimento de 10k episodios
    configs = design_10k_episode_experiment()
    
    # Evaluar beneficios GPU
    evaluate_gpu_benefits()
    
    # Crear configuración optimizada
    optimized_config = create_gpu_optimized_config()
    
    # Guardar configuración
    with open('flan_10k_config.json', 'w') as f:
        json.dump(optimized_config, f, indent=2)
    
    # Guardar código de implementación
    impl_code = generate_implementation_code()
    with open('flan_10k_implementation.py', 'w') as f:
        f.write(impl_code)
    
    # Resumen final
    print("\n" + "="*80)
    print("📋 RESUMEN PARA 10,000 EPISODIOS")
    print("="*80)
    
    print(f"🎯 CONFIGURACIÓN RECOMENDADA:")
    print(f"   • Total episodios: {optimized_config['estimated_episodes']:,}")
    print(f"   • Tiempo estimado: {optimized_config['estimated_time_minutes']:.0f} minutos")
    print(f"   • Estrategia GPU: {optimized_config['gpu_strategy']['type']}")
    print(f"   • Paralelización: {optimized_config['parallelization']['strategy']}")
    
    print(f"\n💡 RECOMENDACIONES GPU:")
    for rec in gpu_info['recommendations']:
        print(f"   • {rec}")
    
    print(f"\n🛠️  ARCHIVOS GENERADOS:")
    print(f"   • flan_10k_config.json - Configuración optimizada para 10K episodios")
    print(f"   • flan_10k_implementation.py - Código con soporte GPU/CPU híbrido")
    
    print(f"\n🚀 PRÓXIMOS PASOS:")
    print(f"   1. Revisar configuración en flan_10k_config.json")
    print(f"   2. Adaptar código actual con implementación híbrida")
    print(f"   3. Ejecutar experimento optimizado (estimado: {optimized_config['estimated_time_minutes']:.0f} min)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main() 