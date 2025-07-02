#!/bin/bash
# Script optimizado para MacBook M1 Pro
# Maximiza el uso de los cores de performance del M1 Pro

echo "======================================"
echo "Ejecutando experimentos en MacBook M1 Pro"
echo "======================================"

# Configurar variables de entorno para optimización
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Función para verificar dependencias
check_dependencies() {
    echo "Verificando dependencias..."
    
    # Verificar Python
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python3 no está instalado"
        exit 1
    fi
    
    # Verificar numpy
    python3 -c "import numpy" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Instalando numpy..."
        pip3 install numpy
    fi
    
    # Verificar matplotlib
    python3 -c "import matplotlib" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Instalando matplotlib..."
        pip3 install matplotlib
    fi
    
    # Verificar gymnasium
    python3 -c "import gymnasium" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Instalando gymnasium..."
        pip3 install gymnasium
    fi
    
    # Verificar seaborn
    python3 -c "import seaborn" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Instalando seaborn..."
        pip3 install seaborn
    fi
    
    echo "✓ Todas las dependencias están instaladas"
}

# Función para ejecutar FLAN
run_flan() {
    echo ""
    echo "======================================"
    echo "Ejecutando Proyecto FLAN"
    echo "======================================"
    
    cd descent-env
    
    # Prueba rápida
    echo "1. Ejecutando pruebas..."
    python3 test_flan.py
    
    # Demo
    echo ""
    echo "2. Ejecutando demo..."
    python3 demo_flan.py
    
    # Experimento completo (reducido para prueba rápida)
    echo ""
    echo "3. Ejecutando experimento completo..."
    # Modificar temporalmente para ejecución rápida
    python3 -c "
import sys
sys.path.append('.')
from flan_qlearning_solution import *

# Configurar para ejecución rápida
env = DescentEnv(render_mode=None)

# Solo una discretización para prueba rápida
discretization = DiscretizationScheme('Media', 10, 10, 10, 10, 5)

print('Entrenando Q-Learning...')
ql_agent = QLearningAgent(discretization, learning_rate=0.1, epsilon=0.2)
trainer = QLearningTrainer(env, ql_agent, discretization)
trainer.train(episodes=50, verbose=True)

print('\\nEntrenando Stochastic Q-Learning...')
stoch_agent = StochasticQLearningAgent(discretization, learning_rate=0.1, epsilon=0.2, sample_size=3)
trainer2 = QLearningTrainer(env, stoch_agent, discretization)
trainer2.train(episodes=50, verbose=True)

print('\\nGuardando modelos...')
import pickle
models = {
    'qlearning': ql_agent,
    'stochastic': stoch_agent,
    'discretization': discretization
}
with open('flan_models_quick.pkl', 'wb') as f:
    pickle.dump(models, f)

print('✓ Modelos guardados en flan_models_quick.pkl')
"
    
    cd ..
}

# Función para ejecutar BORED
run_bored() {
    echo ""
    echo "======================================"
    echo "Ejecutando Proyecto BORED"
    echo "======================================"
    
    cd tactix
    
    # Demo
    echo "1. Ejecutando demo..."
    python3 demo_bored.py
    
    # Experimento rápido
    echo ""
    echo "2. Ejecutando experimento rápido..."
    python3 -c "
import sys
sys.path.append('.')
from bored_solution import *

# Configurar para ejecución rápida
env = TacTixEnv(board_size=6, misere=False)

print('Creando agentes...')
agents = {
    'Minimax_D2': MinimaxAgent(env, max_depth=2, use_alpha_beta=True),
    'Expectimax_D2': ExpectimaxAgent(env, max_depth=2, use_alpha_beta=True),
    'Random': RandomTacTixAgent(env)
}

print('\\nEjecutando mini-torneo...')
evaluator = GameEvaluator(env)
results = evaluator.run_tournament(agents, games_per_matchup=5)

print('\\nResultados:')
for agent_name, stats in results['agent_stats'].items():
    if stats['total_games'] > 0:
        win_rate = stats['wins'] / stats['total_games']
        print(f'{agent_name}: {win_rate:.2%} tasa de victoria')

print('\\nGuardando modelos...')
save_models(agents, 'bored_models_quick.pkl')
"
    
    cd ..
}

# Función principal
main() {
    # Verificar dependencias
    check_dependencies
    
    # Ejecutar FLAN
    run_flan
    
    # Ejecutar BORED
    run_bored
    
    echo ""
    echo "======================================"
    echo "✓ Experimentos completados"
    echo "======================================"
    echo ""
    echo "Archivos generados:"
    echo "- descent-env/flan_models_quick.pkl"
    echo "- descent-env/demo_flan_results.png"
    echo "- tactix/bored_models_quick.pkl"
    echo ""
    echo "Para ejecutar experimentos completos:"
    echo "- cd descent-env && python3 flan_qlearning_solution.py"
    echo "- cd tactix && python3 bored_solution.py"
}

# Ejecutar
main 