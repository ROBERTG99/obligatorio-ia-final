#!/usr/bin/env python3
"""
Demo del proyecto BORED - Demostración rápida de los agentes implementados
"""

from bored_solution import MinimaxAgent, ExpectimaxAgent, GameEvaluator
from tactix_env import TacTixEnv
from agent import Agent
from trainer_agent_wrapper import TrainerAgentWrapper
from random_agent import RandomTacTixAgent
import time

def demo_single_game():
    """Demuestra una partida individual entre Minimax y Trainer"""
    print("="*60)
    print("DEMO: Partida Individual - Minimax vs Trainer (Hard)")
    print("="*60)
    
    env = TacTixEnv(board_size=6, misere=False)
    minimax = MinimaxAgent(env, max_depth=4, use_alpha_beta=True)
    trainer = TrainerAgentWrapper(env, difficulty=0.9)
    
    evaluator = GameEvaluator(env)
    result = evaluator.play_single_game(minimax, trainer, render=True, verbose=True)
    
    print(f"\nEstadísticas de la partida:")
    print(f"- Ganador: Jugador {result['winner'] + 1}")
    print(f"- Total de movimientos: {result['total_moves']}")
    print(f"- Nodos evaluados por Minimax: {sum(result['nodes_evaluated'])}")
    print(f"- Podas realizadas: {sum(result['pruning_counts'])}")
    print(f"- Tiempo total de evaluación: {sum(result['evaluation_times']):.2f}s")

def demo_alpha_beta_comparison():
    """Compara el rendimiento con y sin alpha-beta pruning"""
    print("\n" + "="*60)
    print("DEMO: Comparación Alpha-Beta Pruning")
    print("="*60)
    
    env = TacTixEnv(board_size=6, misere=False)
    with_pruning = MinimaxAgent(env, max_depth=3, use_alpha_beta=True)
    without_pruning = MinimaxAgent(env, max_depth=3, use_alpha_beta=False)
    random_agent = RandomTacTixAgent(env)
    
    evaluator = GameEvaluator(env)
    
    print("\nPartida CON alpha-beta pruning:")
    start = time.time()
    result1 = evaluator.play_single_game(with_pruning, random_agent)
    time1 = time.time() - start
    nodes1 = sum(result1['nodes_evaluated'])
    
    print(f"- Nodos evaluados: {nodes1}")
    print(f"- Tiempo total: {time1:.2f}s")
    print(f"- Podas realizadas: {sum(result1['pruning_counts'])}")
    
    print("\nPartida SIN alpha-beta pruning:")
    start = time.time()
    result2 = evaluator.play_single_game(without_pruning, random_agent)
    time2 = time.time() - start
    nodes2 = sum(result2['nodes_evaluated'])
    
    print(f"- Nodos evaluados: {nodes2}")
    print(f"- Tiempo total: {time2:.2f}s")
    
    print(f"\nMejora con alpha-beta pruning:")
    print(f"- Reducción de nodos: {((nodes2 - nodes1) / nodes2 * 100):.1f}%")
    print(f"- Aceleración: {time2 / time1:.1f}x")

def demo_minimax_vs_expectimax():
    """Compara Minimax vs Expectimax"""
    print("\n" + "="*60)
    print("DEMO: Minimax vs Expectimax")
    print("="*60)
    
    env = TacTixEnv(board_size=6, misere=False)
    minimax = MinimaxAgent(env, max_depth=3, use_alpha_beta=True)
    expectimax = ExpectimaxAgent(env, max_depth=3, use_alpha_beta=True)
    
    evaluator = GameEvaluator(env)
    
    print("\nPartida 1: Minimax (P1) vs Expectimax (P2)")
    result1 = evaluator.play_single_game(minimax, expectimax, render=False)
    winner1 = "Minimax" if result1['winner'] == 0 else "Expectimax"
    print(f"Ganador: {winner1}")
    
    print("\nPartida 2: Expectimax (P1) vs Minimax (P2)")
    result2 = evaluator.play_single_game(expectimax, minimax, render=False)
    winner2 = "Expectimax" if result2['winner'] == 0 else "Minimax"
    print(f"Ganador: {winner2}")

def demo_heuristic_evaluation():
    """Demuestra la evaluación heurística de un estado"""
    print("\n" + "="*60)
    print("DEMO: Evaluación Heurística")
    print("="*60)
    
    env = TacTixEnv(board_size=6, misere=False)
    agent = MinimaxAgent(env, max_depth=4, use_alpha_beta=True)
    
    # Estado inicial
    obs = env.reset()
    board = obs['board']
    
    print("\nEstado inicial del tablero:")
    env.render()
    
    # Evaluar heurísticas
    pieces = agent.env.board_size ** 2
    moves = len(agent.get_valid_actions(board))
    connectivity = agent._calculate_connectivity(board)
    center = agent._calculate_center_control(board)
    
    print(f"\nHeurísticas del estado inicial:")
    print(f"- Piezas restantes: {pieces}")
    print(f"- Movimientos disponibles: {moves}")
    print(f"- Conectividad: {connectivity}")
    print(f"- Control del centro: {center}")
    
    # Hacer algunos movimientos y reevaluar
    env.step([0, 0, 2, 1])  # Remover fila 0, columnas 0-2
    obs = env.reset()
    obs['board'][0, 0:3] = 0
    board = obs['board']
    
    print("\nEstado después de un movimiento:")
    for row in board:
        print(' '.join('O' if cell else '.' for cell in row))
    
    pieces = agent.env.board_size ** 2 - 3
    moves = len(agent.get_valid_actions(board))
    connectivity = agent._calculate_connectivity(board)
    center = agent._calculate_center_control(board)
    
    print(f"\nHeurísticas después del movimiento:")
    print(f"- Piezas restantes: {pieces}")
    print(f"- Movimientos disponibles: {moves}")
    print(f"- Conectividad: {connectivity}")
    print(f"- Control del centro: {center}")

def main():
    """Ejecuta todas las demos"""
    print("BORED - Board-Oriented Reasoning for Emergent Domination")
    print("Demostración de Funcionalidades\n")
    
    # Demo 1: Partida individual
    demo_single_game()
    
    # Demo 2: Comparación alpha-beta
    demo_alpha_beta_comparison()
    
    # Demo 3: Minimax vs Expectimax
    demo_minimax_vs_expectimax()
    
    # Demo 4: Evaluación heurística
    demo_heuristic_evaluation()
    
    print("\n" + "="*60)
    print("Demo completada!")
    print("Para ejecutar el experimento completo, use: python bored_solution.py")
    print("="*60)

if __name__ == "__main__":
    main() 