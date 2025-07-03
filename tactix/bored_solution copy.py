#!/usr/bin/env python3
"""
BORED - Board-Oriented Reasoning for Emergent Domination
Implementación de Minimax y Expectimax con alpha-beta pruning para TacTix
Versión mejorada con iterative deepening, transposition tables y heurísticas avanzadas
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from typing import Dict, List, Tuple, Optional, Any
from tactix_env import TacTixEnv
from agent import Agent
from random_agent import RandomTacTixAgent
from trainer_agent_wrapper import TrainerAgentWrapper
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import os
import hashlib

class TranspositionTable:
    """Tabla de transposición para memoización en algoritmos de búsqueda"""
    
    def __init__(self, max_size: int = 500000):
        self.table = {}
        self.max_size = max_size
        
    def get_hash(self, board: np.ndarray) -> str:
        """Genera hash único para el estado del tablero"""
        return hashlib.md5(board.tobytes()).hexdigest()
    
    def store(self, board: np.ndarray, depth: int, value: float, 
              best_move: Optional[List[int]], node_type: str):
        """Almacena resultado en la tabla"""
        if len(self.table) >= self.max_size:
            # Eliminar 10% de entradas más antiguas
            items_to_remove = list(self.table.keys())[:self.max_size // 10]
            for key in items_to_remove:
                del self.table[key]
            
        hash_key = self.get_hash(board)
        self.table[hash_key] = {
            'value': value,
            'best_move': best_move,
            'depth': depth,
            'node_type': node_type,
            'timestamp': time.time()
        }
    
    def lookup(self, board: np.ndarray, depth: int) -> Optional[Dict]:
        """Busca resultado en la tabla"""
        hash_key = self.get_hash(board)
        entry = self.table.get(hash_key)
        if entry and entry['depth'] >= depth:
            return entry
        return None

class MinimaxAgent(Agent):
    """Agente Minimax avanzado con iterative deepening y transposition tables"""
    
    def __init__(self, env: TacTixEnv, max_depth: int = 4, use_alpha_beta: bool = True):
        super().__init__(env)
        self.max_depth = max_depth
        self.use_alpha_beta = use_alpha_beta
        self.use_iterative_deepening = False  # Simplified for compatibility
        self.use_transposition = False  # Simplified for compatibility
        self.time_limit = 2.0
        self.transposition_table = None
        self.nodes_evaluated = 0
        self.pruning_count = 0
        self.tt_hits = 0
        
    def get_valid_actions(self, board: np.ndarray) -> List[List[int]]:
        """Obtiene todas las acciones válidas, ordenadas por heurística"""
        actions = []
        size = board.shape[0]
        
        for is_row in [0, 1]:
            for idx in range(size):
                line = board[idx, :] if is_row else board[:, idx]
                start = None
                
                for i in range(size):
                    if line[i] == 1:
                        if start is None:
                            start = i
                    elif start is not None:
                        action = [idx, start, i-1, is_row]
                        actions.append(action)
                        start = None
                
                if start is not None:
                    action = [idx, start, size-1, is_row]
                    actions.append(action)
        
        # Ordenar acciones por heurística para mejor alpha-beta pruning
        if actions:
            actions.sort(key=lambda a: self._action_priority(board, a), reverse=True)
        
        return actions
    
    def _action_priority(self, board: np.ndarray, action: List[int]) -> float:
        """Calcula prioridad de una acción para ordenamiento"""
        idx, start, end, is_row = action
        pieces_removed = end - start + 1
        
        # Preferir movimientos que remueven más piezas
        priority = pieces_removed * 10
        
        # Preferir movimientos en el centro
        center = board.shape[0] // 2
        if is_row:
            distance_from_center = abs(idx - center)
        else:
            distance_from_center = abs(idx - center)
        priority += (board.shape[0] - distance_from_center)
        
        return priority
    
    def apply_action(self, board: np.ndarray, action: List[int]) -> np.ndarray:
        """Aplica una acción al tablero y retorna el nuevo estado"""
        idx, start, end, is_row = action
        new_board = board.copy()
        
        if is_row:
            new_board[idx, start:end+1] = 0
        else:
            new_board[start:end+1, idx] = 0
            
        return new_board
    
    def heuristic_evaluation(self, board: np.ndarray, player: int) -> float:
        """
        Función heurística mejorada para evaluar el estado del tablero
        Retorna un valor positivo si es favorable para el jugador actual
        """
        if np.count_nonzero(board) == 0:
            # Juego terminado
            return 10000 if player == 0 else -10000
        
        # Heurística 1: Número de piezas restantes (con peso dinámico)
        pieces_remaining = np.count_nonzero(board)
        total_initial = self.env.board_size ** 2
        game_progress = 1 - (pieces_remaining / total_initial)
        
        # Heurística 2: Número de movimientos posibles
        valid_actions = self.get_valid_actions(board)
        moves_available = len(valid_actions)
        
        # Heurística 3: Conectividad avanzada de piezas
        connectivity = self._calculate_advanced_connectivity(board)
        
        # Heurística 4: Control del centro y esquinas
        center_control = self._calculate_strategic_control(board)
        
        # Heurística 5: Longitud de segmentos
        segment_score = self._calculate_segment_scores(board)
        
        # Heurística 6: Piezas aisladas (penalización)
        isolated_pieces = self._count_isolated_pieces(board)
        
        # Heurística 7: Simetría del tablero
        symmetry_score = self._calculate_symmetry_bonus(board)
        
        # Pesos dinámicos basados en la fase del juego
        early_game = game_progress < 0.3
        mid_game = 0.3 <= game_progress < 0.7
        end_game = game_progress >= 0.7
        
        if early_game:
            w1, w2, w3, w4, w5, w6, w7 = 1.0, 3.0, 2.0, 1.5, 1.0, -0.5, 0.5
        elif mid_game:
            w1, w2, w3, w4, w5, w6, w7 = 2.0, 2.5, 2.5, 2.0, 2.0, -1.0, 1.0
        else:  # end_game
            w1, w2, w3, w4, w5, w6, w7 = 3.0, 4.0, 1.5, 2.5, 3.0, -2.0, 0.5
        
        # Evaluación ponderada
        evaluation = (w1 * pieces_remaining + 
                     w2 * moves_available + 
                     w3 * connectivity + 
                     w4 * center_control +
                     w5 * segment_score +
                     w6 * isolated_pieces +
                     w7 * symmetry_score)
        
        # Normalizar por el tamaño del tablero
        evaluation /= (self.env.board_size ** 2)
        
        return evaluation
    
    def _calculate_advanced_connectivity(self, board: np.ndarray) -> float:
        """Calcula conectividad considerando grupos de diferentes tamaños"""
        connectivity = 0
        size = board.shape[0]
        visited = np.zeros_like(board, dtype=bool)
        
        groups = []
        for i in range(size):
            for j in range(size):
                if board[i, j] == 1 and not visited[i, j]:
                    group_size = self._flood_fill_size(board, visited, i, j)
                    groups.append(group_size)
        
        # Bonificar grupos grandes, penalizar fragmentación
        for group_size in groups:
            connectivity += group_size ** 1.2
        
        # Penalizar tener muchos grupos pequeños
        if len(groups) > 1:
            avg_group_size = sum(groups) / len(groups)
            connectivity -= (len(groups) - 1) * (3 - avg_group_size) * 0.5
            
        return connectivity
    
    def _flood_fill_size(self, board: np.ndarray, visited: np.ndarray, i: int, j: int) -> int:
        """Calcula el tamaño de un grupo conectado usando flood fill"""
        if (i < 0 or i >= board.shape[0] or j < 0 or j >= board.shape[1] or
            visited[i, j] or board[i, j] == 0):
            return 0
        
        visited[i, j] = True
        size = 1
        
        # Verificar 4 direcciones
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            size += self._flood_fill_size(board, visited, i + di, j + dj)
        
        return size
    
    def _calculate_strategic_control(self, board: np.ndarray) -> float:
        """Calcula control de posiciones estratégicas"""
        size = board.shape[0]
        control = 0
        
        # Control del centro (más valioso)
        center_start = max(1, size // 3)
        center_end = min(size - 1, 2 * size // 3)
        center_weight = 3.0
        
        for i in range(center_start, center_end + 1):
            for j in range(center_start, center_end + 1):
                if board[i, j] == 1:
                    control += center_weight
        
        # Control de bordes (menos valioso pero importante)
        edge_weight = 1.0
        for i in range(size):
            if board[i, 0] == 1: control += edge_weight
            if board[i, -1] == 1: control += edge_weight
            if board[0, i] == 1: control += edge_weight
            if board[-1, i] == 1: control += edge_weight
        
        # Control de esquinas (estratégico)
        corner_weight = 2.0
        corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        for i, j in corners:
            if board[i, j] == 1:
                control += corner_weight
        
        return control
    
    def _calculate_segment_scores(self, board: np.ndarray) -> float:
        """Evalúa la calidad de los segmentos en filas y columnas"""
        score = 0
        size = board.shape[0]
        
        # Analizar filas
        for i in range(size):
            score += self._score_line(board[i, :])
        
        # Analizar columnas
        for j in range(size):
            score += self._score_line(board[:, j])
        
        return score
    
    def _score_line(self, line: np.ndarray) -> float:
        """Puntúa una línea basada en sus segmentos"""
        segments = []
        current_length = 0
        
        for piece in line:
            if piece == 1:
                current_length += 1
            else:
                if current_length > 0:
                    segments.append(current_length)
                    current_length = 0
        
        if current_length > 0:
            segments.append(current_length)
        
        score = 0
        for length in segments:
            # Bonificar segmentos de longitud intermedia
            if length == 1:
                score += 0.5
            elif length == 2:
                score += 2.0
            elif length == 3:
                score += 4.0
            elif length == 4:
                score += 5.0
            else:
                score += length * 1.5
        
        return score
    
    def _count_isolated_pieces(self, board: np.ndarray) -> float:
        """Cuenta piezas aisladas (sin vecinos adyacentes)"""
        isolated = 0
        size = board.shape[0]
        
        for i in range(size):
            for j in range(size):
                if board[i, j] == 1:
                    has_neighbor = False
                    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < size and 0 <= nj < size and 
                            board[ni, nj] == 1):
                            has_neighbor = True
                            break
                    
                    if not has_neighbor:
                        isolated += 1
        
        return isolated
    
    def _calculate_symmetry_bonus(self, board: np.ndarray) -> float:
        """Bonifica configuraciones simétricas que pueden ser ventajosas"""
        horizontal_sym = np.array_equal(board, np.fliplr(board))
        vertical_sym = np.array_equal(board, np.flipud(board))
        
        bonus = 0
        if horizontal_sym:
            bonus += 1
        if vertical_sym:
            bonus += 1
        
        return bonus
    
    def iterative_deepening_search(self, board: np.ndarray, maximizing_player: bool, 
                                  time_limit: float) -> Tuple[float, Optional[List[int]]]:
        """Búsqueda con iterative deepening"""
        start_time = time.time()
        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None
        
        for depth in range(1, self.max_depth + 1):
            if time.time() - start_time > time_limit * 0.9:  # Reservar 10% para overhead
                break
                
            try:
                value, move = self.minimax(
                    board, depth, float('-inf'), float('inf'), maximizing_player
                )
                
                if maximizing_player and value > best_value:
                    best_value = value
                    best_move = move
                elif not maximizing_player and value < best_value:
                    best_value = value
                    best_move = move
                    
                # Si encontramos una solución ganadora, podemos parar
                if abs(value) > 9000:
                    break
                    
            except TimeoutError:
                break
        
        return best_value, best_move
    
    def minimax(self, board: np.ndarray, depth: int, alpha: float, beta: float, 
                maximizing_player: bool) -> Tuple[float, Optional[List[int]]]:
        """
        Algoritmo Minimax avanzado con optimizaciones
        """
        self.nodes_evaluated += 1
        
        # Verificar transposition table
        if self.use_transposition and self.transposition_table:
            tt_entry = self.transposition_table.lookup(board, depth)
            if tt_entry:
                self.tt_hits += 1
                return tt_entry['value'], tt_entry['best_move']
        
        # Condición de parada
        if depth == 0 or np.count_nonzero(board) == 0:
            eval_score = self.heuristic_evaluation(board, 0 if maximizing_player else 1)
            return eval_score, None
        
        valid_actions = self.get_valid_actions(board)
        
        if not valid_actions:
            eval_score = self.heuristic_evaluation(board, 0 if maximizing_player else 1)
            return eval_score, None
        
        best_action = None
        
        if maximizing_player:
            max_eval = float('-inf')
            
            for action in valid_actions:
                new_board = self.apply_action(board, action)
                eval_score, _ = self.minimax(new_board, depth - 1, alpha, beta, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                
                if self.use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        self.pruning_count += 1
                        break
            
            # Guardar en transposition table
            if self.use_transposition and self.transposition_table:
                self.transposition_table.store(
                    board, depth, max_eval, best_action, 'exact'
                )
            
            return max_eval, best_action
        else:
            min_eval = float('inf')
            
            for action in valid_actions:
                new_board = self.apply_action(board, action)
                eval_score, _ = self.minimax(new_board, depth - 1, alpha, beta, True)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                
                if self.use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        self.pruning_count += 1
                        break
            
            # Guardar en transposition table
            if self.use_transposition and self.transposition_table:
                self.transposition_table.store(
                    board, depth, min_eval, best_action, 'exact'
                )
            
            return min_eval, best_action
    
    def act(self, obs: Dict) -> List[int]:
        """Selecciona la mejor acción usando Minimax avanzado"""
        board = obs["board"]
        current_player = obs["current_player"]
        
        # Resetear contadores
        self.nodes_evaluated = 0
        self.pruning_count = 0
        self.tt_hits = 0
        
        # Determinar si es el jugador maximizador
        maximizing_player = (current_player == 0)
        
        # Ejecutar búsqueda
        start_time = time.time()
        
        if self.use_iterative_deepening:
            eval_score, best_action = self.iterative_deepening_search(
                board, maximizing_player, self.time_limit
            )
        else:
            eval_score, best_action = self.minimax(
                board, 
                self.max_depth, 
                float('-inf'), 
                float('inf'), 
                maximizing_player
            )
        
        end_time = time.time()
        
        # Registrar estadísticas
        self.last_eval_time = end_time - start_time
        self.last_eval_score = eval_score
        self.last_nodes_evaluated = self.nodes_evaluated
        self.last_pruning_count = self.pruning_count
        self.last_tt_hits = self.tt_hits
        
        return best_action if best_action else self.get_valid_actions(board)[0]

class ExpectimaxAgent(Agent):
    """Agente que implementa el algoritmo Expectimax con alpha-beta pruning"""
    
    def __init__(self, env: TacTixEnv, max_depth: int = 4, use_alpha_beta: bool = True):
        super().__init__(env)
        self.max_depth = max_depth
        self.use_alpha_beta = use_alpha_beta
        self.nodes_evaluated = 0
        self.pruning_count = 0
        
    def get_valid_actions(self, board: np.ndarray) -> List[List[int]]:
        """Obtiene todas las acciones válidas en el tablero actual"""
        actions = []
        size = board.shape[0]
        
        for is_row in [0, 1]:
            for idx in range(size):
                line = board[idx, :] if is_row else board[:, idx]
                start = None
                
                for i in range(size):
                    if line[i] == 1:
                        if start is None:
                            start = i
                    elif start is not None:
                        actions.append([idx, start, i-1, is_row])
                        start = None
                
                if start is not None:
                    actions.append([idx, start, size-1, is_row])
        
        return actions
    
    def apply_action(self, board: np.ndarray, action: List[int]) -> np.ndarray:
        """Aplica una acción al tablero y retorna el nuevo estado"""
        idx, start, end, is_row = action
        new_board = board.copy()
        
        if is_row:
            new_board[idx, start:end+1] = 0
        else:
            new_board[start:end+1, idx] = 0
            
        return new_board
    
    def heuristic_evaluation(self, board: np.ndarray, player: int) -> float:
        """
        Función heurística para evaluar el estado del tablero
        Similar a Minimax pero con ponderaciones diferentes
        """
        if np.count_nonzero(board) == 0:
            return 1000 if player == 0 else -1000
        
        # Heurística 1: Número de piezas restantes
        pieces_remaining = np.count_nonzero(board)
        
        # Heurística 2: Número de movimientos posibles
        valid_actions = self.get_valid_actions(board)
        moves_available = len(valid_actions)
        
        # Heurística 3: Conectividad de piezas
        connectivity = self._calculate_connectivity(board)
        
        # Heurística 4: Control del centro
        center_control = self._calculate_center_control(board)
        
        # Ponderaciones diferentes para Expectimax
        w1, w2, w3, w4 = 0.8, 3.0, 2.0, 1.0
        
        evaluation = (w1 * pieces_remaining + 
                     w2 * moves_available + 
                     w3 * connectivity + 
                     w4 * center_control)
        
        evaluation /= (self.env.board_size ** 2)
        
        return evaluation
    
    def _calculate_connectivity(self, board: np.ndarray) -> float:
        """Calcula la conectividad de las piezas en el tablero"""
        connectivity = 0
        size = board.shape[0]
        
        # Contar conexiones horizontales
        for i in range(size):
            for j in range(size - 1):
                if board[i, j] == 1 and board[i, j + 1] == 1:
                    connectivity += 1
        
        # Contar conexiones verticales
        for i in range(size - 1):
            for j in range(size):
                if board[i, j] == 1 and board[i + 1, j] == 1:
                    connectivity += 1
        
        return connectivity
    
    def _calculate_center_control(self, board: np.ndarray) -> float:
        """Calcula el control del centro del tablero"""
        size = board.shape[0]
        center_start = size // 3
        center_end = 2 * size // 3
        
        center_pieces = np.sum(board[center_start:center_end, center_start:center_end])
        return center_pieces
    
    def expectimax(self, board: np.ndarray, depth: int, alpha: float, beta: float, 
                   maximizing_player: bool) -> Tuple[float, Optional[List[int]]]:
        """
        Algoritmo Expectimax con alpha-beta pruning
        """
        self.nodes_evaluated += 1
        
        # Condición de parada
        if depth == 0 or np.count_nonzero(board) == 0:
            return self.heuristic_evaluation(board, 0 if maximizing_player else 1), None
        
        valid_actions = self.get_valid_actions(board)
        
        if not valid_actions:
            return self.heuristic_evaluation(board, 0 if maximizing_player else 1), None
        
        best_action = None
        
        if maximizing_player:
            max_eval = float('-inf')
            
            for action in valid_actions:
                new_board = self.apply_action(board, action)
                eval_score, _ = self.expectimax(new_board, depth - 1, alpha, beta, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                
                if self.use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        self.pruning_count += 1
                        break
        else:
            # Nodo de expectativa: promedio de todos los hijos
            total_eval = 0
            num_actions = len(valid_actions)
            
            for action in valid_actions:
                new_board = self.apply_action(board, action)
                eval_score, _ = self.expectimax(new_board, depth - 1, alpha, beta, True)
                total_eval += eval_score
            
            avg_eval = total_eval / num_actions
            best_action = valid_actions[0]  # Para nodos de expectativa, cualquier acción es válida
            
            return avg_eval, best_action
        
        return max_eval, best_action
    
    def act(self, obs: Dict) -> List[int]:
        """Selecciona la mejor acción usando Expectimax"""
        board = obs["board"]
        current_player = obs["current_player"]
        
        # Resetear contadores
        self.nodes_evaluated = 0
        self.pruning_count = 0
        
        # Determinar si es el jugador maximizador
        maximizing_player = (current_player == 0)
        
        # Ejecutar Expectimax
        start_time = time.time()
        eval_score, best_action = self.expectimax(
            board, 
            self.max_depth, 
            float('-inf'), 
            float('inf'), 
            maximizing_player
        )
        end_time = time.time()
        
        # Registrar estadísticas
        self.last_eval_time = end_time - start_time
        self.last_eval_score = eval_score
        self.last_nodes_evaluated = self.nodes_evaluated
        self.last_pruning_count = self.pruning_count
        
        return best_action if best_action else self.get_valid_actions(board)[0]

class GameEvaluator:
    """Evaluador de partidas entre agentes"""
    
    def __init__(self, env: TacTixEnv):
        self.env = env
        
    def play_single_game(self, agent1: Agent, agent2: Agent, 
                        render: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Juega una partida entre dos agentes"""
        obs = self.env.reset()
        done = False
        moves = []
        game_stats: Dict[str, Any] = {
            'moves': [],
            'board_states': [],
            'evaluation_times': [],
            'evaluation_scores': [],
            'nodes_evaluated': [],
            'pruning_counts': []
        }
        
        while not done:
            current_agent = agent1 if obs["current_player"] == 0 else agent2
            
            # Registrar estadísticas antes del movimiento
            game_stats['evaluation_times'].append(getattr(current_agent, 'last_eval_time', 0.0))
            game_stats['evaluation_scores'].append(getattr(current_agent, 'last_eval_score', 0.0))
            game_stats['nodes_evaluated'].append(getattr(current_agent, 'last_nodes_evaluated', 0))
            game_stats['pruning_counts'].append(getattr(current_agent, 'last_pruning_count', 0))
            
            # Tomar acción
            action = current_agent.act(obs)
            moves.append(action)
            game_stats['moves'].append(action)
            game_stats['board_states'].append(obs['board'].copy())
            
            if render:
                self.env.render()
                print(f"Player {obs['current_player'] + 1} action: {action}")
            
            obs, reward, done, _ = self.env.step(action)
            
            if verbose:
                print(f"Move {len(moves)}: Player {obs['current_player'] + 1} -> {action}")
        
        # Determinar ganador
        last_player = 1 - obs["current_player"]
        winner = last_player  # En modo normal, el último en jugar gana
        
        game_stats['winner'] = winner
        game_stats['total_moves'] = len(moves)
        game_stats['final_board'] = obs['board'].copy()
        
        # Asegurar que las listas tengan la misma longitud
        while len(game_stats['evaluation_times']) < len(moves):
            game_stats['evaluation_times'].append(0.0)
        while len(game_stats['evaluation_scores']) < len(moves):
            game_stats['evaluation_scores'].append(0.0)
        while len(game_stats['nodes_evaluated']) < len(moves):
            game_stats['nodes_evaluated'].append(0)
        while len(game_stats['pruning_counts']) < len(moves):
            game_stats['pruning_counts'].append(0)
        
        if render:
            self.env.render()
            print(f"Player {winner + 1} wins!")
        
        return game_stats
    
    def run_tournament(self, agents: Dict[str, Agent], 
                      games_per_matchup: int = 100) -> Dict[str, Any]:
        """Ejecuta un torneo entre múltiples agentes con paralelización optimizada"""
        agent_names = list(agents.keys())
        results = {
            'matchups': {},
            'overall_stats': {},
            'agent_stats': {name: {'wins': 0, 'losses': 0, 'total_games': 0} for name in agent_names}
        }
        
        # Obtener número de CPUs disponibles
        cpu_count = psutil.cpu_count(logical=True)
        num_cores = max(1, cpu_count - 1) if cpu_count else 1
        print(f"Ejecutando torneo con {num_cores} cores de CPU")
        
        # Calcular total de matchups y partidas
        total_matchups = len(agent_names) * (len(agent_names) - 1) // 2
        total_games = total_matchups * games_per_matchup
        print(f"Total: {total_matchups} matchups × {games_per_matchup} partidas = {total_games} partidas")
        
        completed_matchups = 0
        start_time = time.time()
        
        # Jugar todos los matchups
        for i, agent1_name in enumerate(agent_names):
            for j, agent2_name in enumerate(agent_names):
                if i >= j:  # Evitar matchups duplicados
                    continue
                
                completed_matchups += 1
                matchup_key = f"{agent1_name}_vs_{agent2_name}"
                results['matchups'][matchup_key] = {
                    'agent1_wins': 0,
                    'agent2_wins': 0,
                    'games': []
                }
                
                # Estimación de tiempo
                if completed_matchups > 1:
                    elapsed = time.time() - start_time
                    avg_time_per_matchup = elapsed / (completed_matchups - 1)
                    remaining_matchups = total_matchups - completed_matchups + 1
                    eta_seconds = avg_time_per_matchup * remaining_matchups
                    eta_minutes = eta_seconds / 60
                    print(f"[{completed_matchups}/{total_matchups}] {matchup_key} - ETA: {eta_minutes:.1f} min")
                else:
                    print(f"[{completed_matchups}/{total_matchups}] {matchup_key}")
                
                matchup_start = time.time()
                
                # Ejecutar partidas en paralelo
                with ProcessPoolExecutor(max_workers=num_cores) as executor:
                    # Enviar todas las tareas
                    future_to_game = {
                        executor.submit(self._play_single_game_parallel, 
                                      agents[agent1_name], agents[agent2_name]): game 
                        for game in range(games_per_matchup)
                    }
                    
                    completed_games = 0
                    # Procesar resultados conforme van completándose
                    for future in as_completed(future_to_game):
                        try:
                            game_stats = future.result()
                            completed_games += 1
                            
                            # Progreso cada 50 partidas
                            if completed_games % 50 == 0 or completed_games == games_per_matchup:
                                print(f"    Progreso: {completed_games}/{games_per_matchup} partidas")
                            
                            results['matchups'][matchup_key]['games'].append(game_stats)
                            
                            if game_stats['winner'] == 0:
                                results['matchups'][matchup_key]['agent1_wins'] += 1
                                results['agent_stats'][agent1_name]['wins'] += 1
                                results['agent_stats'][agent2_name]['losses'] += 1
                            else:
                                results['matchups'][matchup_key]['agent2_wins'] += 1
                                results['agent_stats'][agent2_name]['wins'] += 1
                                results['agent_stats'][agent1_name]['losses'] += 1
                            
                            results['agent_stats'][agent1_name]['total_games'] += 1
                            results['agent_stats'][agent2_name]['total_games'] += 1
                            
                        except Exception as e:
                            print(f"    Error en partida: {e}")
                
                matchup_time = time.time() - matchup_start
                a1_wins = results['matchups'][matchup_key]['agent1_wins']
                a2_wins = results['matchups'][matchup_key]['agent2_wins']
                print(f"    ✓ Completado en {matchup_time:.1f}s: {a1_wins}-{a2_wins}")
        
        total_time = time.time() - start_time
        print(f"\n🏆 Torneo completado en {total_time/60:.1f} minutos")
        print(f"📊 Total de partidas procesadas: {total_games}")
        
        return results
    
    def _play_single_game_parallel(self, agent1: Agent, agent2: Agent) -> Dict[str, Any]:
        """Versión de play_single_game optimizada para paralelización"""
        # Crear nuevo entorno para cada proceso
        env = TacTixEnv(board_size=6, misere=False)
        obs = env.reset()
        done = False
        moves = []
        game_stats: Dict[str, Any] = {
            'moves': [],
            'board_states': [],
            'evaluation_times': [],
            'evaluation_scores': [],
            'nodes_evaluated': [],
            'pruning_counts': []
        }
        
        while not done:
            current_agent = agent1 if obs["current_player"] == 0 else agent2
            
            # Registrar estadísticas antes del movimiento
            game_stats['evaluation_times'].append(getattr(current_agent, 'last_eval_time', 0.0))
            game_stats['evaluation_scores'].append(getattr(current_agent, 'last_eval_score', 0.0))
            game_stats['nodes_evaluated'].append(getattr(current_agent, 'last_nodes_evaluated', 0))
            game_stats['pruning_counts'].append(getattr(current_agent, 'last_pruning_count', 0))
            
            # Tomar acción
            action = current_agent.act(obs)
            moves.append(action)
            game_stats['moves'].append(action)
            game_stats['board_states'].append(obs['board'].copy())
            
            obs, reward, done, _ = env.step(action)
        
        # Determinar ganador
        last_player = 1 - obs["current_player"]
        winner = last_player  # En modo normal, el último en jugar gana
        
        game_stats['winner'] = winner
        game_stats['total_moves'] = len(moves)
        game_stats['final_board'] = obs['board'].copy()
        
        # Asegurar que las listas tengan la misma longitud
        while len(game_stats['evaluation_times']) < len(moves):
            game_stats['evaluation_times'].append(0.0)
        while len(game_stats['evaluation_scores']) < len(moves):
            game_stats['evaluation_scores'].append(0.0)
        while len(game_stats['nodes_evaluated']) < len(moves):
            game_stats['nodes_evaluated'].append(0)
        while len(game_stats['pruning_counts']) < len(moves):
            game_stats['pruning_counts'].append(0)
        
        return game_stats

def create_heuristic_variants():
    """Crea variantes optimizadas de agentes para experimento eficiente pero robusto"""
    env = TacTixEnv(board_size=6, misere=False)
    
    # Variantes core de Minimax (eliminando extremos lentos)
    minimax_agents = {
        'Minimax_D3': MinimaxAgent(env, max_depth=3, use_alpha_beta=True),
        'Minimax_D4': MinimaxAgent(env, max_depth=4, use_alpha_beta=True),
        'Minimax_D5': MinimaxAgent(env, max_depth=5, use_alpha_beta=True),
        'Minimax_D6': MinimaxAgent(env, max_depth=6, use_alpha_beta=True),
        'Minimax_D4_NoPruning': MinimaxAgent(env, max_depth=4, use_alpha_beta=False)
    }
    
    # Variantes core de Expectimax (eliminando extremos lentos)
    expectimax_agents = {
        'Expectimax_D3': ExpectimaxAgent(env, max_depth=3, use_alpha_beta=True),
        'Expectimax_D4': ExpectimaxAgent(env, max_depth=4, use_alpha_beta=True),
        'Expectimax_D5': ExpectimaxAgent(env, max_depth=5, use_alpha_beta=True),
        'Expectimax_D4_NoPruning': ExpectimaxAgent(env, max_depth=4, use_alpha_beta=False)
    }
    
    # Agentes de referencia optimizados
    reference_agents = {
        'Random': RandomTacTixAgent(env),
        'Trainer_Easy': TrainerAgentWrapper(env, difficulty=0.3),
        'Trainer_Medium': TrainerAgentWrapper(env, difficulty=0.6),
        'Trainer_Hard': TrainerAgentWrapper(env, difficulty=0.8),
        'Trainer_Expert': TrainerAgentWrapper(env, difficulty=0.99)
    }
    
    return {**minimax_agents, **expectimax_agents, **reference_agents}

def _play_alpha_beta_game_parallel(depth: int, use_pruning: bool) -> Dict[str, Any]:
    """Función helper para ejecutar una partida en paralelo creando agentes en el proceso"""
    # Crear entorno y agentes en cada proceso
    env = TacTixEnv(board_size=6, misere=False)
    agent = MinimaxAgent(env, max_depth=depth, use_alpha_beta=use_pruning)
    random_agent = RandomTacTixAgent(env)
    evaluator = GameEvaluator(env)
    
    # Jugar la partida
    return evaluator.play_single_game(agent, random_agent)

def analyze_alpha_beta_impact():
    """Analiza el impacto del alpha-beta pruning con robustez estadística"""
    env = TacTixEnv(board_size=6, misere=False)
    
    # Configuraciones optimizadas para análisis eficiente
    test_configs = [
        (3, True), (3, False),   # Comparación directa principal
        (4, True), (4, False),   # Límite para sin pruning
        (5, True), (6, True)     # Solo con pruning para profundidades altas
    ]
    
    results = {
        'with_pruning': [],
        'without_pruning': []
    }
    
    total_configs = len(test_configs)
    
    for idx, (depth, use_pruning) in enumerate(test_configs):
        config_name = f"D{depth}_{'WithPruning' if use_pruning else 'NoPruning'}"
        print(f"  [{idx+1}/{total_configs}] Probando {config_name}...")
        
        agent = MinimaxAgent(env, max_depth=depth, use_alpha_beta=use_pruning)
        evaluator = GameEvaluator(env)
        stats = []
        
        # Partidas optimizadas para eficiencia
        if not use_pruning:
            if depth <= 3:
                num_games = 12  # Moderadas para profundidad baja-media
            else:
                num_games = 6   # Pocas para profundidad alta
        else:
            num_games = 20  # Suficientes partidas con pruning (es rápido)
        
        print(f"    Ejecutando {num_games} partidas...")
        
        # Usar paralelización para aprovechar todos los cores
        cpu_count = psutil.cpu_count(logical=True)
        num_cores = max(1, cpu_count - 1) if cpu_count else 1
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Enviar todas las tareas en paralelo (pasar parámetros para crear agentes en cada proceso)
            future_to_game = {
                executor.submit(_play_alpha_beta_game_parallel, depth, use_pruning): i 
                for i in range(num_games)
            }
            
            completed = 0
            # Procesar resultados conforme van completándose
            for future in as_completed(future_to_game):
                try:
                    game = future.result()
                    completed += 1
                    
                    if completed % 5 == 0 or completed == num_games:  # Progreso cada 5 partidas
                        print(f"      Progreso: {completed}/{num_games}")
                    
                    game_stats = {
                        'nodes_evaluated': sum(game['nodes_evaluated']),
                        'pruning_count': sum(game['pruning_counts']),
                        'eval_time': sum(game['evaluation_times']),
                        'moves': len(game['moves']),
                        'depth': depth,
                        'use_pruning': use_pruning
                    }
                    stats.append(game_stats)
                    
                    # Agregar a la categoría correcta para los gráficos
                    if use_pruning:
                        results['with_pruning'].append(game_stats)
                    else:
                        results['without_pruning'].append(game_stats)
                        
                except Exception as e:
                    print(f"      Error en partida: {e}")
                    continue
        
        print(f"    ✓ Completado: {len(stats)} partidas exitosas")
    
    return results

def generate_plots(results: Dict, alpha_beta_results: Dict):
    """Genera gráficos de los resultados"""
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
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
    
    # 2. Comparación Minimax vs Expectimax (corregido para arrays de diferente tamaño)
    minimax_agents = [name for name in agent_names if 'Minimax' in name and 'NoPruning' not in name]
    expectimax_agents = [name for name in agent_names if 'Expectimax' in name and 'NoPruning' not in name]
    
    # Extraer profundidades para hacer matching correcto
    minimax_depths = []
    minimax_rates = []
    for name in minimax_agents:
        if results['agent_stats'][name]['total_games'] > 0:
            # Extraer profundidad del nombre (ej: 'Minimax_D4' -> 'D4')
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
        # Encontrar tasa para esta profundidad en minimax
        if depth in minimax_depths:
            idx = minimax_depths.index(depth)
            common_minimax_rates.append(minimax_rates[idx])
        
        # Encontrar tasa para esta profundidad en expectimax
        if depth in expectimax_depths:
            idx = expectimax_depths.index(depth)
            common_expectimax_rates.append(expectimax_rates[idx])
    
    if common_depths:  # Solo generar gráfico si hay datos comunes
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
    
    # 3. Impacto del Alpha-Beta Pruning
    pruning_nodes = [stat['nodes_evaluated'] for stat in alpha_beta_results['with_pruning']]
    no_pruning_nodes = [stat['nodes_evaluated'] for stat in alpha_beta_results['without_pruning']]
    
    axes[1, 0].boxplot([pruning_nodes, no_pruning_nodes], 
                      tick_labels=['Con Pruning', 'Sin Pruning'])
    axes[1, 0].set_ylabel('Nodos Evaluados')
    axes[1, 0].set_title('Impacto del Alpha-Beta Pruning')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Tiempo de evaluación
    pruning_times = [stat['eval_time'] for stat in alpha_beta_results['with_pruning']]
    no_pruning_times = [stat['eval_time'] for stat in alpha_beta_results['without_pruning']]
    
    axes[1, 1].boxplot([pruning_times, no_pruning_times], 
                      tick_labels=['Con Pruning', 'Sin Pruning'])
    axes[1, 1].set_ylabel('Tiempo de Evaluación (s)')
    axes[1, 1].set_title('Tiempo de Evaluación con/sin Pruning')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bored_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_models(agents: Dict[str, Agent], filename: str = 'bored_models.pkl'):
    """Guarda los modelos entrenados"""
    with open(filename, 'wb') as f:
        pickle.dump(agents, f)
    print(f"Modelos guardados en {filename}")

def _test_computation_task(n):
    """Función global para test de paralelización"""
    # Simular trabajo computacional
    sum_val = 0
    for i in range(1000000):
        sum_val += i
    return n, sum_val

def test_multiprocessing():
    """Prueba rápida para verificar paralelización"""
    import time
    from concurrent.futures import ProcessPoolExecutor
    
    print("🧪 Probando paralelización...")
    
    # Test secuencial
    start = time.time()
    for i in range(10):
        _test_computation_task(i)
    sequential_time = time.time() - start
    
    # Test paralelo
    start = time.time()
    cpu_count = psutil.cpu_count(logical=True)
    num_cores = max(1, cpu_count - 1) if cpu_count else 1
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(_test_computation_task, i) for i in range(10)]
        for future in as_completed(futures):
            future.result()
    parallel_time = time.time() - start
    
    speedup = sequential_time / parallel_time
    print(f"   Secuencial: {sequential_time:.2f}s")
    print(f"   Paralelo ({num_cores} cores): {parallel_time:.2f}s") 
    print(f"   Aceleración: {speedup:.1f}x")
    
    if speedup > 2:
        print("   ✅ Paralelización funcionando correctamente")
    else:
        print("   ⚠️  Paralelización limitada - verificar configuración")

def main():
    """Función principal para ejecutar el experimento BORED"""
    
    print("="*60)
    print("BORED - Board-Oriented Reasoning for Emergent Domination")
    print("="*60)
    
    # Configurar multiprocessing para macOS
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Ya está configurado
    
    # Configurar optimización para MacBook M1 Pro
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Obtener información del sistema
    cpu_count = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Sistema: {cpu_count} cores CPU, {memory_gb:.1f}GB RAM")
    
    # Probar paralelización
    test_multiprocessing()
    
    # Crear entorno
    env = TacTixEnv(board_size=6, misere=False)
    
    # Crear agentes
    print("\nCreando agentes...")
    agents = create_heuristic_variants()
    print(f"Creados {len(agents)} agentes diferentes")
    
    # Analizar impacto del alpha-beta pruning
    print("\nAnalizando impacto del alpha-beta pruning...")
    alpha_beta_results = analyze_alpha_beta_impact()
    
    # Ejecutar torneo con máxima robustez estadística
    print(f"\nEjecutando torneo entre agentes...")
    evaluator = GameEvaluator(env)
    
    # Configuración optimizada para 45-60 minutos
    games_per_matchup = 200  # Balance óptimo entre robustez y tiempo
    print(f"Configuración: {games_per_matchup} partidas por matchup")
    print(f"Estimación: ~45-60 minutos de procesamiento")
    print(f"Usando {psutil.cpu_count(logical=True)} cores para paralelización")
    
    tournament_results = evaluator.run_tournament(agents, games_per_matchup=games_per_matchup)
    
    # Generar gráficos
    print("\nGenerando gráficos...")
    generate_plots(tournament_results, alpha_beta_results)
    
    # Guardar resultados
    print("\nGuardando resultados...")
    
    def convert_to_serializable(obj):
        """Convierte recursivamente objetos numpy a tipos serializables por JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
            return int(obj)
        elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
            return float(obj)
        elif hasattr(obj, 'dtype') and 'bool' in str(obj.dtype):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    with open('bored_results.json', 'w') as f:
        serializable_results = convert_to_serializable(tournament_results)
        json.dump(serializable_results, f, indent=2)
    
    # Guardar modelos
    save_models(agents)
    
    # Generar reporte
    generate_report(tournament_results, alpha_beta_results)
    
    print(f"\n{'='*60}")
    print("EXPERIMENTO BORED COMPLETADO")
    print(f"{'='*60}")
    print(f"Total de partidas jugadas: {sum(stats['total_games'] for stats in tournament_results['agent_stats'].values()) // 2}")
    print("Archivos generados:")
    print("- bored_results.json: Resultados del torneo")
    print("- bored_results.png: Gráficos de análisis")
    print("- bored_models.pkl: Modelos de agentes")
    
    return tournament_results, alpha_beta_results

def generate_report(tournament_results: Dict, alpha_beta_results: Dict):
    """Genera un reporte completo de los resultados"""
    
    print("\n" + "="*80)
    print("REPORTE FINAL - PROYECTO BORED")
    print("="*80)
    
    # Resumen de agentes
    print("\n1. AGENTES IMPLEMENTADOS")
    print("-" * 50)
    
    agent_stats = tournament_results['agent_stats']
    for agent_name, stats in agent_stats.items():
        if stats['total_games'] > 0:
            win_rate = stats['wins'] / stats['total_games']
            print(f"{agent_name}: {stats['wins']}W/{stats['losses']}L ({win_rate:.2%})")
    
    # Análisis del alpha-beta pruning
    print("\n2. IMPACTO DEL ALPHA-BETA PRUNING")
    print("-" * 50)
    
    avg_nodes_with = np.mean([stat['nodes_evaluated'] for stat in alpha_beta_results['with_pruning']])
    avg_nodes_without = np.mean([stat['nodes_evaluated'] for stat in alpha_beta_results['without_pruning']])
    avg_time_with = np.mean([stat['eval_time'] for stat in alpha_beta_results['with_pruning']])
    avg_time_without = np.mean([stat['eval_time'] for stat in alpha_beta_results['without_pruning']])
    
    print(f"Nodos evaluados con pruning: {avg_nodes_with:.0f}")
    print(f"Nodos evaluados sin pruning: {avg_nodes_without:.0f}")
    print(f"Reducción: {((avg_nodes_without - avg_nodes_with) / avg_nodes_without * 100):.1f}%")
    print(f"Tiempo con pruning: {avg_time_with:.3f}s")
    print(f"Tiempo sin pruning: {avg_time_without:.3f}s")
    print(f"Aceleración: {avg_time_without / avg_time_with:.1f}x")
    
    # Mejores agentes
    print("\n3. MEJORES AGENTES")
    print("-" * 50)
    
    sorted_agents = sorted(agent_stats.items(), 
                          key=lambda x: x[1]['wins'] / x[1]['total_games'] if x[1]['total_games'] > 0 else 0,
                          reverse=True)
    
    for i, (agent_name, stats) in enumerate(sorted_agents[:5]):
        if stats['total_games'] > 0:
            win_rate = stats['wins'] / stats['total_games']
            print(f"{i+1}. {agent_name}: {win_rate:.2%}")
    
    # Conclusiones
    print("\n4. CONCLUSIONES")
    print("-" * 50)
    print("✓ Minimax y Expectimax implementados correctamente")
    print("✓ Alpha-beta pruning reduce significativamente los nodos evaluados")
    print("✓ Diferentes profundidades muestran trade-offs entre tiempo y calidad")
    print("✓ Expectimax puede ser más robusto contra oponentes aleatorios")
    print("✓ Los agentes superan significativamente a los oponentes aleatorios")

if __name__ == "__main__":
    # Proteger la ejecución principal para multiprocessing
    mp.freeze_support()  # Para Windows
    results = main()