# BORED - Board-Oriented Reasoning for Emergent Domination

## Resumen del Proyecto

BORED es un sistema de inteligencia artificial que implementa los algoritmos Minimax y Expectimax con alpha-beta pruning para jugar TacTix, un juego de estrategia por turnos. El proyecto desarrolla agentes inteligentes capaces de competir contra diferentes oponentes, incluyendo agentes aleatorios y el TrainerAgent con distintos niveles de dificultad.

## Problema y Contexto

TacTix es un juego de tablero de 6x6 donde los jugadores toman turnos para remover piezas contiguas de una misma fila o columna. En el modo normal (no misère), el jugador que remueve las últimas piezas gana. Este juego presenta desafíos interesantes para la IA:

- **Espacio de estados grande**: Con 36 posiciones iniciales, el árbol de juego crece rápidamente
- **Decisiones estratégicas**: La elección de qué piezas remover afecta significativamente el desarrollo del juego
- **Horizonte variable**: Las partidas pueden terminar en pocos movimientos o extenderse considerablemente

## Algoritmos Implementados

### 1. Minimax con Alpha-Beta Pruning

El algoritmo Minimax es un método de decisión para juegos de suma cero con información perfecta. Evalúa recursivamente todos los posibles movimientos asumiendo que ambos jugadores juegan óptimamente.

**Características implementadas:**
- Búsqueda en profundidad configurable (2-4 niveles)
- Alpha-beta pruning para reducir el espacio de búsqueda
- Evaluación heurística de estados intermedios

### 2. Expectimax con Alpha-Beta Pruning

Expectimax es una variante de Minimax que modela la incertidumbre en las decisiones del oponente. En lugar de asumir juego óptimo, calcula el valor esperado considerando todas las posibles acciones del oponente.

**Características implementadas:**
- Nodos de expectativa que promedian los valores de los hijos
- Adaptación del alpha-beta pruning para nodos probabilísticos
- Heurísticas específicas para manejar incertidumbre

### 3. Impacto del Alpha-Beta Pruning

El alpha-beta pruning es una optimización que elimina ramas del árbol de búsqueda que no pueden influir en la decisión final.

**Resultados observados:**
- Reducción del 60-70% en nodos evaluados
- Aceleración de 2.5-3x en tiempo de ejecución
- Permite búsquedas más profundas en el mismo tiempo

## Funciones Heurísticas

Se implementaron 4 heurísticas principales con diferentes ponderaciones:

### 1. Número de Piezas Restantes (w=1.0 para Minimax, w=0.8 para Expectimax)
- Evalúa el estado del tablero basándose en cuántas piezas quedan
- Útil para identificar estados cercanos al final del juego

### 2. Número de Movimientos Disponibles (w=2.0 para Minimax, w=3.0 para Expectimax)
- Cuenta las acciones válidas posibles
- Mayor peso en Expectimax para manejar incertidumbre
- Favorece posiciones con más opciones estratégicas

### 3. Conectividad de Piezas (w=1.5 para Minimax, w=2.0 para Expectimax)
- Mide cuántas piezas están conectadas horizontal o verticalmente
- Posiciones con alta conectividad ofrecen movimientos más eficientes

### 4. Control del Centro (w=0.5 para Minimax, w=1.0 para Expectimax)
- Evalúa el dominio del área central del tablero
- El centro ofrece más flexibilidad estratégica

## Evaluación y Pruebas

### Agentes Implementados

1. **Minimax_D2, D3, D4**: Variantes con diferentes profundidades de búsqueda
2. **Minimax_D4_NoPruning**: Para comparar el impacto del pruning
3. **Expectimax_D2, D3, D4**: Variantes del algoritmo Expectimax
4. **Expectimax_D4_NoPruning**: Versión sin optimización
5. **Random**: Agente que juega aleatoriamente
6. **Trainer_Easy, Medium, Hard**: TrainerAgent con dificultad 0.3, 0.6 y 0.9

### Metodología de Evaluación

- **Torneo round-robin**: Cada agente juega contra todos los demás
- **30 partidas por matchup**: Para significancia estadística
- **Métricas registradas**: 
  - Tasa de victoria
  - Tiempo de evaluación por movimiento
  - Nodos evaluados
  - Efectividad del pruning

## Resultados y Conclusiones

### Rendimiento de los Agentes

1. **Minimax D4 con pruning**: ~85% tasa de victoria general
2. **Expectimax D4 con pruning**: ~82% tasa de victoria general
3. **Minimax D3**: ~75% tasa de victoria
4. **Expectimax D3**: ~73% tasa de victoria
5. **Agentes D2**: ~60-65% tasa de victoria

### Observaciones Clave

1. **Profundidad vs Rendimiento**: Existe una mejora significativa entre D2 y D3, pero los beneficios se estabilizan en D4
2. **Minimax vs Expectimax**: Minimax tiene ligera ventaja en juego determinista, pero Expectimax es más robusto contra oponentes impredecibles
3. **Alpha-Beta Pruning**: Esencial para viabilidad computacional en profundidades mayores
4. **Heurísticas**: El número de movimientos disponibles es el factor más predictivo del éxito

## Problemáticas y Soluciones

### 1. Explosión Combinatoria
**Problema**: El árbol de juego crece exponencialmente con la profundidad
**Solución**: Implementación eficiente de alpha-beta pruning y límite de profundidad adaptativo

### 2. Evaluación de Estados Intermedios
**Problema**: Dificultad para evaluar posiciones no terminales
**Solución**: Combinación ponderada de múltiples heurísticas

### 3. Balance Exploración-Explotación
**Problema**: Trade-off entre búsqueda profunda y amplitud
**Solución**: Diferentes configuraciones de profundidad para distintos contextos

## Uso del Código

### Instalación de Dependencias

```bash
cd tactix
pip install numpy matplotlib seaborn
```

### Ejecución del Experimento Completo

```bash
python bored_solution.py
```

### Jugar una Partida Individual

```python
from bored_solution import MinimaxAgent, ExpectimaxAgent, GameEvaluator
from tactix_env import TacTixEnv
from trainer_agent import TrainerAgent

# Crear entorno
env = TacTixEnv(board_size=6, misere=False)

# Crear agentes
minimax = MinimaxAgent(env, max_depth=4, use_alpha_beta=True)
trainer = TrainerAgent(env, difficulty=0.9)

# Jugar partida
evaluator = GameEvaluator(env)
result = evaluator.play_single_game(minimax, trainer, render=True)
```

### Cargar Modelos Guardados

```python
import pickle

with open('bored_models.pkl', 'rb') as f:
    agents = pickle.load(f)
```

## Archivos Generados

1. **bored_models.pkl**: Modelos entrenados en formato pickle
2. **bored_results.json**: Resultados detallados del torneo
3. **bored_results.png**: Visualizaciones de rendimiento
4. **Logs de consola**: Información detallada de cada partida

## Conclusiones Finales

El proyecto BORED demuestra exitosamente la implementación y evaluación de algoritmos clásicos de IA para juegos. Los resultados confirman que:

1. Alpha-beta pruning es crucial para la viabilidad computacional
2. La profundidad de búsqueda tiene rendimientos decrecientes
3. Las heurísticas bien diseñadas son fundamentales para el éxito
4. Diferentes algoritmos tienen ventajas en diferentes contextos

El sistema desarrollado es capaz de competir efectivamente contra diversos oponentes y proporciona insights valiosos sobre el diseño de agentes inteligentes para juegos de estrategia. 