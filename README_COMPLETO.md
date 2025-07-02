# Obligatorio IA - Marzo 2025

## Guía Completa de Ejecución y Evaluación

Este documento proporciona una guía completa para ejecutar y evaluar los dos proyectos implementados: FLAN (Flight-Level Adjustment Network) y BORED (Board-Oriented Reasoning for Emergent Domination).

## Estructura del Proyecto

```
Obligatorio IA - Marzo 2025 (1)/
├── descent-env/                    # Proyecto FLAN
│   ├── descent_env.py             # Entorno del simulador de vuelo
│   ├── flan_qlearning_solution.py # Implementación completa de Q-Learning
│   ├── demo_flan.py              # Script de demostración
│   ├── test_flan.py              # Tests unitarios
│   ├── README_FLAN.md            # Documentación del proyecto FLAN
│   └── RESUMEN_FLAN.md           # Resumen ejecutivo FLAN
│
├── tactix/                        # Proyecto BORED
│   ├── tactix_env.py             # Entorno del juego TacTix
│   ├── bored_solution.py         # Implementación de Minimax/Expectimax
│   ├── demo_bored.py             # Script de demostración
│   ├── README_BORED.md           # Documentación del proyecto BORED
│   └── trainer_agent.py          # Agente oponente configurable
│
└── README_COMPLETO.md            # Esta guía

```

## Instalación de Dependencias

### Requisitos Generales

```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias comunes
pip install numpy matplotlib seaborn pandas gymnasium tqdm
```

### Dependencias Específicas por Proyecto

#### FLAN
```bash
cd descent-env
pip install -r requirements.txt  # Si existe
# O instalar manualmente:
pip install numpy matplotlib gymnasium pandas seaborn tqdm
```

#### BORED
```bash
cd tactix
pip install numpy matplotlib seaborn
```

## Proyecto 1: FLAN (Flight-Level Adjustment Network)

### Descripción
FLAN implementa Q-Learning y Q-Learning Estocástico para controlar la velocidad vertical de una aeronave experimental, manteniendo la altitud objetivo y realizando descensos suaves.

### Ejecución Rápida

```bash
cd descent-env

# 1. Ejecutar tests para verificar instalación
python test_flan.py

# 2. Demo rápida (5 minutos)
python demo_flan.py

# 3. Experimento completo (30-60 minutos)
python flan_qlearning_solution.py
```

### Resultados Esperados

1. **Tests**: Todos los tests deben pasar, verificando:
   - Funcionamiento del entorno
   - Discretización correcta
   - Algoritmos de aprendizaje

2. **Demo**: Muestra:
   - Comparación antes/después del entrenamiento
   - Diferentes niveles de discretización
   - Métricas de rendimiento

3. **Experimento Completo**: Genera:
   - `flan_results/`: Directorio con todos los resultados
   - `best_models.pkl`: Mejores modelos entrenados
   - `training_curves.png`: Curvas de aprendizaje
   - `performance_comparison.png`: Comparación de rendimiento
   - `hyperparameter_analysis.png`: Análisis de hiperparámetros
   - `results_summary.json`: Resumen de resultados

### Interpretación de Resultados

- **Tasa de Éxito**: >90% indica buen aprendizaje
- **Recompensa Promedio**: Valores cercanos a 0 son óptimos
- **Curvas de Aprendizaje**: Deben mostrar mejora gradual y estabilización

## Proyecto 2: BORED (Board-Oriented Reasoning for Emergent Domination)

### Descripción
BORED implementa los algoritmos Minimax y Expectimax con alpha-beta pruning para jugar TacTix, un juego de estrategia por turnos.

### Ejecución Rápida

```bash
cd tactix

# 1. Demo rápida (5 minutos)
python demo_bored.py

# 2. Experimento completo (20-30 minutos)
python bored_solution.py
```

### Resultados Esperados

1. **Demo**: Muestra:
   - Partida individual con visualización
   - Comparación con/sin alpha-beta pruning
   - Minimax vs Expectimax
   - Evaluación heurística

2. **Experimento Completo**: Genera:
   - `bored_models.pkl`: Modelos entrenados
   - `bored_results.json`: Resultados detallados del torneo
   - `bored_results.png`: Visualizaciones de rendimiento
   - Reporte en consola con estadísticas

### Interpretación de Resultados

- **Tasa de Victoria**: >80% contra Random indica buen rendimiento
- **Reducción por Pruning**: 60-70% es típico
- **Profundidad vs Rendimiento**: D3-D4 suele ser óptimo

## Comparación de Modelos

### FLAN: Q-Learning vs Q-Learning Estocástico

| Aspecto | Q-Learning | Q-Learning Estocástico |
|---------|------------|------------------------|
| Exploración | ε-greedy determinista | Muestreo estocástico |
| Convergencia | Más rápida | Más lenta pero robusta |
| Rendimiento Final | Alto con buena discretización | Consistente entre discretizaciones |
| Uso de Memoria | Tabla Q completa | Tabla Q + distribución de probabilidades |

### BORED: Minimax vs Expectimax

| Aspecto | Minimax | Expectimax |
|---------|---------|------------|
| Asunción | Oponente óptimo | Oponente probabilístico |
| Complejidad | O(b^d) con pruning | O(b^d) sin pruning efectivo en nodos chance |
| Mejor Contra | Oponentes fuertes | Oponentes aleatorios/subóptimos |
| Tiempo de Decisión | Rápido con pruning | Más lento |

## Métricas de Evaluación

### FLAN
1. **Tasa de Éxito**: Porcentaje de episodios completados exitosamente
2. **Recompensa Promedio**: Indicador de calidad del vuelo
3. **Desviación de Altitud**: Precisión en el mantenimiento de altitud
4. **Suavidad del Descenso**: Cambios en velocidad vertical

### BORED
1. **Tasa de Victoria**: Porcentaje de partidas ganadas
2. **Nodos Evaluados**: Eficiencia computacional
3. **Tiempo por Movimiento**: Viabilidad práctica
4. **Efectividad del Pruning**: Optimización lograda

## Troubleshooting

### Problemas Comunes

1. **ImportError**: Verificar que todas las dependencias estén instaladas
2. **MemoryError en FLAN**: Reducir el tamaño de discretización o número de episodios
3. **Timeout en BORED**: Reducir la profundidad de búsqueda
4. **Gráficos no se muestran**: Verificar backend de matplotlib

### Soluciones

```python
# Para gráficos en servidores sin display
import matplotlib
matplotlib.use('Agg')  # Antes de importar pyplot

# Para problemas de memoria
import gc
gc.collect()  # Liberar memoria entre experimentos
```

## Resumen de Cumplimiento de Rúbricas

### FLAN ✓
- [x] Q-Learning clásico implementado
- [x] Q-Learning estocástico implementado
- [x] Múltiples discretizaciones (3 niveles)
- [x] Búsqueda sistemática de hiperparámetros
- [x] Evaluación exhaustiva con métricas
- [x] Gráficos relevantes con interpretación
- [x] Modelos guardados en .pkl

### BORED ✓
- [x] Minimax con alpha-beta pruning
- [x] Expectimax con alpha-beta pruning
- [x] Análisis del impacto del pruning
- [x] 4 heurísticas justificadas con ponderaciones
- [x] Evaluación contra múltiples oponentes
- [x] Registro detallado de partidas
- [x] Gráficos comparativos
- [x] Modelos guardados en .pkl

## Conclusiones

Ambos proyectos demuestran la aplicación exitosa de técnicas fundamentales de IA:

1. **FLAN**: Muestra cómo el aprendizaje por refuerzo puede resolver problemas de control continuo mediante discretización apropiada.

2. **BORED**: Ilustra la efectividad de algoritmos de búsqueda adversarial clásicos con optimizaciones modernas.

Los resultados obtenidos validan las implementaciones y proporcionan insights sobre:
- La importancia de la discretización en RL
- El impacto de las optimizaciones algorítmicas
- El diseño de funciones heurísticas
- La evaluación sistemática de agentes inteligentes

Para más detalles técnicos, consultar los README específicos de cada proyecto. 