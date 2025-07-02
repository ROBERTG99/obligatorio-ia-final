# Proyecto FLAN - Flight-Level Adjustment Network

## Descripción del Problema

El proyecto FLAN (Flight-Level Adjustment Network) es parte del Programa de Aprendizaje Profundamente Aproximado (PAPA) de SkyNotCrash™. El objetivo es entrenar un agente de inteligencia artificial que controle la velocidad vertical de una aeronave experimental para mantenerla a la altura objetivo el mayor tiempo posible antes de iniciar el descenso hacia la pista de aterrizaje.

## Contexto del Entorno

### Espacio de Observaciones
El entorno proporciona observaciones continuas que incluyen:
- **altitude**: Altitud actual de la aeronave (normalizada)
- **vz**: Velocidad vertical actual (normalizada)
- **target_altitude**: Altitud objetivo (normalizada)
- **runway_distance**: Distancia a la pista de aterrizaje (normalizada)

### Espacio de Acciones
- **Acción continua**: Valor entre -1 y 1 que representa la velocidad vertical deseada
- **-1**: Máximo descenso
- **1**: Máximo ascenso

### Función de Recompensa
- **Recompensa negativa**: Proporcional a la diferencia entre altitud actual y objetivo
- **Penalización por choque**: -100 si la aeronave toca el suelo
- **Recompensa final**: Basada en la altitud final al llegar a la pista

## Implementación

### 1. Discretización de Espacios

Se implementaron múltiples esquemas de discretización para convertir los espacios continuos en discretos:

#### Esquemas Probados:
- **Gruesa**: 5 bins por dimensión (625 estados totales)
- **Media**: 10 bins por dimensión (10,000 estados totales)
- **Fina**: 15 bins por dimensión (50,625 estados totales)

#### Justificación de la Discretización:
- **Granularidad vs. Complejidad**: Más bins proporcionan mayor precisión pero aumentan la complejidad computacional
- **Exploración**: Discretizaciones más gruesas facilitan la exploración inicial
- **Convergencia**: Discretizaciones más finas permiten políticas más precisas

### 2. Q-Learning Estándar

Implementación clásica de Q-Learning con:
- **Política epsilon-greedy**: Balance entre exploración y explotación
- **Learning rate adaptativo**: Se reduce con el número de visitas al estado-acción
- **Ecuación de Bellman**: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

### 3. Stochastic Q-Learning

Implementación basada en el artículo "Reinforcement Learning in Large Discrete Action Spaces" que aborda el problema de espacios de acción grandes:

#### Características Principales:
- **Muestreo de acciones**: En lugar de evaluar todas las acciones, se muestrea un subconjunto
- **Búsqueda eficiente**: Complejidad O(k) en lugar de O(|A|) donde k es el tamaño de muestra
- **Generalización**: Permite manejar espacios de acción discretos grandes

#### Algoritmo:
1. **Selección de acción**: Muestrear k acciones del espacio de acciones
2. **Evaluación**: Calcular Q-values para las acciones muestreadas
3. **Selección**: Elegir la acción con mayor Q-value entre las muestreadas
4. **Actualización**: Usar la ecuación de Bellman con el máximo de las acciones muestreadas

### 4. Búsqueda de Hiperparámetros

Se implementó una búsqueda en cuadrícula sistemática para optimizar:

#### Para Q-Learning Estándar:
- **Learning rate**: [0.01, 0.05, 0.1, 0.2]
- **Discount factor**: [0.9, 0.95, 0.99]
- **Epsilon**: [0.05, 0.1, 0.2]

#### Para Stochastic Q-Learning:
- **Learning rate**: [0.01, 0.05, 0.1, 0.2]
- **Discount factor**: [0.9, 0.95, 0.99]
- **Epsilon**: [0.05, 0.1, 0.2]
- **Sample size**: [5, 10, 15]

#### Métrica de Evaluación:
- **Recompensa promedio**: Se evalúa cada configuración durante 50 episodios
- **Consistencia**: Se selecciona la configuración con mayor recompensa promedio

## Resultados Esperados

### Comparación de Algoritmos

#### Q-Learning Estándar:
- **Ventajas**: Simplicidad, convergencia garantizada
- **Desventajas**: Escalabilidad limitada, exploración ineficiente en espacios grandes

#### Stochastic Q-Learning:
- **Ventajas**: Escalabilidad, exploración eficiente, mejor rendimiento en espacios grandes
- **Desventajas**: Mayor complejidad, convergencia aproximada

### Impacto de la Discretización

#### Discretización Gruesa:
- **Aprendizaje rápido**: Menos estados para explorar
- **Política limitada**: Precisión reducida
- **Adecuada para**: Exploración inicial y prototipado

#### Discretización Fina:
- **Aprendizaje lento**: Más estados para explorar
- **Política precisa**: Mayor capacidad de control
- **Adecuada para**: Optimización final y rendimiento máximo

#### Discretización Media:
- **Balance óptimo**: Entre velocidad de aprendizaje y precisión
- **Recomendada para**: Producción y aplicaciones reales

## Uso del Código

### Ejecutar Demo Simplificado:
```bash
python demo_flan.py
```

### Ejecutar Experimentación Completa:
```bash
python flan_qlearning_solution.py
```

### Archivos Generados:
- `flan_results.json`: Resultados detallados de todos los experimentos
- `flan_results.png`: Gráficos comparativos de rendimiento
- `demo_flan_results.png`: Gráficos de la demostración simplificada

## Análisis de Resultados

### Métricas de Evaluación:
1. **Recompensa promedio**: Indicador principal de rendimiento
2. **Tiempo de supervivencia**: Duración de los episodios
3. **Error de altitud**: Precisión en el control de altura
4. **Consistencia**: Variabilidad en el rendimiento

### Interpretación de Resultados:
- **Stochastic Q-Learning** debería mostrar mejor rendimiento en discretizaciones más finas
- **Q-Learning estándar** puede ser más efectivo en discretizaciones gruesas
- **Discretización media** debería proporcionar el mejor balance general

## Conclusiones

### Ventajas de Stochastic Q-Learning:
1. **Escalabilidad**: Maneja eficientemente espacios de acción grandes
2. **Exploración**: Mejor balance exploración/explotación
3. **Rendimiento**: Mejores resultados en problemas complejos

### Recomendaciones:
1. **Desarrollo**: Usar discretización gruesa para prototipado rápido
2. **Optimización**: Usar discretización media para ajuste fino
3. **Producción**: Usar Stochastic Q-Learning con discretización media
4. **Investigación**: Explorar discretizaciones adaptativas

### Aplicaciones Futuras:
- Control de drones autónomos
- Sistemas de navegación aérea
- Simuladores de vuelo
- Robótica aérea

## Referencias

1. Dulac-Arnold, G., et al. "Reinforcement Learning in Large Discrete Action Spaces." arXiv:1512.07679 (2015)
2. Van de Wiele, T., et al. "Q-learning in Enormous Action Spaces via Amortized Approximate Maximization." arXiv:2001.08116 (2020)
3. Sutton, R.S., & Barto, A.G. "Reinforcement Learning: An Introduction." MIT Press (2018)

## Estructura del Proyecto

```
descent-env/
├── descent_env.py              # Entorno de simulación
├── flan_qlearning_solution.py  # Implementación completa
├── demo_flan.py               # Demo simplificado
├── README_FLAN.md             # Este archivo
├── continuous_descent_env.ipynb # Notebook de ejemplo
└── example.gif                # Visualización del entorno
```

## Autores

Proyecto desarrollado como parte del Programa de Aprendizaje Profundamente Aproximado (PAPA) de SkyNotCrash™.

---

*"El único inconveniente: olvidaron diseñar al piloto. Pero no hay problema, porque para eso están ustedes."* 