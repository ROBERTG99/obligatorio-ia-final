# Obligatorio IA - Marzo 2025

**Repositorio PRIVADO** para el proyecto obligatorio de Inteligencia Artificial.

## 🤖 Proyectos Implementados

### 1. FLAN - Control de Descenso de Aeronaves (`descent-env/`)
Implementación de algoritmos de Q-Learning para el control autónomo del descenso de aeronaves.

**Algoritmos implementados:**
- Q-Learning estándar con Double Q-Learning
- Stochastic Q-Learning optimizado
- Reward Shaping avanzado
- Optimización de hiperparámetros con paralelización

**Características destacadas:**
- ✅ Uso prioritario de DescentEnv real (BlueSky)
- ✅ MockDescentEnv como fallback de compatibilidad  
- ✅ Entrenamiento con 5,000 episodios (competitivo)
- ✅ Evaluación robusta con 500 episodios
- ✅ Paralelización en todos los cores disponibles

### 2. BORED - Juego TacTix (`tactix/`)
Implementación de algoritmos Minimax y Expectimax para el juego estratégico TacTix.

**Algoritmos implementados:**
- Minimax con poda alfa-beta
- Expectimax para manejo de incertidumbre
- Optimizaciones de rendimiento
- Interfaz de juego interactiva

## 📁 Estructura del Proyecto

```
obligatorio-ia-marzo-2025/
├── descent-env/           # Proyecto FLAN
│   ├── flan_qlearning_solution.py
│   ├── descent_env.py
│   ├── mock_descent_env.py
│   └── [otros archivos FLAN]
├── tactix/               # Proyecto BORED  
│   ├── bored_solution.py
│   ├── tactix_env.py
│   ├── agent.py
│   └── [otros archivos BORED]
├── INSTRUCCIONES_EJECUCION_FINAL.md
├── README_COMPLETO.md
├── run_all_experiments_optimized.py
└── verificar_entrega.py
```

## 🚀 Ejecución Rápida

### FLAN (Descenso de Aeronaves)
```bash
cd descent-env
python flan_qlearning_solution.py
```

### BORED (TacTix Game)  
```bash
cd tactix
python bored_solution.py
```

### Experimentos Completos
```bash
python run_all_experiments_optimized.py
```

## 📊 Rendimiento Destacado

### FLAN
- **Entrenamiento:** 5,000 episodios por agente
- **Evaluación:** 500 episodios robustos  
- **Optimización:** Paralelización completa
- **Técnicas:** Double Q-Learning + Reward Shaping

### BORED
- **Profundidad:** Minimax optimizado
- **Velocidad:** Poda alfa-beta eficiente
- **Robustez:** Expectimax para incertidumbre

## 🔧 Configuración del Entorno

Cada proyecto tiene su propio `pyproject.toml` con dependencias específicas:

```bash
# Para FLAN
cd descent-env && poetry install

# Para BORED  
cd tactix && poetry install
```

## 📝 Documentación

- `README_COMPLETO.md` - Documentación detallada del proyecto
- `INSTRUCCIONES_EJECUCION_FINAL.md` - Guía de ejecución paso a paso
- `descent-env/README_FLAN.md` - Detalles específicos de FLAN
- `tactix/README_BORED.md` - Detalles específicos de BORED

## ✅ Verificación

```bash
python verificar_entrega.py
```

---

**Autor:** ROBERT G  
**Fecha:** Julio 2024  
**Repositorio:** Privado ✅  
**Estado:** Entrega Final Completada 🎯 