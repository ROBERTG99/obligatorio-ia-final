# Instrucciones de Ejecución - MacBook M1 Pro

## Estado del Proyecto

✅ **Todos los archivos están listos para ejecutar**
✅ **Dependencias verificadas**
✅ **Código corregido y optimizado**

## Resumen de Correcciones Realizadas

### Proyecto FLAN
- ✅ Creado `MockDescentEnv` para ejecutar sin BlueSky
- ✅ Corregidos errores de tipos en discretización
- ✅ Agregada importación condicional del entorno
- ✅ Archivos listos: `flan_qlearning_solution.py`, `demo_flan.py`, `test_flan.py`

### Proyecto BORED
- ✅ Creado `TrainerAgentWrapper` para compatibilidad con interfaz Agent
- ✅ Corregidos errores de tipos y herencia
- ✅ Verificación de atributos opcionales
- ✅ Archivos listos: `bored_solution.py`, `demo_bored.py`

## Orden de Ejecución Recomendado

### Opción 1: Ejecución Automática (Recomendada)

```bash
# Dar permisos de ejecución
chmod +x run_experiments_m1.sh

# Ejecutar todos los experimentos
./run_experiments_m1.sh
```

Este script:
- Verifica e instala dependencias automáticamente
- Ejecuta pruebas y demos de ambos proyectos
- Genera modelos guardados en formato .pkl
- Optimizado para cores de performance del M1 Pro

### Opción 2: Ejecución Manual Paso a Paso

#### 1. Verificar Configuración
```bash
python3 check_setup.py
```

#### 2. Proyecto FLAN

```bash
# Cambiar al directorio
cd descent-env

# Ejecutar pruebas
python3 test_flan.py

# Ejecutar demo (rápido)
python3 demo_flan.py

# Ejecutar experimento completo (más lento)
python3 flan_qlearning_solution.py

# Volver al directorio principal
cd ..
```

#### 3. Proyecto BORED

```bash
# Cambiar al directorio
cd tactix

# Ejecutar demo (rápido)
python3 demo_bored.py

# Ejecutar experimento completo (más lento)
python3 bored_solution.py

# Volver al directorio principal
cd ..
```

## Archivos Generados

### FLAN
- `descent-env/flan_models_quick.pkl` - Modelos entrenados (versión rápida)
- `descent-env/flan_models.pkl` - Modelos entrenados (versión completa)
- `descent-env/flan_results.json` - Resultados detallados
- `descent-env/flan_results.png` - Gráficos comparativos
- `descent-env/demo_flan_results.png` - Gráficos de demo

### BORED
- `tactix/bored_models_quick.pkl` - Modelos entrenados (versión rápida)
- `tactix/bored_models.pkl` - Modelos entrenados (versión completa)
- `tactix/bored_results.json` - Resultados del torneo
- `tactix/bored_results.png` - Visualizaciones

## Tiempos Estimados de Ejecución

### MacBook M1 Pro (8 cores de performance)

| Comando | Tiempo Estimado |
|---------|----------------|
| `check_setup.py` | < 1 segundo |
| `test_flan.py` | < 5 segundos |
| `demo_flan.py` | 1-2 minutos |
| `flan_qlearning_solution.py` | 15-20 minutos |
| `demo_bored.py` | < 1 minuto |
| `bored_solution.py` | 10-15 minutos |
| `run_experiments_m1.sh` | 3-5 minutos (versión rápida) |

## Optimizaciones para M1 Pro

El código está optimizado para aprovechar:
- Vectorización con NumPy (usa Accelerate framework de Apple)
- Threads únicos para evitar overhead (OPENBLAS_NUM_THREADS=1)
- Uso eficiente de memoria
- Matplotlib con backend optimizado

## Notas Importantes

1. **BlueSky no es necesario**: Se usa `MockDescentEnv` automáticamente
2. **Los modelos se guardan automáticamente** en formato .pkl
3. **Los gráficos se guardan como PNG** para revisión posterior
4. **Todos los experimentos son reproducibles** con semillas fijas

## Solución de Problemas

### Si faltan dependencias:
```bash
pip3 install numpy matplotlib gymnasium seaborn
```

### Si hay errores de permisos:
```bash
chmod +x run_experiments_m1.sh
chmod +x check_setup.py
```

### Si matplotlib no muestra gráficos:
```bash
# Instalar backend para macOS
pip3 install pyobjc-framework-Cocoa
```

## Verificación de Resultados

Después de ejecutar, verificar:

1. **Archivos .pkl creados** en cada carpeta
2. **Gráficos PNG generados**
3. **Archivos JSON con resultados**
4. **Logs en consola sin errores**

## Contacto y Soporte

Si encuentras algún problema:
1. Verificar que todas las dependencias estén instaladas
2. Revisar los logs de error
3. Ejecutar `python3 check_setup.py` para diagnóstico

¡Los experimentos están listos para ejecutar! 