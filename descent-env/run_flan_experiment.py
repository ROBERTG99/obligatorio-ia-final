#!/usr/bin/env python3
"""
Script optimizado para ejecutar el experimento FLAN completo
Optimizado para MacBook M1 Pro
"""

import os
import sys
import time
from flan_qlearning_solution import main

def run_experiment():
    """Ejecuta el experimento completo"""
    print("="*80)
    print("PROYECTO FLAN - Q-LEARNING PARA CONTROL DE DESCENSO")
    print("="*80)
    print("Optimizado para MacBook M1 Pro")
    print()
    
    # Configurar entorno para optimización
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    start_time = time.time()
    
    try:
        print("Iniciando experimento...")
        results = main()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*80}")
        print("EXPERIMENTO COMPLETADO EXITOSAMENTE")
        print(f"{'='*80}")
        print(f"Tiempo total: {duration:.1f} segundos ({duration/60:.1f} minutos)")
        print()
        print("Archivos generados:")
        print("- flan_results.json: Resultados completos")
        print("- flan_results.png: Gráficos de análisis")
        print("- models_fina/: Modelos con discretización fina")
        print("- models_media/: Modelos con discretización media") 
        print("- models_gruesa/: Modelos con discretización gruesa")
        print()
        print("¡Experimento FLAN completado con éxito!")
        
        return results
        
    except Exception as e:
        print(f"\nError durante el experimento: {e}")
        print("Traceback completo:")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_experiment()
    
    if results is not None:
        sys.exit(0)
    else:
        sys.exit(1) 