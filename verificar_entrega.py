#!/usr/bin/env python3
"""
Script de verificación de entrega completa
Verifica que todos los archivos necesarios estén presentes
"""

import os
import sys

def verificar_estructura():
    """Verifica la estructura de directorios y archivos"""
    
    print("="*60)
    print("VERIFICACIÓN DE ENTREGA - Obligatorio IA Marzo 2025")
    print("="*60)
    
    # Archivos requeridos por proyecto
    archivos_flan = {
        'descent-env/descent_env.py': 'Entorno del simulador',
        'descent-env/flan_qlearning_solution.py': 'Solución Q-Learning',
        'descent-env/demo_flan.py': 'Demo FLAN',
        'descent-env/test_flan.py': 'Tests FLAN',
        'descent-env/README_FLAN.md': 'Documentación FLAN',
        'descent-env/RESUMEN_FLAN.md': 'Resumen ejecutivo FLAN'
    }
    
    archivos_bored = {
        'tactix/tactix_env.py': 'Entorno TacTix',
        'tactix/bored_solution.py': 'Solución Minimax/Expectimax',
        'tactix/demo_bored.py': 'Demo BORED',
        'tactix/README_BORED.md': 'Documentación BORED',
        'tactix/RESUMEN_BORED.md': 'Resumen ejecutivo BORED',
        'tactix/agent.py': 'Clase base Agent',
        'tactix/trainer_agent.py': 'Agente entrenador',
        'tactix/random_agent.py': 'Agente aleatorio'
    }
    
    archivos_generales = {
        'README_COMPLETO.md': 'Guía completa del proyecto'
    }
    
    # Verificar FLAN
    print("\n1. PROYECTO FLAN")
    print("-" * 40)
    flan_ok = True
    for archivo, descripcion in archivos_flan.items():
        if os.path.exists(archivo):
            print(f"✓ {archivo} - {descripcion}")
        else:
            print(f"✗ {archivo} - FALTANTE")
            flan_ok = False
    
    # Verificar BORED
    print("\n2. PROYECTO BORED")
    print("-" * 40)
    bored_ok = True
    for archivo, descripcion in archivos_bored.items():
        if os.path.exists(archivo):
            print(f"✓ {archivo} - {descripcion}")
        else:
            print(f"✗ {archivo} - FALTANTE")
            bored_ok = False
    
    # Verificar archivos generales
    print("\n3. DOCUMENTACIÓN GENERAL")
    print("-" * 40)
    general_ok = True
    for archivo, descripcion in archivos_generales.items():
        if os.path.exists(archivo):
            print(f"✓ {archivo} - {descripcion}")
        else:
            print(f"✗ {archivo} - FALTANTE")
            general_ok = False
    
    # Verificar archivos generados (opcionales)
    print("\n4. ARCHIVOS GENERADOS (Opcional)")
    print("-" * 40)
    
    archivos_opcionales = [
        'descent-env/best_models.pkl',
        'descent-env/flan_results/',
        'tactix/bored_models.pkl',
        'tactix/bored_results.json'
    ]
    
    for archivo in archivos_opcionales:
        if os.path.exists(archivo):
            print(f"✓ {archivo}")
        else:
            print(f"○ {archivo} - No generado aún")
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE VERIFICACIÓN")
    print("="*60)
    
    todo_ok = flan_ok and bored_ok and general_ok
    
    if todo_ok:
        print("✅ TODOS LOS ARCHIVOS REQUERIDOS ESTÁN PRESENTES")
        print("\nLa entrega está completa. Pasos siguientes:")
        print("1. Ejecutar demos para verificar funcionamiento:")
        print("   - python descent-env/demo_flan.py")
        print("   - python tactix/demo_bored.py")
        print("2. Ejecutar experimentos completos si es necesario:")
        print("   - python descent-env/flan_qlearning_solution.py")
        print("   - python tactix/bored_solution.py")
        print("3. Comprimir y entregar el directorio completo")
    else:
        print("❌ FALTAN ARCHIVOS REQUERIDOS")
        print("\nPor favor, verifica los archivos faltantes antes de entregar.")
    
    # Verificar tamaño total
    print("\n" + "="*60)
    print("INFORMACIÓN ADICIONAL")
    print("="*60)
    
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk('.'):
        # Ignorar directorios ocultos y __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if not file.startswith('.') and not file.endswith('.pyc'):
                filepath = os.path.join(root, file)
                try:
                    size = os.path.getsize(filepath)
                    total_size += size
                    file_count += 1
                except:
                    pass
    
    print(f"Total de archivos: {file_count}")
    print(f"Tamaño total: {total_size / (1024*1024):.2f} MB")
    
    return todo_ok

def verificar_imports():
    """Verifica que los imports principales funcionen"""
    print("\n" + "="*60)
    print("VERIFICACIÓN DE IMPORTS")
    print("="*60)
    
    imports_ok = True
    
    # Verificar imports comunes
    imports_comunes = [
        'numpy',
        'matplotlib',
        'gymnasium',
        'pandas',
        'seaborn',
        'tqdm',
        'pickle',
        'json'
    ]
    
    for modulo in imports_comunes:
        try:
            __import__(modulo)
            print(f"✓ {modulo}")
        except ImportError:
            print(f"✗ {modulo} - NO INSTALADO")
            imports_ok = False
    
    if not imports_ok:
        print("\n⚠️  Algunos módulos no están instalados.")
        print("Instala las dependencias con:")
        print("pip install numpy matplotlib gymnasium pandas seaborn tqdm")
    else:
        print("\n✅ Todas las dependencias están instaladas")
    
    return imports_ok

def main():
    """Función principal"""
    
    # Cambiar al directorio del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Verificar estructura
    estructura_ok = verificar_estructura()
    
    # Verificar imports
    imports_ok = verificar_imports()
    
    # Resultado final
    print("\n" + "="*60)
    print("RESULTADO FINAL")
    print("="*60)
    
    if estructura_ok and imports_ok:
        print("✅ LA ENTREGA ESTÁ LISTA")
        return 0
    else:
        print("❌ HAY PROBLEMAS QUE RESOLVER")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 