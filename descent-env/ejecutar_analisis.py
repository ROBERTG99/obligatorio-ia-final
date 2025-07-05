#!/usr/bin/env python3
"""
🚀 EJECUTOR DE ANÁLISIS AUTOMÁTICO FLAN
Script simple para ejecutar análisis de resultados JSON
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🎯 EJECUTOR DE ANÁLISIS AUTOMÁTICO FLAN")
    print("="*50)
    
    # Buscar archivos JSON disponibles
    json_files = list(Path('.').glob('flan_results*.json'))
    
    if not json_files:
        print("❌ No se encontraron archivos JSON de resultados")
        print("   Asegúrate de que el experimento haya generado resultados")
        return
    
    print(f"📊 Se encontraron {len(json_files)} archivos JSON:")
    for i, f in enumerate(json_files, 1):
        size = f.stat().st_size / 1024 / 1024  # MB
        print(f"  {i}. {f.name} ({size:.1f} MB)")
    
    # Permitir al usuario elegir
    if len(json_files) == 1:
        selected_file = json_files[0]
        print(f"\n📈 Analizando automáticamente: {selected_file.name}")
    else:
        try:
            choice = input(f"\n🔍 Selecciona archivo (1-{len(json_files)}, o Enter para el más reciente): ")
            if not choice:
                selected_file = sorted(json_files, key=lambda x: x.stat().st_mtime)[-1]
                print(f"📈 Usando archivo más reciente: {selected_file.name}")
            else:
                selected_file = json_files[int(choice) - 1]
                print(f"📈 Analizando: {selected_file.name}")
        except (ValueError, IndexError):
            selected_file = sorted(json_files, key=lambda x: x.stat().st_mtime)[-1]
            print(f"�� Usando archivo más reciente: {selected_file.name}")
    
    # Ejecutar análisis
    print("\n🔍 Ejecutando análisis avanzado...")
    try:
        result = subprocess.run([
            sys.executable, 'analizar_json_avanzado.py', 
            '--json', str(selected_file)
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("⚠️ Advertencias:")
            print(result.stderr)
            
        print(f"\n✅ Análisis completado")
        print(f"📊 Resultados guardados en: mejoras_automaticas.py")
        print(f"📈 Visualización guardada en: diagnostico_json_flan.png")
        
    except Exception as e:
        print(f"❌ Error al ejecutar análisis: {e}")

if __name__ == "__main__":
    main()
