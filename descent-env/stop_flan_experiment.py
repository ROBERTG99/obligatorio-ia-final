#!/usr/bin/env python3
"""
Script para detener de forma segura el experimento FLAN en ejecución
"""

import os
import signal
import psutil
import sys

def find_flan_processes():
    """Encuentra procesos relacionados con FLAN"""
    flan_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if any(keyword in cmdline.lower() for keyword in ['flan', 'descent', 'qlearning']):
                flan_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return flan_processes

def stop_flan_experiment():
    """Detiene el experimento FLAN de forma segura"""
    
    print("🔍 Buscando procesos del experimento FLAN...")
    processes = find_flan_processes()
    
    if not processes:
        print("✅ No se encontraron procesos FLAN en ejecución")
        return
    
    print(f"📋 Encontrados {len(processes)} procesos relacionados:")
    for proc in processes:
        try:
            cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else proc.name()
            print(f"   PID {proc.pid}: {cmdline[:80]}...")
        except:
            print(f"   PID {proc.pid}: {proc.name()}")
    
    response = input("\n❓ ¿Detener estos procesos? (y/N): ")
    if response.lower() != 'y':
        print("❌ Operación cancelada")
        return
    
    print("🛑 Deteniendo procesos...")
    for proc in processes:
        try:
            proc.send_signal(signal.SIGTERM)  # Señal suave primero
            print(f"✅ Enviada señal SIGTERM a PID {proc.pid}")
        except Exception as e:
            print(f"❌ Error enviando señal a PID {proc.pid}: {e}")
    
    print("\n⏱️  Esperando 5 segundos para terminación suave...")
    import time
    time.sleep(5)
    
    # Verificar si aún están ejecutándose
    still_running = []
    for proc in processes:
        try:
            if proc.is_running():
                still_running.append(proc)
        except:
            pass
    
    if still_running:
        print(f"⚠️  {len(still_running)} procesos aún ejecutándose. Forzando terminación...")
        for proc in still_running:
            try:
                proc.send_signal(signal.SIGKILL)
                print(f"🔨 Enviada señal SIGKILL a PID {proc.pid}")
            except Exception as e:
                print(f"❌ Error forzando terminación de PID {proc.pid}: {e}")
    
    print("\n✅ Operación completada")

if __name__ == "__main__":
    stop_flan_experiment()
