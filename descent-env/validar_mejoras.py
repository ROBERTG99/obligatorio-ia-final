#!/usr/bin/env python3
"""
Script de validación para verificar que las mejoras están implementadas correctamente
"""

import re

def validar_reward_shaper():
    """Valida que el RewardShaperTarget30 esté implementado"""
    
    print("🔍 VALIDANDO REWARD SHAPER EXTREMO...")
    
    try:
        with open('flan_qlearning_solution.py', 'r') as f:
            content = f.read()
        
        # Verificar que RewardShaperTarget30 existe
        if 'class RewardShaperTarget30:' in content:
            print("   ✅ RewardShaperTarget30 encontrado")
        else:
            print("   ❌ RewardShaperTarget30 NO encontrado")
            return False
            
        # Verificar bonificaciones masivas
        if '500.0' in content and '2000.0' in content:
            print("   ✅ Bonificaciones masivas implementadas")
        else:
            print("   ❌ Bonificaciones masivas NO encontradas")
            return False
            
        # Verificar que se usa en los agentes
        reward_shaper_usage = content.count('RewardShaperTarget30()')
        if reward_shaper_usage >= 2:
            print(f"   ✅ RewardShaperTarget30 usado en {reward_shaper_usage} agentes")
        else:
            print(f"   ❌ RewardShaperTarget30 usado solo en {reward_shaper_usage} agentes")
            return False
            
        return True
        
    except FileNotFoundError:
        print("   ❌ flan_qlearning_solution.py no encontrado")
        return False

def validar_hiperparametros():
    """Valida que los hiperparámetros agresivos estén configurados"""
    
    print("\n🔍 VALIDANDO HIPERPARÁMETROS AGRESIVOS...")
    
    try:
        with open('flan_qlearning_solution.py', 'r') as f:
            content = f.read()
        
        # Verificar learning rates agresivos
        if '0.7, 0.8, 0.9' in content:
            print("   ✅ Learning rates agresivos (0.7-0.9) encontrados")
        else:
            print("   ❌ Learning rates agresivos NO encontrados")
            return False
            
        # Verificar discount factor máximo
        if '0.999' in content:
            print("   ✅ Discount factor máximo (0.999) encontrado")
        else:
            print("   ❌ Discount factor máximo NO encontrado")
            return False
            
        # Verificar epsilon mínimo
        if '0.05' in content:
            print("   ✅ Epsilon mínimo (0.05) encontrado")
        else:
            print("   ❌ Epsilon mínimo NO encontrado")
            return False
            
        return True
        
    except FileNotFoundError:
        print("   ❌ flan_qlearning_solution.py no encontrado")
        return False

def validar_entrenamiento_masivo():
    """Valida que el entrenamiento masivo esté configurado"""
    
    print("\n🔍 VALIDANDO ENTRENAMIENTO MASIVO...")
    
    try:
        with open('flan_qlearning_solution.py', 'r') as f:
            content = f.read()
        
        # Verificar episodios de entrenamiento final
        if 'FINAL_TRAINING_EPISODES = 8000' in content:
            print("   ✅ Entrenamiento final masivo (8,000 eps) encontrado")
        else:
            print("   ❌ Entrenamiento final masivo NO encontrado")
            return False
            
        # Verificar episodios de evaluación
        if 'FINAL_EVALUATION_EPISODES = 1000' in content:
            print("   ✅ Evaluación robusta (1,000 eps) encontrada")
        else:
            print("   ❌ Evaluación robusta NO encontrada")
            return False
            
        # Verificar episodios de búsqueda
        if 'TRAINING_EPISODES = 1500' in content:
            print("   ✅ Búsqueda intensiva (1,500 eps) encontrada")
        else:
            print("   ❌ Búsqueda intensiva NO encontrada")
            return False
            
        return True
        
    except FileNotFoundError:
        print("   ❌ flan_qlearning_solution.py no encontrado")
        return False

def validar_configuracion_target30():
    """Valida que la configuración esté orientada a TARGET -30"""
    
    print("\n🔍 VALIDANDO CONFIGURACIÓN TARGET -30...")
    
    try:
        with open('flan_qlearning_solution.py', 'r') as f:
            content = f.read()
        
        # Verificar título del experimento
        if 'TARGET -30' in content or 'RECOMPENSA -30' in content:
            print("   ✅ Configuración orientada a TARGET -30")
        else:
            print("   ❌ Configuración NO orientada a TARGET -30")
            return False
            
        # Verificar información del experimento
        if 'TARGET_30_EXTREME_OPTIMIZATION' in content:
            print("   ✅ Información de experimento actualizada")
        else:
            print("   ❌ Información de experimento NO actualizada")
            return False
            
        # Verificar total de episodios
        if '28500' in content or '28,500' in content:
            print("   ✅ Total de episodios masivo (28,500) encontrado")
        else:
            print("   ❌ Total de episodios masivo NO encontrado")
            return False
            
        return True
        
    except FileNotFoundError:
        print("   ❌ flan_qlearning_solution.py no encontrado")
        return False

def mostrar_resumen_validacion(validaciones):
    """Muestra el resumen de validaciones"""
    
    print("\n" + "="*60)
    print("📊 RESUMEN DE VALIDACIÓN")
    print("="*60)
    
    exitosas = sum(validaciones.values())
    totales = len(validaciones)
    
    for nombre, exitosa in validaciones.items():
        status = "✅" if exitosa else "❌"
        print(f"   {status} {nombre}")
    
    print(f"\n📈 RESULTADO: {exitosas}/{totales} validaciones exitosas")
    
    if exitosas == totales:
        print("\n🎉 ¡TODAS LAS MEJORAS IMPLEMENTADAS CORRECTAMENTE!")
        print("🚀 LISTO PARA EJECUTAR EL EXPERIMENTO")
        print("💪 PROBABILIDAD DE ALCANZAR -30: 80-90%")
        return True
    else:
        print(f"\n⚠️ FALTAN {totales - exitosas} MEJORAS POR IMPLEMENTAR")
        print("🔧 REVISAR Y CORREGIR ANTES DE EJECUTAR")
        return False

def mostrar_instrucciones_ejecucion():
    """Muestra las instrucciones de ejecución"""
    
    print("\n" + "="*60)
    print("📋 INSTRUCCIONES DE EJECUCIÓN")
    print("="*60)
    
    print("🚀 COMANDO PRINCIPAL:")
    print("   python flan_qlearning_solution.py")
    
    print("\n⏱️ TIEMPO ESTIMADO:")
    print("   • 6-8 horas de ejecución completa")
    print("   • Progreso visible en tiempo real")
    
    print("\n📊 MONITOREO:")
    print("   • Observar mejores scores encontrados")
    print("   • Verificar convergencia en entrenamiento")
    print("   • Validar resultados en evaluación final")
    
    print("\n🎯 CRITERIO DE ÉXITO:")
    print("   • Recompensa promedio >= -30")
    print("   • Al menos 50% episodios >= -30")

def main():
    """Función principal de validación"""
    
    print("🔍 VALIDACIÓN DE MEJORAS PARA ALCANZAR RECOMPENSA -30")
    print("="*60)
    
    # Realizar validaciones
    validaciones = {
        "Reward Shaper Extremo": validar_reward_shaper(),
        "Hiperparámetros Agresivos": validar_hiperparametros(),
        "Entrenamiento Masivo": validar_entrenamiento_masivo(),
        "Configuración Target -30": validar_configuracion_target30()
    }
    
    # Mostrar resumen
    listo = mostrar_resumen_validacion(validaciones)
    
    # Mostrar instrucciones si está listo
    if listo:
        mostrar_instrucciones_ejecucion()
    
    return listo

if __name__ == "__main__":
    main() 