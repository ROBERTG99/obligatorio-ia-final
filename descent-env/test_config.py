#!/usr/bin/env python3
"""
Test rápido para verificar que CONFIG_AUTOMATICO es importable
"""

try:
    from flan_qlearning_solution import CONFIG_AUTOMATICO, RewardShaperAutomatico
    print("✅ SUCCESS: CONFIG_AUTOMATICO importado correctamente")
    print(f"📊 CONFIG_AUTOMATICO: {CONFIG_AUTOMATICO}")
    print(f"✅ RewardShaperAutomatico también importable")
    print("🎉 Problema de importación RESUELTO")
except ImportError as e:
    print(f"❌ ERROR: {e}") 