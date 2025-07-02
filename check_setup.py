#!/usr/bin/env python3
"""
Script para verificar la configuración del proyecto
"""
import sys
import os

def check_dependencies():
    """Verifica las dependencias necesarias"""
    print("Verificando dependencias...")
    
    dependencies = {
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'gymnasium': 'Gymnasium',
        'seaborn': 'Seaborn'
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} instalado")
        except ImportError:
            print(f"✗ {name} NO instalado")
            missing.append(module)
    
    # Verificar BlueSky (opcional)
    try:
        import bluesky
        print("✓ BlueSky instalado (opcional)")
        bluesky_available = True
    except ImportError:
        print("⚠️  BlueSky NO instalado (se usará MockDescentEnv)")
        bluesky_available = False
    
    return missing, bluesky_available

def check_files():
    """Verifica que los archivos necesarios existen"""
    print("\nVerificando archivos del proyecto...")
    
    required_files = {
        'FLAN': [
            'descent-env/flan_qlearning_solution.py',
            'descent-env/demo_flan.py',
            'descent-env/test_flan.py',
            'descent-env/mock_descent_env.py'
        ],
        'BORED': [
            'tactix/bored_solution.py',
            'tactix/demo_bored.py',
            'tactix/trainer_agent_wrapper.py',
            'tactix/tactix_env.py',
            'tactix/agent.py',
            'tactix/random_agent.py'
        ]
    }
    
    all_good = True
    
    for project, files in required_files.items():
        print(f"\n{project}:")
        for file in files:
            if os.path.exists(file):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} NO encontrado")
                all_good = False
    
    return all_good

def main():
    """Función principal"""
    print("="*60)
    print("Verificación de Configuración - Proyectos FLAN y BORED")
    print("="*60)
    
    # Verificar dependencias
    missing_deps, bluesky_available = check_dependencies()
    
    # Verificar archivos
    files_ok = check_files()
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN:")
    print("="*60)
    
    if missing_deps:
        print(f"\n⚠️  Dependencias faltantes: {', '.join(missing_deps)}")
        print("Instalar con: pip3 install " + " ".join(missing_deps))
    else:
        print("\n✓ Todas las dependencias están instaladas")
    
    if not bluesky_available:
        print("\n⚠️  BlueSky no está disponible")
        print("   Los experimentos usarán MockDescentEnv")
        print("   Esto es normal y no afecta la funcionalidad")
    
    if not files_ok:
        print("\n✗ Algunos archivos faltan")
    else:
        print("\n✓ Todos los archivos están presentes")
    
    # Comandos recomendados
    print("\n" + "="*60)
    print("PRÓXIMOS PASOS:")
    print("="*60)
    
    if missing_deps:
        print(f"\n1. Instalar dependencias:")
        print(f"   pip3 install {' '.join(missing_deps)}")
    
    print("\n2. Ejecutar experimentos rápidos:")
    print("   ./run_experiments_m1.sh")
    
    print("\n3. O ejecutar individualmente:")
    print("   - FLAN Demo: cd descent-env && python3 demo_flan.py")
    print("   - BORED Demo: cd tactix && python3 demo_bored.py")
    
    print("\n4. Para experimentos completos:")
    print("   - FLAN: cd descent-env && python3 flan_qlearning_solution.py")
    print("   - BORED: cd tactix && python3 bored_solution.py")
    
    return len(missing_deps) == 0 and files_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 