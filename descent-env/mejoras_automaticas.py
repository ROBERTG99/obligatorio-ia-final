
# 🚀 MEJORAS AUTOMÁTICAS GENERADAS - OBJETIVO: -25
# Peor caso detectado: -81.6 (mejora necesaria: +56.6)

class RewardShaperAutomatico:
    """Reward shaper generado automáticamente basado en análisis JSON"""
    
    def __init__(self):
        self.step_count = 0
        self.prev_reward = 0
        self.prev_action = 0
        self.best_error = float('inf')
        self.consecutive_improvements = 0
        
    def shape_reward(self, obs, action, reward, done):
        shaped = reward
        self.step_count += 1
        
        # Métricas actuales
        altitude_error = abs(obs.get('target_altitude', [0])[0] - obs.get('altitude', [0])[0])
        
        # SUPERVIVENCIA MASIVA (detectado problema crítico)
        if not done:
            # Bonificación base por supervivencia
            shaped += 50.0
            
            # Bonificación exponencial por longevidad
            if self.step_count > 100:
                shaped += (self.step_count - 100) ** 1.2 * 0.1
            
            # Bonus por hitos de supervivencia
            if self.step_count % 50 == 0:
                shaped += 500.0
            if self.step_count % 200 == 0:
                shaped += 2000.0
        else:
            # Penalización por muerte temprana
            if self.step_count < 100:
                shaped -= 1000.0
        
        # PRECISIÓN ULTRA AGRESIVA (detectado problema crítico)
        if altitude_error < 0.01:
            shaped += 10000.0  # JACKPOT
        elif altitude_error < 0.05:
            shaped += 5000.0
        elif altitude_error < 0.1:
            shaped += 2000.0
        elif altitude_error < 0.15:
            shaped += 1000.0
        
        # Penalización cúbica por errores grandes
        if altitude_error > 0.1:
            shaped -= (altitude_error ** 3) * 10000
        
        # Bonus por mejora progresiva
        if altitude_error < self.best_error:
            self.best_error = altitude_error
            self.consecutive_improvements += 1
            shaped += 100.0 + (self.consecutive_improvements * 50)
        else:
            self.consecutive_improvements = 0
        
        # CONSISTENCIA (detectado problema variabilidad)
        action_consistency = abs(action - self.prev_action)
        if action_consistency < 0.1:
            shaped += 100.0
        else:
            shaped -= action_consistency * 200
        
        # Smoothing para reducir variabilidad
        shaped = 0.6 * shaped + 0.4 * self.prev_reward
        
        # Actualizar estado
        self.prev_reward = shaped
        self.prev_action = action
        
        return shaped
    
    def reset(self):
        self.step_count = 0
        self.prev_reward = 0
        self.prev_action = 0
        self.best_error = float('inf')
        self.consecutive_improvements = 0

# HIPERPARÁMETROS AUTOMÁTICOS OPTIMIZADOS
CONFIG_AUTOMATICO = {
    'learning_rate': 0.95,           # Ultra agresivo para convergencia rápida
    'epsilon': 0.98,                 # Exploración máxima
    'epsilon_decay': 0.999995,       # Decay ultra lento
    'epsilon_min': 0.15,             # Mantener exploración mínima
    'discount_factor': 0.99999,      # Peso máximo al futuro
    'episodes_fase1': 50000,         # Fase supervivencia
    'episodes_fase2': 100000,        # Fase supervivencia + precisión
    'episodes_fase3': 50000,         # Fase precisión fina
    'step_limit': 5000,              # Límite muy alto para exploración
    'early_stopping_threshold': -15,  # Más permisivo inicialmente
}

# FUNCIÓN DE ENTRENAMIENTO AUTOMÁTICO
def entrenar_con_mejoras_automaticas(env, agent, discretization):
    """Entrenamiento automático con fases optimizadas"""
    
    # Fase 1: Solo supervivencia
    print("🎯 FASE 1: Entrenamiento de supervivencia")
    agent.reward_shaper = RewardShaperSupervivencia()  # Solo supervivencia
    trainer = QLearningTrainer(env, agent, discretization)
    trainer.train(episodes=50000)
    
    # Fase 2: Supervivencia + precisión básica
    print("🎯 FASE 2: Supervivencia + precisión básica")
    agent.reward_shaper = RewardShaperAutomatico()
    trainer.train(episodes=100000)
    
    # Fase 3: Precisión fina
    print("🎯 FASE 3: Precisión fina")
    agent.reward_shaper = RewardShaperPrecisionFina()
    trainer.train(episodes=50000)
    
    return agent

class RewardShaperSupervivencia:
    """Reward shaper solo para supervivencia (Fase 1)"""
    def __init__(self):
        self.step_count = 0
    
    def shape_reward(self, obs, action, reward, done):
        self.step_count += 1
        if not done:
            return reward + 100.0 + (self.step_count * 0.5)
        return reward - 500.0 if self.step_count < 200 else reward
    
    def reset(self):
        self.step_count = 0

class RewardShaperPrecisionFina:
    """Reward shaper para precisión fina (Fase 3)"""
    def shape_reward(self, obs, action, reward, done):
        altitude_error = abs(obs.get('target_altitude', [0])[0] - obs.get('altitude', [0])[0])
        if altitude_error < 0.05:
            return reward + 5000.0
        elif altitude_error < 0.1:
            return reward + 1000.0
        else:
            return reward - (altitude_error ** 2) * 500
    
    def reset(self):
        pass
