import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple

class MockDescentEnv(gym.Env):
    """
    Mock environment que simula DescentEnv para compatibilidad.
    Solo para uso como fallback cuando DescentEnv real no esté disponible.
    """
    
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        """Inicializar el entorno mock"""
        
        # Espacios de observación y acción idénticos a DescentEnv
        self.observation_space = spaces.Dict(
            {
                "altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "vz": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "target_altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "runway_distance": spaces.Box(-np.inf, np.inf, dtype=np.float64)
            }
        )
       
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Estado interno simulado
        self.altitude = 3000.0
        self.vz = 0.0
        self.target_altitude = 2000.0
        self.runway_distance = 100.0
        self.step_count = 0
        self.max_steps = 300
        
        # Constantes para normalización (copiadas de DescentEnv)
        self.ALT_MEAN = 1500
        self.ALT_STD = 3000
        self.VZ_MEAN = 0
        self.VZ_STD = 5
        self.RWY_DIS_MEAN = 100
        self.RWY_DIS_STD = 200
        
        # Variables de logging
        self.total_reward = 0
        self.final_altitude = 0

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Generar observación normalizada"""
        obs_altitude = np.array([(self.altitude - self.ALT_MEAN) / self.ALT_STD])
        obs_vz = np.array([(self.vz - self.VZ_MEAN) / self.VZ_STD])
        obs_target_alt = np.array([(self.target_altitude - self.ALT_MEAN) / self.ALT_STD])
        obs_runway_distance = np.array([(self.runway_distance - self.RWY_DIS_MEAN) / self.RWY_DIS_STD])

        return {
            "altitude": obs_altitude,
            "vz": obs_vz,
            "target_altitude": obs_target_alt,
            "runway_distance": obs_runway_distance,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Información adicional para logging"""
        return {
            "total_reward": self.total_reward,
            "final_altitude": self.final_altitude
        }
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset del entorno"""
        super().reset(seed=seed)
        
        # Reset variables de logging
        self.total_reward = 0
        self.final_altitude = 0
        self.step_count = 0
        
        # Estado inicial aleatorio realista
        self.altitude = np.random.uniform(2000, 4000)
        self.target_altitude = self.altitude + np.random.uniform(-500, 500)
        self.vz = np.random.uniform(-2, 2)
        self.runway_distance = np.random.uniform(80, 120)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Ejecutar un paso en el entorno"""
        self.step_count += 1
        
        # Convertir acción a cambio de velocidad vertical
        action_value = float(action[0])
        action_value = np.clip(action_value, -1, 1)
        
        # Simular dinámica simple pero realista
        dt = 1.0  # time step
        
        # Actualizar velocidad vertical basada en la acción
        target_vz = action_value * 12.5  # Factor de escala similar a DescentEnv
        self.vz += 0.3 * (target_vz - self.vz)  # Respuesta gradual
        
        # Actualizar altitud
        self.altitude += self.vz * dt
        
        # Actualizar distancia a la pista
        self.runway_distance -= 2.0  # Velocidad horizontal constante
        
        # Calcular recompensa similar a DescentEnv
        reward = 0.0
        terminated = False
        
        if self.runway_distance > 0 and self.altitude > 0:
            # En vuelo - penalizar diferencia con altitud objetivo
            altitude_error = abs(self.target_altitude - self.altitude)
            reward = -altitude_error / 3000.0 * 5.0
        elif self.altitude <= 0:
            # Crash
            reward = -100.0
            self.final_altitude = -100
            terminated = True
        elif self.runway_distance <= 0:
            # Llegada a la pista
            reward = -self.altitude / 3000.0 * 50.0
            self.final_altitude = self.altitude
            terminated = True
        
        # Terminar si excede pasos máximos
        if self.step_count >= self.max_steps:
            terminated = True
            if self.final_altitude == 0:
                self.final_altitude = self.altitude
        
        self.total_reward += reward
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        """Renderizado (no implementado en mock)"""
        pass
    
    def close(self):
        """Cerrar entorno"""
        pass 