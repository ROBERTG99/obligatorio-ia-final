"""
Wrapper para TrainerAgent que implementa la interfaz Agent
"""
from agent import Agent
from trainer_agent import TrainerAgent as OriginalTrainerAgent
from typing import Dict, List

class TrainerAgentWrapper(Agent):
    """Wrapper que hace que TrainerAgent sea compatible con la interfaz Agent"""
    
    def __init__(self, env, difficulty=0.3):
        super().__init__(env)
        self.trainer = OriginalTrainerAgent(env, difficulty)
        self.env = env
        
    def act(self, obs: Dict) -> List[int]:
        """Delega la acción al TrainerAgent original"""
        action = self.trainer.act(obs)
        if action is None:
            # Si no hay acciones válidas, retornar una acción por defecto
            return [0, 0, 0, 0]
        return action 