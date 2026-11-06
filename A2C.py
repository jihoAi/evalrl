import Agent from Agent

class A2C(Agent):
  def __init__(self,obs_space,action_space,config,device):
    super().__init__(obs_space,action_space,config,device):
    
