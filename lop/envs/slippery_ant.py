import os
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv


class SlipperyAntEnv(AntEnv, EzPickle):
    """
    SlipperyAnt-v2 (gymnasium version)
    """
    def __init__(self, friction=1.0, xml_file='ant.xml', **kwargs):
        self.friction = friction
        self.custom_xml_file = xml_file
        self.gen_xml_file()
        
        # Initialize with the generated XML file
        AntEnv.__init__(
            self,
            xml_file=self.custom_xml_file,
            ctrl_cost_weight=0.5,
            contact_cost_weight=5e-4,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.2, 1.0),
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=True,
            **kwargs
        )
        EzPickle.__init__(self, friction=friction, xml_file=xml_file, **kwargs)
    
    def gen_xml_file(self):
        # Get the path to the gymnasium ant.xml asset
        import gymnasium.envs.mujoco
        old_file = os.path.join(os.path.dirname(gymnasium.envs.mujoco.__file__), "assets", 'ant.xml')
        # Parse old xml file
        tree = ET.parse(old_file)
        root = tree.getroot()
        # Update friction value - find the <geom> element in worldbody for floor
        for geom in root.iter('geom'):
            if geom.get('name') == 'floor':
                geom.set('friction', f'{self.friction} 0.5 0.5')
                break
        tree.write(self.custom_xml_file)


class SlipperyAntEnv3(AntEnv, EzPickle):
    """
    SlipperyAnt-v3 (gymnasium version, equivalent to Ant-v4)
    """
    def __init__(
        self,
        friction=1.5,
        xml_file="ant.xml",
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        self.friction = friction
        self.custom_xml_file = xml_file
        self.gen_xml_file()
        
        EzPickle.__init__(
            self,
            friction=friction,
            xml_file=xml_file,
            ctrl_cost_weight=ctrl_cost_weight,
            contact_cost_weight=contact_cost_weight,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            contact_force_range=contact_force_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            **kwargs
        )
        
        AntEnv.__init__(
            self,
            xml_file=self.custom_xml_file,
            ctrl_cost_weight=ctrl_cost_weight,
            contact_cost_weight=contact_cost_weight,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            contact_force_range=contact_force_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            **kwargs
        )

    def gen_xml_file(self):
        # Get the path to the gymnasium ant.xml asset
        import gymnasium.envs.mujoco
        old_file = os.path.join(os.path.dirname(gymnasium.envs.mujoco.__file__), "assets", 'ant.xml')
        # Parse old xml file
        tree = ET.parse(old_file)
        root = tree.getroot()
        # Update friction value - find the <geom> element for floor
        for geom in root.iter('geom'):
            if geom.get('name') == 'floor':
                geom.set('friction', f'{self.friction} 0.5 0.5')
                break
        tree.write(self.custom_xml_file)