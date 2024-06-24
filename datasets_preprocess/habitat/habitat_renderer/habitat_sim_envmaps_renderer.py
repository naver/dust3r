# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Render environment maps from 3D meshes using the Habitat Sim simulator.
# --------------------------------------------------------
import numpy as np
import habitat_sim
import math
from habitat_renderer import projections

# OpenCV to habitat camera convention transformation
R_OPENCV2HABITAT = np.stack((habitat_sim.geo.RIGHT, -habitat_sim.geo.UP, habitat_sim.geo.FRONT), axis=0)

CUBEMAP_FACE_LABELS = ["left", "front", "right", "back", "up", "down"]
# Expressed while considering Habitat coordinates systems
CUBEMAP_FACE_ORIENTATIONS_ROTVEC = [
    [0, math.pi / 2, 0],  # Left
    [0, 0, 0],  # Front
                [0, - math.pi / 2, 0],  # Right
                [0, math.pi, 0],  # Back
                [math.pi / 2, 0, 0],  # Up
                [-math.pi / 2, 0, 0],]  # Down

class NoNaviguableSpaceError(RuntimeError):
    def __init__(self, *args):
        super().__init__(*args)

class HabitatEnvironmentMapRenderer:
    def __init__(self,
                 scene,
                 navmesh,
                 scene_dataset_config_file,
                 render_equirectangular=False,
                 equirectangular_resolution=(512, 1024),
                 render_cubemap=False,
                 cubemap_resolution=(512, 512),
                 render_depth=False,
                 gpu_id=0):
        self.scene = scene
        self.navmesh = navmesh
        self.scene_dataset_config_file = scene_dataset_config_file
        self.gpu_id = gpu_id

        self.render_equirectangular = render_equirectangular
        self.equirectangular_resolution = equirectangular_resolution
        self.equirectangular_projection = projections.EquirectangularProjection(*equirectangular_resolution)
        # 3D unit ray associated to each pixel of the equirectangular map
        equirectangular_rays = projections.get_projection_rays(self.equirectangular_projection)
        # Not needed, but just in case.
        equirectangular_rays /= np.linalg.norm(equirectangular_rays, axis=-1, keepdims=True)
        # Depth map created by Habitat are produced by warping a cubemap,
        # so the values do not correspond to distance to the center and need some scaling.
        self.equirectangular_depth_scale_factors = 1.0 / np.max(np.abs(equirectangular_rays), axis=-1)

        self.render_cubemap = render_cubemap
        self.cubemap_resolution = cubemap_resolution

        self.render_depth = render_depth

        self.seed = None
        self._lazy_initialization()

    def _lazy_initialization(self):
        # Lazy random seeding and instantiation of the simulator to deal with multiprocessing properly
        if self.seed == None:
            # Re-seed numpy generator
            np.random.seed()
            self.seed = np.random.randint(2**32-1)
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = self.scene
            if self.scene_dataset_config_file is not None and self.scene_dataset_config_file != "":
                sim_cfg.scene_dataset_config_file = self.scene_dataset_config_file
            sim_cfg.random_seed = self.seed
            sim_cfg.load_semantic_mesh = False
            sim_cfg.gpu_device_id = self.gpu_id

            sensor_specifications = []

            # Add cubemaps
            if self.render_cubemap:
                for face_id, orientation in enumerate(CUBEMAP_FACE_ORIENTATIONS_ROTVEC):
                    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
                    rgb_sensor_spec.uuid = f"color_cubemap_{CUBEMAP_FACE_LABELS[face_id]}"
                    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
                    rgb_sensor_spec.resolution = self.cubemap_resolution
                    rgb_sensor_spec.hfov = 90
                    rgb_sensor_spec.position = [0.0, 0.0, 0.0]
                    rgb_sensor_spec.orientation = orientation
                    sensor_specifications.append(rgb_sensor_spec)

                    if self.render_depth:
                        depth_sensor_spec = habitat_sim.CameraSensorSpec()
                        depth_sensor_spec.uuid = f"depth_cubemap_{CUBEMAP_FACE_LABELS[face_id]}"
                        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
                        depth_sensor_spec.resolution = self.cubemap_resolution
                        depth_sensor_spec.hfov = 90
                        depth_sensor_spec.position = [0.0, 0.0, 0.0]
                        depth_sensor_spec.orientation = orientation
                        sensor_specifications.append(depth_sensor_spec)

            # Add equirectangular map
            if self.render_equirectangular:
                rgb_sensor_spec = habitat_sim.bindings.EquirectangularSensorSpec()
                rgb_sensor_spec.uuid = "color_equirectangular"
                rgb_sensor_spec.resolution = self.equirectangular_resolution
                rgb_sensor_spec.position = [0.0, 0.0, 0.0]
                sensor_specifications.append(rgb_sensor_spec)

                if self.render_depth:
                    depth_sensor_spec = habitat_sim.bindings.EquirectangularSensorSpec()
                    depth_sensor_spec.uuid = "depth_equirectangular"
                    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
                    depth_sensor_spec.resolution = self.equirectangular_resolution
                    depth_sensor_spec.position = [0.0, 0.0, 0.0]
                    depth_sensor_spec.orientation
                    sensor_specifications.append(depth_sensor_spec)

            agent_cfg = habitat_sim.agent.AgentConfiguration(sensor_specifications=sensor_specifications)

            cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
            self.sim = habitat_sim.Simulator(cfg)
            if self.navmesh is not None and self.navmesh != "":
                # Use pre-computed navmesh (the one generated automatically does some weird stuffs like going on top of the roof)
                # See https://youtu.be/kunFMRJAu2U?t=1522 regarding navmeshes
                self.sim.pathfinder.load_nav_mesh(self.navmesh)

            # Check that the navmesh is not empty
            if not self.sim.pathfinder.is_loaded:
                # Try to compute a navmesh
                navmesh_settings = habitat_sim.NavMeshSettings()
                navmesh_settings.set_defaults()
                self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings, True)

            # Check that the navmesh is not empty
            if not self.sim.pathfinder.is_loaded:
                raise NoNaviguableSpaceError(f"No naviguable location (scene: {self.scene} -- navmesh: {self.navmesh})")

            self.agent = self.sim.initialize_agent(agent_id=0)

    def close(self):
        if hasattr(self, 'sim'):
            self.sim.close()

    def __del__(self):
        self.close()

    def render_viewpoint(self, viewpoint_position):
        agent_state = habitat_sim.AgentState()
        agent_state.position = viewpoint_position
        # agent_state.rotation = viewpoint_orientation
        self.agent.set_state(agent_state)
        viewpoint_observations = self.sim.get_sensor_observations(agent_ids=0)

        try:
            # Depth map values have been obtained using cubemap rendering internally,
            # so they do not really correspond to distance to the viewpoint in practice
            # and they need some scaling
            viewpoint_observations["depth_equirectangular"] *= self.equirectangular_depth_scale_factors
        except KeyError:
            pass

        data = dict(observations=viewpoint_observations, position=viewpoint_position)
        return data

    def up_direction(self):
        return np.asarray(habitat_sim.geo.UP).tolist()
    
    def R_cam_to_world(self):
        return R_OPENCV2HABITAT.tolist()
