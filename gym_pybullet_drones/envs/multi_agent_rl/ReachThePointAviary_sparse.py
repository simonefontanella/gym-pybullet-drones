import os

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.enums import DroneModel, Physics

DRONE_RADIUS = .06
WORLDS_MARGIN = [-20, 60, -10, 10, 0, 10]  # minX maxX minY maxY minZ maxZ
WORLDS_MARGIN_MINUS_DRONE_RADIUS = WORLDS_MARGIN.copy()

# world margin alredy take the radius of the drone
WORLDS_MARGIN_MINUS_DRONE_RADIUS[0] = WORLDS_MARGIN_MINUS_DRONE_RADIUS[0] + DRONE_RADIUS
WORLDS_MARGIN_MINUS_DRONE_RADIUS[1] = WORLDS_MARGIN_MINUS_DRONE_RADIUS[1] - DRONE_RADIUS
WORLDS_MARGIN_MINUS_DRONE_RADIUS[2] = WORLDS_MARGIN_MINUS_DRONE_RADIUS[2] + DRONE_RADIUS
WORLDS_MARGIN_MINUS_DRONE_RADIUS[3] = WORLDS_MARGIN_MINUS_DRONE_RADIUS[3] - DRONE_RADIUS
WORLDS_MARGIN_MINUS_DRONE_RADIUS[4] = WORLDS_MARGIN_MINUS_DRONE_RADIUS[4] + DRONE_RADIUS
WORLDS_MARGIN_MINUS_DRONE_RADIUS[5] = WORLDS_MARGIN_MINUS_DRONE_RADIUS[5] - DRONE_RADIUS

# threasold used to punish drone when it get near to spheres
SPHERES_THRESHOLD = 0.5
BOUNDARIES_THRESHOLD = 0.13
# working dir need to be constant
WORKING_DIR = os.getcwd()


class ReachThePointAviary_sparse(BaseMultiagentAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 2,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 5,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.episode = 0
        self.closest_sphere_distance = {}
        self.prev_sphere_treshold = 0
        self.prev_drones_pos = []
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self.EPISODE_LEN_SEC = 25
        self.last_drones_dist = [1000000 for _ in self.get_agent_ids()]
        self.done_ep = {i: False for i in self.get_agent_ids()}
        self.prev_drones_pos.append(self.INIT_XYZS[0, :])
        self.prev_drones_pos.append(self.INIT_XYZS[1, :])
        self.actual_step_drones_states = np.array([])
        self.drone_has_collided = {i: False for i in range(self.NUM_DRONES)}

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.
        This method is called once per reset, the environment is recreated each time, maybe caching sphere is a good idea(Gyordan)
        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        import csv
        import assignment1 as module_path
        from random import randrange
        p.setAdditionalSearchPath(
            WORKING_DIR + "/shapes/")  # used by loadURDF
        # Override of small_sphere.urdf for changing starting radius to match collision
        # disable shadow for performance
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        n_env = 100
        difficulty = "/medium/"
        if self.episode % 10 == 0:
            env_number = str(randrange(n_env))
            print('CHOOSEN_ENV' + env_number)
            csv_file_path = os.path.dirname(
                module_path.__file__) + "/environment_generator/generated_envs" + difficulty + "{0}/static_obstacles.csv".format(
                "environment_" + env_number)

            with open(csv_file_path, mode='r') as infile:
                reader = csv.reader(infile)
                self.spheres = [[str(rows[0]), float(rows[1]), float(rows[2]), float(rows[3]), float(rows[4])] for rows
                                in
                                reader]

        for sphere in self.spheres:
            temp = p.loadURDF(sphere[0],
                              sphere[1:4:],
                              p.getQuaternionFromEuler([0, 0, 0]),
                              physicsClientId=self.CLIENT,
                              useFixedBase=True,
                              globalScaling=1 * (sphere[4]),

                              # flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                              )
            p.changeVisualShape(temp, -1, rgbaColor=[0, 0, 1, 1])

    ################################################################################

    def step(self,
             action
             ):
        self.actual_step_drones_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        return super().step(action)

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        rewards = {}
        for i in self.get_agent_ids():
            # if self.drone_has_collided[i]:
            #    rewards[i] = 0
            #    continue
            punishment_near_spheres = self.negRewardBaseOnSphereDistance(i)
            pushishment_near_walls = self.negRewardBaseOnTouchBoundary(i)
            # drone has won
            if self.actual_step_drones_states[i, 0] >= WORLDS_MARGIN[1]:
                self.drone_has_collided[i] = True
                rewards[i] = 100
            else:
                if punishment_near_spheres != 0 or pushishment_near_walls != 0:
                    rewards[i] = punishment_near_spheres
                else:
                    rewards[i] = self.rewardBaseOnForward(self.actual_step_drones_states[i, :3],
                                                          self.prev_drones_pos[i],
                                                          self.actual_step_drones_states[i, 10])
            self.prev_drones_pos[i] = self.actual_step_drones_states[i, 0:3]
        return rewards

    ################################################################################
    def rewardBaseOnForward(self, drone_pos, prev_drone_pos, vel_x):
        """Compute the reward based on distance from x, if the drone approaches the target point (x=WORLDS_MARGIN[1]) the reward goes up
        """
        # return -0.5 * np.linalg.norm(np.array([WORLDS_MARGIN[1], drone_pos[1], drone_pos[2]]) - drone_pos)
        # Speed max is self.SPEED_LIMIT, see BaseMultiagentAviary, use min max scaling
        vel_x = vel_x if vel_x <= self.SPEED_LIMIT else self.SPEED_LIMIT
        return ((vel_x - 0) / (self.SPEED_LIMIT - 0)) if (prev_drone_pos[0] < drone_pos[0] and vel_x > 0) else 0

    ################################################################################
    def negRewardBaseOnTouchBoundary(self, drone_id):
        """ If the drones touch world boundary receive a reward of -100
        """
        if self.hit_world(self.actual_step_drones_states[drone_id, 0:3]):
            self.drone_has_collided[drone_id] = True
            return -10
        else:
            return 0

    ################################################################################
    def negRewardBaseOnSphereDistance(self, drone_index):
        """ If the drones touch a sphere receive a reward of -100
        """
        self.closest_sphere_distance[drone_index] = self.getClosestSpheres(
            self.actual_step_drones_states[drone_index, 0:3])
        reward = 0
        for sphere in self.closest_sphere_distance[drone_index]:
            radius = sphere['radius']
            dist = sphere['dist'] - radius - DRONE_RADIUS
            if dist <= SPHERES_THRESHOLD:
                if dist <= 0:
                    # drone collide with a sphere
                    self.drone_has_collided[drone_index] = True
                    reward = -10
                    break
                # the more near the drone gets the more penalty it receive
                reward = -(2 * (SPHERES_THRESHOLD - dist))
                break
        return reward

    ################################################################################
    def reset(self):
        import random
        self.episode += 1
        self.prev_drones_pos = []

        # check WORLDS_MARGIN for this, randomize the spawn position remaining away from sphere spawn area, need to be done before
        # super.reset()
        # self.INIT_XYZS[0] = np.array([random.randrange(2, 3), random.randrange(0, 2), random.randrange(2, 8)])
        # self.INIT_XYZS[1] = np.array([random.randrange(2, 3), random.randrange(0, 1), random.randrange(2, 8)])

        self.INIT_XYZS[0] = np.array([random.randrange(-6, -3), random.randrange(-5, 5), random.randrange(2, 8)])
        self.INIT_XYZS[1] = np.array([random.randrange(-6, -3), random.randrange(-5, 5), random.randrange(2, 8)])

        self.prev_drones_pos.append(self.INIT_XYZS[0, :])
        self.prev_drones_pos.append(self.INIT_XYZS[1, :])
        self.drone_has_collided = {i: False for i in range(self.NUM_DRONES)}
        return super().reset()

    ################################################################################
    def getClosestSpheres(self, drone_pos):
        """
        get the first 10 closest spheres with sphere_x > drone_x
        Return: dictionary of x,y,z distance from drone_pos, sphere radius, x,y,z of sphere, and norm distance
        """
        import operator
        distances = []
        for sphere in self.spheres:
            sphere_x = sphere[1]
            drone_x = drone_pos[0]
            radius = sphere[4]
            if (sphere_x + radius) >= drone_x:
                sphere_y = sphere[2]
                drone_y = drone_pos[1]
                sphere_z = sphere[3]
                drone_z = drone_pos[2]

                distances.append({"x_dist": sphere_x - drone_x,
                                  # TODO non sembrano corrette, le distanze per le componenti possono essere cosi negative
                                  "y_dist": sphere_y - drone_y,
                                  "z_dist": sphere_z - drone_z,
                                  "radius": radius,
                                  "x": sphere_x,
                                  "y": sphere_y,
                                  "z": sphere_z,
                                  "dist": np.linalg.norm(drone_pos - sphere[1:4:])})
        sorted_dist = sorted(distances, key=operator.itemgetter('dist'))

        return sorted_dist[:10] if len(sorted_dist) > 10 else sorted_dist[:len(sorted_dist)]

    ################################################################################
    def hit_world(self, drone_xyz):
        """
        Return: True if the drones touch world boundary False otherwise
        NB:This method alredy take in consideration drone radius
        """
        MIN_X = WORLDS_MARGIN_MINUS_DRONE_RADIUS[0]
        # If reach max_x is won not lost
        # MAX_X = WORLDS_MARGIN_MINUS_DRONE_RADIUS[1]
        MIN_Y = WORLDS_MARGIN_MINUS_DRONE_RADIUS[2]
        MAX_Y = WORLDS_MARGIN_MINUS_DRONE_RADIUS[3]
        MIN_Z = WORLDS_MARGIN_MINUS_DRONE_RADIUS[4]
        MAX_Z = WORLDS_MARGIN_MINUS_DRONE_RADIUS[5]
        # or drone_xyz[0] >= MAX_X \
        if drone_xyz[0] <= MIN_X \
                or MIN_Y >= drone_xyz[1] \
                or drone_xyz[1] >= MAX_Y \
                or MIN_Z >= drone_xyz[2] \
                or drone_xyz[2] >= MAX_Z:
            return True
        return False

    ################################################################################
    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and
            one additional boolean value for key "__all__".

        """

        local_dones = {}
        # Take only the alive drones and then using calculation from reward calculate dones
        for i in self.get_agent_ids().copy():
            if self.drone_has_collided[i]:
                local_dones[i] = True
                self.get_agent_ids().remove(i)
                continue
            else:
                local_dones[i] = False

        local_dones["__all__"] = (True if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC else False) or all(
            [local_dones[k] for k in self.get_agent_ids()])
        return local_dones

    ################################################################################

    def is_drone_hit_spheres(self, drone_index):
        for sphere in self.closest_sphere_distance[drone_index]:
            if sphere['dist'] - sphere['radius'] - DRONE_RADIUS <= 0:
                # self.step_drones_states = self._getDroneStateVector(drone_index)
                return True
        return False

    ################################################################################
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        return {i: {} for i in self.get_agent_ids()}

    ################################################################################

    def _observationSpace(self):
        from gym import spaces
        #             x  y  z r   sphere
        sphere_low = [-1, -1, 0, 0] * 10
        sphere_high = [1, 1, 1, 1] * 10
        low = [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1] + sphere_low
        high = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + sphere_high

        return spaces.Dict({i: spaces.Box(low=np.array(low),
                                          high=np.array(high),
                                          dtype=np.float64
                                          ) for i in range(self.NUM_DRONES)})

    ################################################################################
    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

        """
        ############################################################
        #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
        # return {   i   : self._clipAndNormalizeState(self._getDroneStateVector(i)) for i in range(self.NUM_DRONES) }
        ############################################################
        #### OBS SPACE OF SIZE 52
        obs_52 = np.zeros((self.NUM_DRONES, 52))
        for i in self.get_agent_ids():
            drone_pos = self._getDroneStateVector(i)
            obs = self._clipAndNormalizeState(drone_pos)
            close_sphere = self.getClosestSpheres(drone_pos[0:3])
            normalize_sphere = self.clipAndNormalizeSphere(close_sphere)
            obs_52[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], np.array(normalize_sphere)]).reshape(
                52, )

        return {i: obs_52[i, :] for i in self.get_agent_ids()}

    #     ############################################################

    def clipAndNormalizeSphere(self, spheres):
        # TODO necessita raggio sfera piu raggio drone?
        MAX_MARGIN = np.array([WORLDS_MARGIN[1], WORLDS_MARGIN[3], WORLDS_MARGIN[5]])
        MIN_MARGIN = np.array([WORLDS_MARGIN[0], WORLDS_MARGIN[2], WORLDS_MARGIN[4]])
        MIN_DISTANCE = 0

        MAX_DISTANCE = np.linalg.norm(MAX_MARGIN - MIN_MARGIN) - DRONE_RADIUS
        norm_and_clipped = []

        max_dist_x = abs(WORLDS_MARGIN[0]) + abs(WORLDS_MARGIN[1])
        max_dist_y = abs(WORLDS_MARGIN[2]) + abs(WORLDS_MARGIN[3])
        max_dist_z = abs(WORLDS_MARGIN[4] + abs(WORLDS_MARGIN[5]))
        for s in spheres:
            clipped_pos_x = np.clip(s['x_dist'] - s["radius"] - DRONE_RADIUS,  # posizione
                                    -max_dist_x + s["radius"] + DRONE_RADIUS,  # min distanza con segno
                                    max_dist_x - s["radius"] - DRONE_RADIUS)  # max distanza con segno
            clipped_pos_y = np.clip(s['y_dist'] - s["radius"] - DRONE_RADIUS,
                                    -max_dist_y + s["radius"] + DRONE_RADIUS,
                                    max_dist_y - s["radius"] - DRONE_RADIUS)
            clipped_pos_z = np.clip(s['z_dist'] - s["radius"] - DRONE_RADIUS,
                                    -max_dist_z + s["radius"] + DRONE_RADIUS,
                                    max_dist_z - s["radius"] - DRONE_RADIUS)

            normalized_x = clipped_pos_x / (max_dist_x - s["radius"] - DRONE_RADIUS)
            normalized_y = clipped_pos_y / (max_dist_y - s["radius"] - DRONE_RADIUS)
            normalized_z = clipped_pos_z / (max_dist_z - s["radius"] - DRONE_RADIUS)
            clipped_distance = np.clip(s["dist"] - s["radius"] - DRONE_RADIUS, MIN_DISTANCE, MAX_DISTANCE - s["radius"])
            normalized_dist = clipped_distance / (MAX_DISTANCE - s["radius"])
            norm_and_clipped.extend([normalized_x, normalized_y, normalized_z, normalized_dist])

        return norm_and_clipped

    ################################################################################
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        # Check BaseMultiagentAviary to increase the maximum speed
        MAX_LIN_VEL_XY = self.SPEED_LIMIT
        MAX_LIN_VEL_Z = self.SPEED_LIMIT
        MIN_X = WORLDS_MARGIN[0]
        MAX_X = WORLDS_MARGIN[1]
        MIN_Y = WORLDS_MARGIN[2]
        MAX_Y = WORLDS_MARGIN[3]
        MIN_Z = WORLDS_MARGIN[4]
        MAX_Z = WORLDS_MARGIN[5]

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_x = np.clip(state[0], MIN_X, MAX_X)
        clipped_pos_y = np.clip(state[1], MIN_Y, MAX_Y)
        clipped_pos_z = np.clip(state[2], MIN_Z, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_x,
                                               clipped_pos_y,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_x = clipped_pos_x / MAX_X
        normalized_pos_y = clipped_pos_y / MAX_Y
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_x,
                                      normalized_pos_y,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20, )

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_x,
                                      clipped_pos_y,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_x == np.array(state[0])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped x position {:.2f}".format(
                      state[0]))
        if not (clipped_pos_y == np.array(state[1])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped y position {:.2f}".format(
                      state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                      state[7], state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                      state[10], state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
