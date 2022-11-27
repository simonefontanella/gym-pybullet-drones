import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary

WORLDS_MARGIN = [-20, 60, -10, 10, 0, 10]  # minX maxX minY maxY minZ maxZ


# minima -20 -10 0
# massima 60 0 10
class ReachThePointAviary(BaseMultiagentAviary):
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
                 aggregate_phy_steps: int = 1,
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
        self.last_drones_dist = [1000000 for _ in range(self.NUM_DRONES)]
        self.closest_sphere_distance = {}
        self.prev_sphere_treshold = 0
        self.prev_drones_pos = []
        self.prev_drones_pos.append(self.INIT_XYZS[0, :])
        self.prev_drones_pos.append(self.INIT_XYZS[1, :])
        self.done_ep = {i: False for i in range(self.NUM_DRONES)}
        self.DRONE_RADIUS = .06
        self.spheres = []
    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.
        This method is called once per reset, the environment is recreated each time, maybe caching sphere is a good idea(Gyordan)
        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        import pybullet as p
        import csv
        import os
        import assignment1 as module_path
        from random import randrange

        # n_env = 100
        # difficulty = "/medium/"
        # if self.episode % 10 == 0:
        #     env_number = str(randrange(n_env))
        #     # env_number = "0"
        #     csv_file_path = os.path.dirname(
        #         module_path.__file__) + "/environment_generator/generated_envs" + difficulty + "{0}/static_obstacles.csv".format(
        #         "environment_" + env_number)
        #
        #     with open(csv_file_path, mode='r') as infile:
        #         reader = csv.reader(infile)
        #         # prefab_name,pos_x,pos_y,pos_z,radius
        #         self.spheres = [[str(rows[0]), float(rows[1]), float(rows[2]), float(rows[3]), float(rows[4])] for rows
        #                         in
        #                         reader]
        #
        # for sphere in self.spheres:
        #     temp = p.loadURDF(sphere[0],
        #                       sphere[1:4:],
        #                       p.getQuaternionFromEuler([0, 0, 0]),
        #                       physicsClientId=self.CLIENT,
        #                       useFixedBase=True,
        #                       globalScaling=10 * sphere[4],
        #                       #flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        #                       )
        #     p.changeVisualShape(temp, -1, rgbaColor=[0, 0, 1, 1])

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        rewards = {}
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(0, self.NUM_DRONES):
            x_vel = states[i, 10]
            rewardForward = self.rewardBaseOnForward(states[i, :3], self.prev_drones_pos[i][0], x_vel)  # states[i, 0], self.prev_drones_pos[i][0],
            # states[i, 1], self.prev_drones_pos[i][1],
            # states[i, 2], self.prev_drones_pos[i][2],
            # x_vel
            rewardSphere = self.rewardBaseOnSphereDistance(states[i, 0:3], i)
            #rewardsBoundary = self.rewardBaseOnTouchBoundary(states[i, 0:3])
            rewards[i] = rewardForward #+ rewardSphere + rewardsBoundary
            self.prev_drones_pos[i] = states[i, 0:3]
        return rewards

    def rewardBaseOnForward(self, drone_pos, x_vel, prev_x):  # x, prev_x, y, prev_y, z, prev_z, x_vel
        reward = 0

        # if drone_pos[0] > prev_x:
        #      reward += 10 + (10 * x_vel)
        # #if x >= 60 :
        return -1 * np.linalg.norm(np.array([60, 0, 1]) - drone_pos) ** 2

        # if y > 8 or y < -8:
        #     reward -= 0.05
        # if z <= 2:
        #     reward -= 0.05
        return reward

    # reward
    # x -0.02
    # closest sphere
    # 0.001
    # 0.003
    # 0.009

    def rewardBaseOnTouchBoundary(self, drone_xyz):
        return -10 if self.hit_world(drone_xyz) else 0

    def rewardBaseOnSphereDistance(self, drone_xyz, drone_index):
        self.closest_sphere_distance[drone_index] = self.getClosestSpheres(drone_xyz)
        # controllo se il drone tocca la sfera piu vicina a lui, distanza dal drone - raggio minore di una distanza fissata

        reward = 0

        for sphere in self.closest_sphere_distance[drone_index]:
            radius = sphere['radius']
            treshold = sphere['dist'] - radius - self.DRONE_RADIUS
            # questo in caso il drone salga tøroppo in alto
            # if sphere['z-dist'] < 0 or (sphere['y-dist'] < 0):
            # reward*=sphere['z-dist']

            # if 3 > treshold > 2:
            #    reward -= 0.003
            # if 2 > treshold > 1:
            #    reward -= 0.006
            # if 0.5 > treshold < 0.3:
            #     reward -= 0.003
            # if 0.3 > treshold < 0.2:
            #     reward -= 0.005
            # if 0.2 > treshold < 0.1:
            #     reward -= 0.007
            # if treshold < 0.5:
            #    reward -= 0.05 + (0.05 * 1-treshold)
            # if treshold < 0.01:
            #   reward -= 0.05 + (0.05 * 1-treshold)
            if treshold <= 0.01:
                reward -= 10

        #  if (self.prev_sphere_treshold + 0.1) <= treshold:
        #      reward += 0.005
        # self.prev_sphere_treshold=treshold

        return reward

    # sphere [ [[x:3, y:4, z,: , r:432, distance]},{x:3, y:4, z,: , r:432, distance],{x:3, y:4, z,: , r:432, distance}
    #     drone0 -> [{x:0, y:1, z:2, distance:23},
    # 	   {x:3, y:4, z:1, distance:22}
    # 	   ]
    # Le sfere sono gli indici degli array, ogni sfera ha la distanza dal drone
    #     ]
    #

    def reset(self):
        self.episode += 1
        # resetMethod = super().reset()
        self.prev_drones_pos = []
        self.prev_drones_pos.append(self.INIT_XYZS[0, :])
        self.prev_drones_pos.append(self.INIT_XYZS[1, :])
        self.done_ep = {i: False for i in range(self.NUM_DRONES)}
        return super().reset()

    def getClosestSpheres(self, drone_pos):
        import operator
        distances = []
        for sphere in self.spheres:
            sphere_x = sphere[1]
            drone_x = drone_pos[0]
            if sphere_x >= drone_x:
                sphere_y = sphere[2]
                drone_y = drone_pos[1]
                sphere_z = sphere[3]
                drone_z = drone_pos[2]
                radius = sphere[4]
                distances.append({"x_dist": sphere_x - drone_x - radius,
                                  "y_dist": sphere_y - drone_y - radius,
                                  "z_dist": sphere_z - drone_z - radius,
                                  "radius": radius,
                                  "dist": np.linalg.norm(drone_pos - sphere[1:4:])})
        sorted_dist = sorted(distances, key=operator.itemgetter('dist'))

        return sorted_dist[:10] if len(sorted_dist) > 10 else sorted_dist[:len(sorted_dist)]

    ################################################################################
    def hit_world(self, drone_xyz):
        MIN_X = WORLDS_MARGIN[0]
        MAX_X = WORLDS_MARGIN[1]
        MIN_Y = WORLDS_MARGIN[2]
        MAX_Y = WORLDS_MARGIN[3]
        MIN_Z = WORLDS_MARGIN[4]
        MAX_Z = WORLDS_MARGIN[5]

        if drone_xyz[0] <= MIN_X \
                or drone_xyz[0] >= MAX_X \
                or MIN_Y >= drone_xyz[1] \
                or drone_xyz[1] >= MAX_Y \
                or MIN_Z >= drone_xyz[2] \
                or drone_xyz[2] >= MAX_Z:
            return True
        return False

    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and
            one additional boolean value for key "__all__".

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        for i in range(self.NUM_DRONES):
            if self.hit_world(states[i, :3]):
                self.done_ep[i] = True
                continue
            if self.is_drone_hit_spheres(i):
                self.done_ep[i] = True
                continue
            if not self.done_ep[i]:
                self.done_ep[i] = True if self.step_counter / self.SIM_FREQ > (self.EPISODE_LEN_SEC + 30) else False

        self.done_ep["__all__"] = all([self.done_ep.get(k) for k in range(self.NUM_DRONES)])
        return self.done_ep

    ################################################################################

    def is_drone_hit_spheres(self, drone_index):
        for sphere in self.closest_sphere_distance[drone_index]:
            if sphere['dist'] - sphere['radius'] <= 0:
                return True
        return False

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################

    # def _observationSpace(self):
    #     # aggiungere 4 numeri per la dimensione delle sfere x 10 volte che sono le sfere
    #     from gym import spaces
    #     #             x  y  z r   sphere
    #     sphere_low = [0, 0, 0] * 10
    #     sphere_high = [1, 1, 1] * 10
    #     low = [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1] + sphere_low
    #     high = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + sphere_high
    #
    #     return spaces.Dict({i: spaces.Box(low=np.array(low),
    #                                       high=np.array(high),
    #                                       dtype=np.float32
    #                                       ) for i in range(self.NUM_DRONES)})
    #
    # def _computeObs(self):
    #     """Returns the current observation of the environment.
    #
    #     Returns
    #     -------
    #     dict[int, ndarray]
    #         A Dict with NUM_DRONES entries indexed by Id in integer format,
    #         each a Box() os shape (H,W,4) or (12,) depending on the observation type.
    #
    #     """
    #     ############################################################
    #     #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
    #     # return {   i   : self._clipAndNormalizeState(self._getDroneStateVector(i)) for i in range(self.NUM_DRONES) }
    #     ############################################################
    #     #### OBS SPACE OF SIZE 52
    #     obs_52 = np.zeros((self.NUM_DRONES, 42))
    #     for i in range(self.NUM_DRONES):
    #         drone_pos = self._getDroneStateVector(i)
    #         obs = self._clipAndNormalizeState(drone_pos)
    #         close_sphere = self.getClosestSpheres(drone_pos[0:3])
    #         normalize_sphere = self.clipAndNormalizeSphere(close_sphere)
    #         obs_52[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], np.array(normalize_sphere)]).reshape(42, )
    #     return {i: obs_52[i, :] for i in range(self.NUM_DRONES)}
    #     ############################################################

    def clipAndNormalizeSphere(self, spheres):
        MIN_X = 0
        MAX_X = 80 - 3 - self.DRONE_RADIUS
        MIN_Y = 0
        MAX_Y = 20 - 3 - self.DRONE_RADIUS
        MIN_Z = 0
        MAX_Z = 10 - 3 - self.DRONE_RADIUS

        norm_and_clipped = []
        for s in spheres:
            # da 0 a 80 - 3 - raggio drone
            clipped_pos_x = np.clip(s["x_dist"], MIN_X, MAX_X)
            clipped_pos_y = np.clip(s["y_dist"], MIN_Y, MAX_Y)
            clipped_pos_z = np.clip(s["z_dist"], MIN_Z, MAX_Z)

            normalized_x = clipped_pos_x / MAX_X
            normalized_y = clipped_pos_y / MAX_Y
            normalized_z = clipped_pos_z / MAX_Z

            norm_and_clipped.extend([normalized_x, normalized_y, normalized_z])
        return norm_and_clipped

        # x_dists = np.array([s['x_dist'] for s in spheres]).reshape(-1, 1)
        # y_dists = np.array([s['y_dist'] for s in spheres]).reshape(-1, 1)
        # z_dists = np.array([s['z_dist'] for s in spheres]).reshape(-1, 1)
        # d_dists = np.array([s['dist'] for s in spheres]).reshape(-1, 1)
        #
        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler()
        # normalized_x = scaler.fit_transform(x_dists)
        # normalized_z = scaler.fit_transform(z_dists)
        #
        # normalized_y = scaler.fit_transform(y_dists)
        #
        # normalized_d = scaler.fit_transform(d_dists)
        #
        # con = np.ravel(np.concatenate((normalized_x, normalized_y, normalized_z, normalized_d)))
        # return con
        # x_dists = [s['x_dist'] for s in spheres]
        # print(x_dists)
        # normalized_x = x_dists / np.linalg.norm(x_dists)
        # L = (L - np.min(L)) / (np.max(L) - np.min(L))
        #
        # y_dists = [s['y_dist'] for s in spheres]
        # normalized_y = y_dists / np.linalg.norm(y_dists)
        #
        # z_dists = [s['z_dist'] for s in spheres]
        # normalized_z = z_dists / np.linalg.norm(z_dists)
        #
        # d_dists = [s['dist'] for s in spheres]
        # normalized_d = d_dists / np.linalg.norm(d_dists)
        #
        # con = np.stack((normalized_x, normalized_y,normalized_z,normalized_d), axis=1)
        # print(con)
        # exit()
        # return [normalized_x, normalized_y, normalized_z, normalized_d]

        # for s in spheres:
        #     clipped_dist_x = np.clip(s["x_dist"], MIN_X, MAX_X)
        #     # print(f"s[y] {y}")
        #     clipped_dist_y = np.clip(s["y_dist"], MIN_Y, MAX_Y)
        #     # print(f"clipped_pos_y {clipped_pos_y}")
        #     clipped_pos_z = np.clip(s["z_dist"], MIN_Z, MAX_Z)
        #     clipped_r = np.clip(s["radius"], MIN_R, MAX_R)
        #     normalized_x = clipped_pos_x / MAX_X
        #     normalized_y = clipped_pos_y / MAX_Y
        #     normalized_z = clipped_pos_z / MAX_Z
        #     normalized_r = clipped_r / MAX_R
        #     # print(f"normalized_x {normalized_x}")
        #     # print(f"normalized_y {normalized_y}")
        #     # print(f"normalized_z {normalized_z}")
        #     # print(f"RAGGIO {normalized_r}")
        #     # res_norm=np.array(np.hstack([normalized_x,
        #     #                        normalized_y,
        #     #                   normalized_z,
        #     #                       normalized_r
        #     #                       ]).reshape(4, ))
        #     # if len(norm_and_clipped) == 0:
        #     #     norm_and_clipped = res_norm
        #     # else:
        #     #     norm_and_clipped+=res_norm
        #     norm_and_clipped.extend([normalized_x, normalized_y, normalized_z, normalized_r])
        # return norm_and_clipped

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
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1
        #
        # MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        # MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

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
