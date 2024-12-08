from math import cos, sin

import numpy as np


class DiscretizedAcrobot:
    LINK_LENGTH_1 = 1.0  # [m]
    LINK_LENGTH_2 = 1.0  # [m]
    LINK_MASS_1 = 1.0  #: [kg] mass of link 1
    LINK_MASS_2 = 1.0  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.0  #: moments of inertia for both links
    AVAIL_TORQUE = [-1, 0, 1]
    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    def __init__(
        self,
        angular_resolution_rad=0.01,
        angular_vel_resolution_rad_per_sec=0.05,
        angle_bins=None,
        velocity_bins=None,
        precomputed_P=None,
        verbose=False,
    ):
        self.verbose = verbose

        if angle_bins is None:
            self.angle_1_bins = int(np.pi // angular_resolution_rad) + 1
            self.angle_2_bins = int(np.pi // angular_resolution_rad) + 1
        else:
            self.angle_1_bins = angle_bins
            self.angle_2_bins = angle_bins

        if velocity_bins is None:
            self.angular_vel_1_bins = (
                int((2 * self.MAX_VEL_1) // angular_vel_resolution_rad_per_sec) + 1
            )
            self.angular_vel_2_bins = (
                int((2 * self.MAX_VEL_2) // angular_vel_resolution_rad_per_sec) + 1
            )
        else:
            self.angular_vel_1_bins = velocity_bins
            self.angular_vel_2_bins = velocity_bins

        self.n_states = (
            self.angle_1_bins
            * self.angle_2_bins
            * self.angular_vel_1_bins
            * self.angular_vel_2_bins
        )

        self.angle_range = (-np.pi, np.pi)
        self.velocity_1_range = (-self.MAX_VEL_1, self.MAX_VEL_1)
        self.velocity_2_range = (-self.MAX_VEL_2, self.MAX_VEL_2)
        self.action_space = len(self.AVAIL_TORQUE)  # -1, 0, 1
        self.dt = 0.2

        if precomputed_P is None:
            self.P = {
                state: {action: [] for action in range(self.action_space)}
                for state in range(self.n_states)
            }
            self.setup_transition_probabilities()

        else:
            self.P = precomputed_P

        # add transform_obs
        self.transform_obs = lambda obs: (
            np.ravel_multi_index(
                (
                    np.clip(
                        np.digitize(
                            obs[0], np.linspace(*self.angle_range, self.angle_1_bins)
                        )
                        - 1,
                        0,
                        self.angle_1_bins - 1,
                    ),
                    np.clip(
                        np.digitize(
                            obs[1], np.linspace(*self.angle_range, self.angle_2_bins)
                        )
                        - 1,
                        0,
                        self.angle_2_bins - 1,
                    ),
                    np.clip(
                        np.digitize(
                            obs[2],
                            np.linspace(
                                *self.velocity_1_range, self.angular_vel_1_bins
                            ),
                        )
                        - 1,
                        0,
                        self.angular_vel_1_bins - 1,
                    ),
                    np.clip(
                        np.digitize(
                            obs[3],
                            np.linspace(
                                *self.velocity_2_range, self.angular_vel_2_bins
                            ),
                        )
                        - 1,
                        0,
                        self.angular_vel_2_bins - 1,
                    ),
                ),
                (
                    self.angle_1_bins,
                    self.angle_2_bins,
                    self.angular_vel_1_bins,
                    self.angular_vel_2_bins,
                ),
            )
        )

    def setup_transition_probabilities(self):
        """
        Sets up the transition probabilities for the environment. This method iterates through all possible
        states and actions, simulates the next state, and records the transition probability
        (deterministic in this setup), reward, and termination status.
        """
        percent = 0

        for state in range(self.n_states):
            angle_1, angle_2, vel_1, vel_2 = self.index_to_state(state)
            for action in range(self.action_space):
                next_state, reward, done = self.compute_next_state(
                    angle_1, angle_2, vel_1, vel_2, action
                )
                self.P[state][action].append((1, next_state, reward, done))

            if state % int(self.n_states / 10) == 0 and state > 0 and self.verbose:
                percent += 10
                print(f"{percent}% of probabilities calculated!")

    def index_to_state(self, index):
        """
        Converts a single index into a multidimensional state representation.

        Parameters:
        - index (int): The flat index representing the state.

        Returns:
        - list: A list of indices representing the state in terms of position, velocity, angle, and angular velocity bins.
        """

        totals = [
            self.angle_1_bins,
            self.angle_2_bins,
            self.angular_vel_1_bins,
            self.angular_vel_2_bins,
        ]
        multipliers = np.cumprod([1] + totals[::-1])[:-1][::-1]
        components = [int((index // multipliers[i]) % totals[i]) for i in range(4)]
        return components

    # Modified from original implementation of Gym Acrobot
    def dsdt(self, state):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = state[-1]
        s = state[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - np.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - np.pi / 2)
            + phi2
        )

        # the following line is consistent with the java implementation and the
        # book
        ddtheta2 = (
            a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2
        ) / (m2 * lc2**2 + I2 - d2**2 / d1)

        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0

    # Modified from original implementation of Gym Acrobot
    def rk4(self, derivs, y0, t):
        """
        Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.

        Example for 2D system:

            >>> def derivs(x):
            ...     d1 =  x[0] + 2*x[1]
            ...     d2 =  -3*x[0] + 4*x[1]
            ...     return d1, d2

            >>> dt = 0.0005
            >>> t = np.arange(0.0, 2.0, dt)
            >>> y0 = (1,2)
            >>> yout = rk4(derivs, y0, t)

        Args:
            derivs: the derivative of the system and has the signature `dy = derivs(yi)`
            y0: initial state vector
            t: sample times

        Returns:
            yout: Runge-Kutta approximation of the ODE
        """

        try:
            Ny = len(y0)
        except TypeError:
            yout = np.zeros((len(t),), np.float64)
        else:
            yout = np.zeros((len(t), Ny), np.float64)

        yout[0] = y0

        for i in np.arange(len(t) - 1):
            this = t[i]
            dt = t[i + 1] - this
            dt2 = dt / 2.0
            y0 = yout[i]

            k1 = np.asarray(derivs(y0))
            k2 = np.asarray(derivs(y0 + dt2 * k1))
            k3 = np.asarray(derivs(y0 + dt2 * k2))
            k4 = np.asarray(derivs(y0 + dt * k3))
            yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        # We only care about the final timestep and we cleave off action value which will be zero
        return yout[-1][:4]

    def compute_next_state(
        self, angle_1_idx, angle_2_idx, vel_1_idx, vel_2_idx, action
    ):
        angle_space = np.linspace(*self.angle_range, self.angle_1_bins)
        velocity_1_space = np.linspace(*self.velocity_1_range, self.angular_vel_1_bins)
        velocity_2_space = np.linspace(*self.velocity_2_range, self.angular_vel_2_bins)

        angle_1 = angle_space[angle_1_idx]
        angle_2 = angle_space[angle_2_idx]

        velocity_1 = velocity_1_space[vel_1_idx]
        velocity_2 = velocity_2_space[vel_2_idx]

        torque = self.AVAIL_TORQUE[action]

        state = [angle_1, angle_2, velocity_1, velocity_2, torque]

        new_state = self.rk4(self.dsdt, state, [0, self.dt])

        new_state[0] = self.wrap(new_state[0], -np.pi, np.pi)
        new_state[1] = self.wrap(new_state[1], -np.pi, np.pi)
        new_state[2] = self.bound(new_state[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        new_state[3] = self.bound(new_state[3], -self.MAX_VEL_2, self.MAX_VEL_2)

        # index new state

        new_angle_1_idx = np.clip(
            np.digitize(new_state[0], np.linspace(*self.angle_range, self.angle_1_bins))
            - 1,
            0,
            self.angle_1_bins - 1,
        )
        new_angle_2_idx = np.clip(
            np.digitize(new_state[1], np.linspace(*self.angle_range, self.angle_2_bins))
            - 1,
            0,
            self.angle_2_bins - 1,
        )
        new_vel_1_idx = np.clip(
            np.digitize(
                new_state[2],
                np.linspace(*self.velocity_1_range, self.angular_vel_1_bins),
            )
            - 1,
            0,
            self.angular_vel_1_bins - 1,
        )
        new_vel_2_idx = np.clip(
            np.digitize(
                new_state[3],
                np.linspace(*self.velocity_2_range, self.angular_vel_2_bins),
            )
            - 1,
            0,
            self.angular_vel_2_bins - 1,
        )

        new_state_idx = np.ravel_multi_index(
            (new_angle_1_idx, new_angle_2_idx, new_vel_1_idx, new_vel_2_idx),
            (
                self.angle_1_bins,
                self.angle_2_bins,
                self.angular_vel_1_bins,
                self.angular_vel_2_bins,
            ),
        )

        # self.state = new_state
        terminated = bool(-cos(new_state[0]) - cos(new_state[1] + new_state[0]) > 1.0)
        reward = -1.0 if not terminated else 0.0

        return new_state_idx, reward, terminated

    def wrap(self, x, m, M):
        """Wraps `x` so m <= x <= M; but unlike `bound()` which
        truncates, `wrap()` wraps x around the coordinate system defined by m,M.\n
        For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

        Args:
            x: a scalar
            m: minimum possible value in range
            M: maximum possible value in range

        Returns:
            x: a scalar, wrapped
        """
        diff = M - m
        while x > M:
            x = x - diff
        while x < m:
            x = x + diff
        return x

    def bound(self, x, m, M=None):
        """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
        have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

        Args:
            x: scalar
            m: The lower bound
            M: The upper bound

        Returns:
            x: scalar, bound between min (m) and Max (M)
        """
        if M is None:
            M = m[1]
            m = m[0]
        # bound x between min (m) and Max (M)
        return min(max(x, m), M)
