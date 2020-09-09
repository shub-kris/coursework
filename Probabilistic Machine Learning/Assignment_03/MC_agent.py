import random
import copy
import numpy as np


class MCAgent:
    def __init__(self, ships, size, nb_samples=1000):
        self.nb_samples = nb_samples
        self.observed_board = np.zeros((size, size))
        self.remaining_ships = ships
        self.size = size

    def find_boundary(self, hit):
        i, j = hit
        l = u = j
        while l > 0 and self.observed_board[i, l - 1] != -1:
            l -= 1
        while u < self.size and self.observed_board[i, u] != -1:
            u += 1
        h_bound = (l, u)

        l = u = i
        while l > 0 and self.observed_board[l - 1, j] != -1:
            l -= 1
        while u < self.size and self.observed_board[u, j] != -1:
            u += 1
        v_bound = (l, u)
        return h_bound, v_bound

    def score_line(self, line, bounds, boat_len):
        l, u = bounds
        dif = (u - l) - boat_len
        if dif >= 0:
            pos = np.random.randint(0, dif + 1)
            line[l + pos : l + pos + boat_len] += 1

    def select_observation(self):
        """
        --------------------------------------------------
        THIS IS THE MONTE CARLO SAMPLER YOU NEED TO ADAPT.
        --------------------------------------------------
        
        Select the next location to be observed
        :returns: i_new: int, j_new: int
        """

        # New board to collect the states sampled by the MC agent
        score_board = np.zeros_like(self.observed_board)

        #  +----------+
        #  |  Task 1  |
        #  +----------+
        # Check if there is already an "open" hit, i.e. a ship that has been hit but not sunk
        # These locations are handled by the observation_board as 1

        hits_inds = np.argwhere(self.observed_board == 1)
        unresolved_hits_n = hits_inds.shape[0]

        #  +------------+
        #  |  Task 1a)  |
        #  +------------+
        # If there is already a hit, choose a random one to deal with next.
        # Create a score board including that hit, and reduce the number of samples to 1/10

        # we have open hits
        if unresolved_hits_n != 0:
            hit = hits_inds[np.random.randint(0, unresolved_hits_n)]
            h_bound, v_bound = self.find_boundary(hit)

            samples_n = self.nb_samples // 10
            boat_inds = np.random.randint(0, len(self.remaining_ships), samples_n)
            boat_orientations = np.random.choice([0, 1], samples_n)

            line, bounds = None, None
            for i in range(samples_n):
                b = self.remaining_ships[boat_inds[i]]

                if boat_orientations[i]:  # vertical
                    line = score_board[:, hit[1]]
                    bounds = v_bound
                else:  # horizontal
                    line = score_board[hit[0], :]
                    bounds = h_bound

                self.score_line(line, bounds, b)

        #  +----------+
        #  |  Task 2  |
        #  +----------+
        # Populate the score_board with possible boat placements

        # no open hits
        if unresolved_hits_n == 0:
            samples_n = self.nb_samples
            boat_inds = np.random.choice(
                np.arange(len(self.remaining_ships)), samples_n
            )
            boat_orientations = np.random.choice([0, 1], samples_n)

            obs_line, score_line = None, None
            for i in range(samples_n):
                b = self.remaining_ships[boat_inds[i]]
                posi, posj = np.random.randint(0, self.size - b + 1, 2)

                if boat_orientations[i]:  # vertical
                    obs_line = self.observed_board[posi : posi + b + 1, posj]
                    score_line = score_board[posi : posi + b + 1, posj]
                else:  # horizontal
                    obs_line = self.observed_board[posi, posj : posj + b + 1]
                    score_line = score_board[posi, posj : posj + b + 1]

                if np.all(obs_line != -1):
                    score_line += 1

        # put score 0 to hits and misses
        score_board[self.observed_board != 0] = 0
        # case when all rejections while sampling when no open hits (may happen at very end)
        # other solution is to sample upper part until no rejection
        if np.all(score_board == 0):
            non_zero = np.argwhere(self.observed_board == 0)
            ind = np.random.randint(0, len(non_zero))
            return non_zero[ind, 0], non_zero[ind, 1], None

        #  +----------+
        #  |  Task 3  |
        #  +----------+
        # Having populated the score board, select a new position by choosing the location with the highest score.

        i_new, j_new = np.unravel_index(
            np.argmax(score_board, axis=None), score_board.shape
        )

        return i_new, j_new, score_board

    def update_observations(self, i, j, observation, sunken_ship):
        """
        i:
        j:
        observation:
        """
        if observation:
            self.observed_board[i, j] = 1
        else:
            self.observed_board[i, j] = -1
        if not sunken_ship is None:
            i, j, l, h = sunken_ship
            self.remaining_ships.remove(l)
            if h:
                self.observed_board[i, j : j + l] = -1
            else:
                self.observed_board[i : i + l, j] = -1
