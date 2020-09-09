import random
import copy
import numpy as np


class MCAgent:
    def __init__(self, ships, size, nb_samples=1000):
        self.nb_samples = nb_samples
        self.observed_board = np.zeros((size, size))
        self.remaining_ships = ships
        self.size = size

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

        open_hits = np.argwhere(self.observed_board == 1).tolist()

        #  +------------+
        #  |  Task 1a)  |
        #  +------------+
        # If there is already a hit, choose a random one to deal with next.
        # Create a score board including that hit, and reduce the number of samples to 1/10

        hit = None

        if len(open_hits) > 0:
            rand = np.random.randint(len(open_hits))

            hit = open_hits[rand]

            row, col = hit
            for _ in range(self.nb_samples):
                ship = self.remaining_ships[
                    np.random.randint(len(self.remaining_ships))
                ]

                ship = self.remaining_ships[
                    np.random.randint(len(self.remaining_ships))
                ]

                position = np.random.choice(["hor", "ver"])

                shift = np.random.randint(ship)

                if position == "hor":
                    check = (self.observed_board[row, col - ship : col] == 0).all() or (
                        self.observed_board[row, col : col + ship] == 0
                    ).all()
                    if (
                        0 <= col - shift < self.size
                        and 0 <= col - shift + ship < self.size
                        and check
                    ):
                        col -= shift
                        score_board[row, col : col + ship] += 1

                if position == "ver":
                    check = (self.observed_board[row - ship : row, col] == 0).all() or (
                        self.observed_board[row : row + ship, col] == 0
                    ).all()
                    if (
                        0 <= row - shift < self.size
                        and 0 <= row - shift + ship < self.size
                        and check
                    ):
                        row -= shift
                        score_board[row : row + ship, col] += 1

            self.nb_samples //= 10

        else:
            #  +----------+
            #  |  Task 2  |
            #  +----------+
            # Populate the score_board with possible boat placements
            for _ in range(self.nb_samples):
                ship = self.remaining_ships[
                    np.random.randint(len(self.remaining_ships))
                ]

                position = np.random.choice(["hor", "ver"])

                valid_len = self.size - ship + 1

                row = np.random.randint(valid_len)
                col = np.random.randint(valid_len)

                if (
                    position == "hor"
                    and (self.observed_board[row, col : col + ship] == 0).all()
                ):
                    score_board[row, col : col + ship] += 1

                if (
                    position == "ver"
                    and (self.observed_board[row : row + ship, col] == 0).all()
                ):
                    score_board[row : row + ship, col] += 1

        #  +----------+
        #  |  Task 3  |
        #  +----------+
        # Having populated the score board, select a new position by choosing the location with the highest score.

        scores = [
            (i, j, score_board[i, j])
            for i in range(self.size)
            for j in range(self.size)
        ]
        scores.sort(key=lambda item: -item[2])

        for idx in range(len(scores)):
            i_new, j_new, value = scores[idx]
            if not self.observed_board[i_new, j_new]:
                # print("====================")
                # print(score_board)
                # print(i_new, j_new)
                # print("====================")

                # return the next location to query, i_new: int, j_new: int
                return i_new, j_new, score_board

        return None

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
