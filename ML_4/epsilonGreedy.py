import math
import random
import csv
import numpy as np

def setup_env():
	env = np.zeros((10, 10))

	neg_coord = [(3,3),(4,5),(4,6),(5,6),(5,8),(6,8),(7,3),(7,5),(7,6)]
	wall_coord = [(2,1),(2,2),(2,3),(2,4), (2,6), (2,7), (2,8), (3,4), (4,4), (5,4), (6,4), (7,4)]
	goal_coord = [(5,5)]

	for c in wall_coord:
		env[c] = 999
	
	for c in neg_coord:
		env[c] = -1

	for c in goal_coord:
		env[c] = 1

	print(env)
	return env


def calculateNextState(currentState):
	#todo 


def greedyQLearning(currentState, epsilon):
	num_of_steps = 0

	while(currentState != goalState):
		print("IN")
		possibleActions = []
		possNextState = [0] * 4
		nextState = 0
		r = 0.0
		qMax = 0
		qMaxExploit = 0

		possNextState = calculateNextState(currentState)

		#Determining the next state
		for action in range(len(R[currentState])):
			if (R[currentState][action] != 999.0):
				possibleActions.append(action)

		#getting random number between 0 and 1 and checking with epsilon
		if (len(possibleActions) > 0):
			r = random.uniform(0, 1)

			# exploit
			if r <= epsilon:
				print ("exploit: ", possNextState)
				for action in range(0, len(R[currentState])):
					if (R[currentState][action] != 999 and qMaxExploit <= Q[currentState][action]):
						qMaxExploit = Q[currentState][action]
						actionIndex = action
				
				nextState = possNextState[actionIndex]

			# explore
			else:
				print ("explore: ", possNextState)
				actionIndex = possibleActions[(random.randrange(len(possibleActions)))]
				nextState = possNextState[actionIndex]

			print ("nextState: ",nextState)

			for action in range(0, len(R[nextState])):
					if (R[nextState][action] != 999 and qMax <= Q[nextState][action]):
						qMax = Q[nextState][action]

			# Q calculation
			Q[currentState][actionIndex] = Q[currentState][actionIndex] + alpha * (R[currentState][actionIndex] + (beta * qMax) - Q[currentState][actionIndex])
			currentState = nextState

			num_of_steps += 1

	print ("\nnum_of_steps: ", num_of_steps)
	print('-'*50)

	print("-----Q-----")
	for i in range(len(Q)):
		print(Q[i])


if __name__ == "__main__":

	# discount factor
	beta = 0.9 
	# learning rate
	alpha = 0.01
	# for greedy algorithm
	epsilons = [0.1, 0.2, 0.3]

	R = setup_env()

	# goal state
	goalState = (5,5)
	# starting state
	startState = (0,0)

	for e in epsilons:
		#storing the q values
		Q = [[0.0] * 4 for j in range(100)]		
		greedyQLearning(startState, e)