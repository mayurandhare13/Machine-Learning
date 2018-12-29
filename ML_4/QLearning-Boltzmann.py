import math
import random
import csv
import numpy as np


def setup_env():
	env = np.zeros((100, 4))

	neg_coord = [33, 45, 46, 56, 58, 68, 73, 75, 76]
	wall_coord = [21, 22, 23, 24, 26, 27, 28, 34, 44, 54, 64, 74]
	goal_coord = [55]

	for i in range(100):
		if (i < 10):
			env[i][0] = 999
		if (i > 89):
			env[i][1] = 999
		if (i % 10) == 0:
			env[i][2] = 999
		if(i % 10) == 9:
			env[i][3] = 999

	for n in neg_coord:
		env[n-1][3] = -1
		env[n-10][1] = -1
		env[n+1][2] = -1
		env[n+10][0] = -1
	
	for n in wall_coord:
		env[n-1][3] = 999
		env[n-10][1] = 999
		env[n+1][2] = 999
		env[n+10][0] = 999

	for n in goal_coord:
		env[n-1][3] = 1
		env[n-10][1] = 1
		env[n+1][2] = 1
		env[n+10][0] = 1

	return env

def calculateNextState(startState):

	possNextState = [0 for i in range(0,4)]	
	
	for i in range(0,4):

		if (i == 0): 
			if(startState - 10 >= 0):
				possNextState[i] = startState - 10
			else:
				possNextState[i] = -1

		if (i == 1):
			if(startState + 10 < 100):
				possNextState[i] = startState + 10
			else:
				possNextState[i] = -1	

		if (i == 2):
			if(startState - 1 >= 0):
				possNextState[i] = startState - 1
			else:
				possNextState[i] = -1

		if (i == 3):
			if(startState + 1 < 100):
				possNextState[i] = startState + 1
			else:
				possNextState[i] = -1

	return possNextState


def getBoltzmannProb(state):

	possNextState = calculateNextState(state)
	actionsProb = [0 for i in range(0,len(possNextState))]
	denominator = 0.0
	numerator = 0.0

	global temperature
	temperature = temperature - 0.05 #need to calculate the t value using temperature

	for action in range(0,len(possNextState)):
		if (possNextState[action] != -1):
			denominator = denominator + math.exp(Q[state][action] / temperature) # temperature

	for action in range(0,len(possNextState)):
		if (possNextState[action] != -1):
			prob = 0.0
			#numerator = math.exp((Q[possNextState[action]][action])/temperature)
			numerator = math.exp((Q[state][action])/temperature)
			if(denominator != 0):
				prob = numerator / denominator
			
			actionsProb[action] = prob

	return actionsProb


def greedyQLearningBoltzmann(startState):

	num_of_steps = 0

	while(startState != goalState):
		
		possibleActions = []
		actionsProb = []
		possNextState = [0 for i in range(0,4)]
		nextState = 0
		r = 0.0
		qMax = 0
		qMaxExploit = 0

		possNextState = calculateNextState(startState)

		for action in range(0, len(R[startState])):
			if (R[startState][action] != 999.0):
				possibleActions.append(action)

		if (len(possibleActions) > 0):

			actionsProb = getBoltzmannProb(startState)
			# print actionsProb

			maxProb = max(actionsProb)
			minProb = min(actionsProb)

			if ((maxProb - minProb) <= 0.001):
				print ("exploit")
				actionIndex = possibleActions[(random.randrange(len(possibleActions)))]
				nextState = possNextState[actionIndex]

			else:
				print ("explore")
				actionIndex = actionsProb.index(max(actionsProb))
				nextState = possNextState[actionIndex]

			# print nextState

			for action in range(0, len(R[nextState])):
					if (R[nextState][action] != 999 and qMax <= Q[nextState][action]):
						qMax = Q[nextState][action]

			Q[startState][actionIndex] = Q[startState][actionIndex] + alpha * (R[startState][actionIndex] + (beta * qMax) - Q[startState][actionIndex])
			startState = nextState

			num_of_steps += 1

	print ("num_of_steps: ", num_of_steps)
	print (Q)



if __name__ == "__main__":
	# discount factor
	beta = 0.9 
	# learning rate
	alpha = 0.01
	# for Boltzmann probability
	temperature = 10.0

	#storing the q values
	Q = [[0.0 for i in range(4)] for j in range(100)]
	R = [[0.0 for i in range(4)] for j in range(100)]

	setup_env()

	# goal state
	goalState = 55
	# starting sate
	startState = 0

	greedyQLearningBoltzmann(startState)