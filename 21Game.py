# coding: utf-8

from random import randint
import random

# The R matrix will represent the environment that our agent is in. In this case, the environment is the
# game 21. The values in the R matrix represent rewards for certain decisions. Each row corresponds to a certain
# state that the agent is in, and each column corresponds to an action the agent can take.

R = [[0,0,0] for j in range(21)]
R[0] = [100,-100,-100]
R[1] = [0,100,-100]
R[2] = [0,0,100]

class AI(object):
    def __init__(self):
        self.states = range(21)
        self.actions = [1,2,3]
        self.Q = [[0,0,0] for i in range(21)]
        
    def legalRand(self, state):
        if state <= 3:
            return randint(1,state)
        else:
            return randint(1,3)
    
    def gameOver(self,state):
        if state == 0:
            return True
        else:
            return False

    def learn(self, games = 1, lrate = 0, discfac = 0, epsilon = 0):
        for i in range(games):
            start = randint(18,21) # By randomly choosing a starting point, the AI learns to play both first and second.  
            #print "Game {0} of {1}".format(i+1, games)
            while start != 0:
                if random.random() > epsilon:
                    move = self.best_move(start)
                else:
                    move = self.legalRand(start)
                reward = R[start-1][move-1]
                if self.gameOver(start-move): # The agent is rewarded if it wins.
                    self.Q[start-1][move-1] += lrate*(reward+discfac*(100)-self.Q[start-1][move-1])
                    #print "Q wins!"
                    break
                else:
                    randmove = self.legalRand(start-move)
                    if self.gameOver(start-move-randmove): # The agent is punished if the opponent wins. 
                        self.Q[start-1][move-1] += lrate*(reward+discfac*(-100)-self.Q[start-1][move-1])
                        #print "Random wins..."
                        break
                    else: # If nobody has won yet, updates to Q-values are made normally. 
                        self.Q[start-1][move-1] += lrate*(reward+discfac*max(self.Q[start-move-randmove-1]) - self.Q[start-1][move-1])
                start -= (move + randmove)
        #print self.Q
        print "Done."
            
    def play(self, state):
        # Once the agent has learned, it will use only the best moves in playing against a human or another AI.
        move = self.best_move(state)
        print "Computer's move: {0}".format(move)
        return move
        
    def best_move(self, state):
        q = [self.getQ(state, a) for a in self.actions]
        count = q.count(max(q))
        if count > 1: # If there is more than one best move, the agent will choose the lower number.
            best_choices = [k for k in range(len(self.actions)) if q[k] == max(q)]
            move_index = best_choices[0]
        else: # Otherwise, it just chooses the best option. 
            move_index = q.index(max(q))
        return self.actions[move_index]

    def getQ(self, state, action):
        return self.Q[state-1][action-1]
    
    def getType(self):
        return "Computer"

class RandomPlayer(object):
    def legalRand(self, state):
        if state <= 3:
            return randint(1,state)
        else:
            return randint(1,3)
        
    def play(self, state):
        move = self.legalRand(state)
        print "Random computer's move: %d" % move
        return move
    
    def getType(self):
        return "Random Computer"


class Human(object):
    def __init__(self):
        pass
    
    def getType(self):
        return "Human"

    def play(self, state):
        move = int(raw_input("My move: "))
        if move <= 3 and move > 0:
            return move
        else:
            raise ValueError("Number must be either 1, 2, or 3")


class TwentyOne(object):
    def __init__(self):
        self.startNumber = 21
        print """How to play: The game begins with the total value 21. Player 1 goes first and chooses 
        a number between 1 and 3. That number is then subtracted from 21; this new number becomes the total value. 
        Player 2 proceeds in the same manner, and the game continues until someone chooses a number that makes 
        the total value go to 0. Whoever causes the total value to be 0 wins!\n"""
    
    # The way I have the play methods set up, the order of who goes first can be switched simply by
    # listing one player before the other; that is, player 1 always goes first, but it doesn't matter
    # if player 1 is human or AI. 
    def play(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        #print "Current number: 21"
        while self.startNumber > 0:
            player1_move = player1.play(self.startNumber)
            #print "\nPlayer 1 move: %d" % player1_move 
            self.startNumber -= player1_move
            print "Total: %d\n" % self.startNumber
            if self.player1_wins():
                break
            
            player2_move = player2.play(self.startNumber)
            #print "\nPlayer 2 move: %d" % player2_move
            self.startNumber -= player2_move
            print "Total: %d\n" % self.startNumber
            self.player2_wins()
            
    def player1_wins(self):
        if self.startNumber == 0:
            player1Type = self.player1.getType()
            print "%s wins!" % player1Type
            return True
    
    def player2_wins(self):
        if self.startNumber == 0:
            player2Type = self.player2.getType()
            print "%s wins!" % player2Type

Me = Human()
Computer = AI()
Computer.learn(games = 20000, lrate = 1, discfac = .8, epsilon = .1)

T = TwentyOne()
T.play(Me, Computer)




