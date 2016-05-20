# a python project to practice on
# run in shell with: python practice.py

#make a simple guessing game and see how this setup works for coding
#it is a game where the computer chooses a number between two values and you have to guess it
# each turn you will make a guess and the computer will tell you if you were high or low

##
# This will be a convenience function that will get the users input
# it should check to see if it is valid, if it is invalid it will repeatedly
# ask the user for input until it is valid
# only integer numbers are valid input
# 
# prompt - a string, that will be printed out before asking for input
# returns a number (not a string) 
def get_numbered_input(self, prompt) :
    pass

class Game :


    ##
    # initializes the game object
    #
    # low - the lower bound on possible numbers to choose
    # high - the upper bound on possible numbers to choose
    # 
    def __init__(self, low, high) :
        pass

    ##
    #this method should go through a loop until the user wins or loses
    #
    # max_guesses - if the user makes more guesses than this they should lose
    def play_game(self, max_guesses) :
        pass
    
    ##
    # this should pick a random number between the low and high values, and store it in
    # the variable self.chosen_number
    #
    # post - self.chosen_number exists and is set to a random number between self.low and self.high
    def choose_number(self) :
        pass

    ##
    #
    # returns true if the guess passed in is correct, false otherwise
    def is_correct(self, guess) :
        pass

    ##
    #
    # returns true if the guess passed in is greater than the chosen number, false otherwise
    def is_high(self, guess) :
        pass



##
#
# TODO: perhaps prompt the user what the range and number of guesses should be before starting the game?
def main() :
    game = Game(0,100)
    play_game(5)


main()
