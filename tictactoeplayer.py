import numpy as np
import random
import time

board = [[' ',' ',' '],
         [' ',' ',' '],
         [' ',' ',' ']]

random.seed(1)
game_over = False
j = 0
while game_over == False:
    correct = False
    chosen_comp = False
    while correct == False:
        player_choice_row = int(input('Choose row: '))
        player_choice_column = int(input('Choose column: '))
        player_choice_row -= 1
        player_choice_column -= 1
        if board[player_choice_row][player_choice_column] == ' ':
            board[player_choice_row][player_choice_column] = 'X'
            correct = True
            print(board[0])
            print(board[1])
            print(board[2])
            if (board[0][0]=='X' and board[0][1]=='X' and board[0][2]=='X') or\
            (board[1][0]=='X' and board[1][1]=='X' and board[1][2]=='X') or\
            (board[2][0]=='X' and board[2][1]=='X' and board[2][2]=='X') or\
            (board[0][0]=='X' and board[1][0]=='X' and board[2][0]=='X') or\
            (board[0][1]=='X' and board[1][1]=='X' and board[2][1]=='X') or\
            (board[0][2]=='X' and board[1][2]=='X' and board[2][2]=='X') or\
            (board[0][0]=='X' and board[1][1]=='X' and board[2][2]=='X') or\
            (board[0][2]=='X' and board[1][1]=='X' and board[2][0]=='X'):
                print('Player wins!')
                game_over = True
        else:
            print('Please choose a space that has not been chosen yet.')
    
    if j < 4 and game_over == False:
        while chosen_comp == False:
            comp_row = random.randint(0,2)
            comp_column = random.randint(0,2)
            if board[comp_row][comp_column] == ' ':
                print("Computer's Move")
                time.sleep(1)
                board[comp_row][comp_column] = 'O'
                print(board[0])
                print(board[1])
                print(board[2])
                chosen_comp = True
                if (board[0][0]=='O' and board[0][1]=='O' and board[0][2]=='O') or\
                (board[1][0]=='O' and board[1][1]=='O' and board[1][2]=='O') or\
                (board[2][0]=='O' and board[2][1]=='O' and board[2][2]=='O') or\
                (board[0][0]=='O' and board[1][0]=='O' and board[2][0]=='O') or\
                (board[0][1]=='O' and board[1][1]=='O' and board[2][1]=='O') or\
                (board[0][2]=='O' and board[1][2]=='O' and board[2][2]=='O') or\
                (board[0][0]=='O' and board[1][1]=='O' and board[2][2]=='O') or\
                (board[0][2]=='O' and board[1][1]=='O' and board[2][0]=='O'):
                    print('Computer Wins!')
                    game_over = True
    j += 1