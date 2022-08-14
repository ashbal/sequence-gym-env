
from random import getstate
import gym
import random
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from copy import deepcopy
from gym_sequence.envs.game import Sequence
from gym_sequence.envs.util import cardToNum, numToCard, numboardList, numboard, handToText, handToNum, decodeAction, encodeAction, normalize, countSpecialCards

class SeqEnv(gym.Env):

    
    def __init__(self):
        self.name = "seqgym"
        self.illegalCount = 1
        self.game = Sequence()
        self.game.setup(2,2)
        self.game.getMoves()
        self.current_player_num = 0
        self.n_players = 2
        self.observation_space = spaces.Box(-1,1,(209,))
        self.action_space = spaces.Discrete(500)
        self.illegalCount = 1
        self.lastPlayer = 0
    
    @property
    def observation(self):
        scores = [self.game.score[0],self.game.score[1]]
        myscore = deepcopy(scores)
        normalScores = normalize(myscore,{'actual': {'lower': 0, 'upper': 10}, 'desired': {'lower': -1, 'upper': 1}})
        
        board = deepcopy(numboard).flatten()
        normalBoard = normalize(board,{'actual': {'lower': 0, 'upper': 52}, 'desired': {'lower': -1, 'upper': 1}})
        
        playState = deepcopy(self.game.playState).flatten()
        normalPlayState = normalize(playState,{'actual': {'lower': 0, 'upper': 4}, 'desired': {'lower': -1, 'upper': 1}})
        
        hand = deepcopy(self.game.players[self.game.currentPlayer])
        numHand = [cardToNum(x) for x in hand]
        normalHand = normalize(numHand,{'actual': {'lower': 0, 'upper': 52}, 'desired': {'lower': -1, 'upper': 1}})
        while len(normalHand) < 7:
            normalHand.append(1)
        # out = np.append(normalBoard,normalPlayState,normalHand)
        out = np.concatenate((normalBoard,normalPlayState,normalHand,normalScores), axis=None)
        return out

    @property
    def legal_actions(self):
        legal_actions = np.zeros(500, dtype=int)
        legal_ids = []
        # implement check
        possible_moves = deepcopy(self.game.currentMoves)
        if possible_moves:
            for key in possible_moves:
                card = key
                locations = possible_moves[key]
                locations.pop(0)
                if locations:
                    for i in locations:
                        legal_ids.append(encodeAction(card,i))
                
        for id in legal_ids:
            legal_actions[id] = 1

        return np.array(legal_actions)
    
    def step(self, action):
        # Initialize Rewards
        reward = [0.0] * self.n_players
        act = ''

        # Set Teams
        if self.game.currentPlayer == 1:
            self.game.currentTeam = self.game.RED
        elif self.game.currentPlayer == 0:
            self.game.currentTeam = self.game.BLUE
        
        # Decode for game
        act, card, position = decodeAction(action)
        self.lastPlayer = deepcopy(self.game.currentPlayer)
        # print([self.current_player_num,act,card,position])
       
        # Update Game State if Legal
        self.game.updateGame(self.game.currentPlayer,self.game.currentTeam,card, act, position)
        
        
       
        # Implement Rewards
        # Legal Move -> +5 Else -5
        reward = [self.game.score[0],self.game.score[1]]
        if not self.game.legalMove:
            reward[self.game.currentPlayer] = -1

        

        # Last player is not always otherplayer
        self.lastPlayer = deepcopy(self.game.currentPlayer)

        # Updating Player state in Game
        if self.game.legalMove:
            self.otherPlayer = deepcopy(self.game.currentPlayer)
            self.game.currentPlayer = (self.game.currentPlayer + 1) % self.n_players
        
        # Updating Player state in Env
        self.current_player_num = deepcopy(self.game.currentPlayer)

        # Getting moves for updated player
        self.game.getMoves()

        # Updating state
        state = self.observation

        # Early Quit Conditions to avoid infinte loops
        done = False

        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver or self.game.score[0] > 2 or self.game.score[1] > 2:
            done = True
            print(self.game.playState)
            print(self.game.score)

        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info            

    
    def step2(self, action):
        # Initialize Rewards
        reward = [0.0] * self.n_players
        act = ''

        # Set Teams
        if self.game.currentPlayer == 1:
            self.game.currentTeam = self.game.RED
        elif self.game.currentPlayer == 0:
            self.game.currentTeam = self.game.BLUE
        
        # Decode for game
        act, card, position = decodeAction(action)
        self.lastPlayer = deepcopy(self.game.currentPlayer)
        # print([self.current_player_num,act,card,position])
       
        # Update Game State if Legal
        self.game.updateGame(self.game.currentPlayer,self.game.currentTeam,card, act, position)
        
        
        # Count Illegal moves
        if self.lastPlayer == self.game.currentPlayer and not self.game.legalMove:
            self.illegalCount = self.illegalCount + 1
        elif self.lastPlayer != self.game.currentPlayer:
            self.illegalCount = 1

        # Implement Rewards
        # Legal Move -> +5 Else -5
        if self.game.legalMove:
            # reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 0.5
            ...
        else:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] - 0.5


        # Empty cells can be used as a measure of match complete %age
        flatState = self.game.playState.flatten().tolist()
        emptyCells = flatState.count(0)

        # reward for early sequence 

        # Each new Sequence -> +100
        newSequences = self.game.score[self.game.currentPlayer] - self.game.oldScore[self.game.currentPlayer]
        reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + (newSequences * 10) + (emptyCells * 0.1)


        
        

        # Reward for Jack to make sequence
        if 'J' in card and newSequences > 0:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + (newSequences * 0.2)

        if (card == 'J♠' and self.game.legalMove) or (card =='J♥' and self.game.legalMove):
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 0.5
            reward[self.game.otherPlayer] = reward[self.game.otherPlayer] - 0.5


        # Score Greater -> +20 / Score Lesser -> +0 / Score Equal -> +5 Each 
        # if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]) and self.game.legalMove:
        #     reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 2
        # elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]) and self.game.legalMove:
        #     reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 0.5
        #     reward[self.game.otherPlayer] = reward[self.game.otherPlayer] + 0.5
        # print(reward)
        
        # Reward Penalty for not using any jacks till end 15% of the game
        # if emptyCells < 16:
        #     specialNum = countSpecialCards(self.game.players[self.game.currentPlayer])
        #     reward[self.game.currentPlayer] = reward[self.game.currentPlayer] - (specialNum * 0.1)

        # print(emptyCells)
       
        # Last player is not always otherplayer
        self.lastPlayer = deepcopy(self.game.currentPlayer)

        # Updating Player state in Game
        if self.game.legalMove:
            self.otherPlayer = deepcopy(self.game.currentPlayer)
            self.game.currentPlayer = (self.game.currentPlayer + 1) % self.n_players
        
        # Updating Player state in Env
        self.current_player_num = deepcopy(self.game.currentPlayer)

        # Getting moves for updated player
        self.game.getMoves()

        # Updating state
        state = self.observation

        # Early Quit Conditions to avoid infinte loops
        done = False


        # if not self.game.legalMove:
        #     reward[self.game.currentPlayer] = -10
        #     done = True
            # print(self.game.playState)
            # print(self.game.score)

        # removed condition emptyCells < 3 or (emptyCells < 7 and self.illegalCount > 200)  or

        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver or self.game.score[0] > 1 or self.game.score[1] > 1:
            # give extra reward on game conclusion
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]) and self.game.legalMove:
                reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 20
                reward[self.game.otherPlayer] = reward[self.game.otherPlayer] - 20
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]) and self.game.legalMove:
                reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 10
                reward[self.game.otherPlayer] = reward[self.game.otherPlayer] + 10
            done = True
            print(self.game.playState)
            print(self.game.score)


        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info



    def reset(self):
        print("\nRestarting Game\n")
        del self.game
        self.game = Sequence()
        self.game.setup(2,2)
        # print(self.game.players)
        # print()
        self.game.getMoves()
        self.current_player_num = 0
        self.n_players = 2
        self.illegalCount = 1
        self.lastPlayer = 0
        return deepcopy(self.observation)

    def render(self, mode='human'):
        # print(self.game.playState)
        # print(self.game.score)
        # print(self.game.players)
        ...
        
        
    

    def rules_move(self):
        # For now rules plays random legal moves to teach legal moves
        # print("Rules Move")
        # print(self.current_player_num)
        allMoves = self.game.currentMoves
        if allMoves:
            available = False
            onlyMoves = list(allMoves.values())
            for item in onlyMoves:
                if len(item) > 1:
                    available = True

            if not available:
                self.gameOver = True
                return

            handCards = self.game.players[self.game.currentPlayer]
            card = random.choice(handCards)
            cardMoves = deepcopy(allMoves[card])


            while len(cardMoves) < 2 and available :
                print("Changing Card")
                card = random.choice(handCards)
                cardMoves = deepcopy(allMoves[card])

            if cardMoves:
                action = cardMoves.pop(0)
                move = random.choice(cardMoves)
                actionID = encodeAction(card,move)
                return self.create_action_probs(actionID)
                # self.updateGame(self.currentPlayer,self.currentTeam,card,action,move)
            else:
                self.gameOver = True
                return
        else:
            self.gameOver = True
            return
        

    def create_action_probs(self, action):
        action_probs = [0.0] * self.action_space.n
        action_probs[action] = 1
        return action_probs 

    def close(self):
        ...  
