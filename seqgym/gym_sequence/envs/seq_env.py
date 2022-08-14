
from random import getstate
import gym
import random
from gym import error, spaces
from gym.utils import seeding
import numpy as np
from copy import deepcopy
from gym_sequence.envs.game import Sequence
from gym_sequence.envs.util import cardToNum, numToCard, numboardList, numboard, handToText, handToNum, decodeAction, encodeAction, normalize, countSpecialCards
from stable_baselines3.common import logger
from utils.agents import Agent

from utils.files import get_best_model_name, load_all_models, load_model

class SeqEnv(gym.Env):

    
    def __init__(self, opponent_type = "mostly_best"):
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
        self.illegalMoves = [0,0]
        self.opponent_type = opponent_type
        self.opponent_models = load_all_models(self)
        self.best_model_name = get_best_model_name(self.name)
        
    
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
    
    def action_masks(self):
        return self.legal_actions

    def setup_opponents(self):
        if self.opponent_type == 'rules':
            self.opponent_agent = Agent('rules')
        else:
            # incremental load of new model
            best_model_name = get_best_model_name(self.name)
            if self.best_model_name != best_model_name:
                self.opponent_models.append(load_model(self, best_model_name ))
                self.best_model_name = best_model_name

            if self.opponent_type == 'random':
                start = 0
                end = len(self.opponent_models) - 1
                i = random.randint(start, end)
                self.opponent_agent = Agent('ppo_opponent', self.opponent_models[i]) 

            elif self.opponent_type == 'best':
                self.opponent_agent = Agent('ppo_opponent', self.opponent_models[-1])  

            elif self.opponent_type == 'mostly_best':
                j = random.uniform(0,1)
                if j < 0.8:
                    self.opponent_agent = Agent('ppo_opponent', self.opponent_models[-1])  
                else:
                    start = 0
                    end = len(self.opponent_models) - 1
                    i = random.randint(start, end)
                    self.opponent_agent = Agent('ppo_opponent', self.opponent_models[i])  

            elif self.opponent_type == 'base':
                self.opponent_agent = Agent('base', self.opponent_models[0])  

        self.agent_player_num = np.random.choice(self.n_players)
        self.agents = [self.opponent_agent] * self.n_players
        self.agents[self.agent_player_num] = None
        try:
            #if self.players is defined on the base environment
            print(f'Agent plays as Player {self.players[self.agent_player_num].id}')
            # logger.debug(f'Agent plays as Player {self.players[self.agent_player_num].id}')
        except:
            pass



    # simple 2 scaled 
    def step_01(self, action):
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
       
        # Update Game State if Legal - legal check also in function
        self.game.updateGame(self.game.currentPlayer,self.game.currentTeam,card, act, position)
        
        
       
        # Implement Rewards
        # Legal Move -> +5 Else -5
        # reward = [self.game.score[0],self.game.score[1]]
        if not self.game.legalMove:
            # reward[self.game.currentPlayer] = -1
            self.illegalMoves[self.current_player_num] = self.illegalMoves[self.current_player_num] + 1
            self.gameOver = True
            # done = True
            

        # Last player is not always otherplayer
        self.lastPlayer = deepcopy(self.game.currentPlayer)

        
        # Early Quit Conditions to avoid infinte loops
        done = False
        assert self.game.currentPlayer != self.game.otherPlayer
        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver or self.game.score[0] > 1 or self.game.score[1] > 1:
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 100
                reward[self.game.otherPlayer] =  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = -100
                reward[self.game.otherPlayer] =  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 50
                reward[self.game.otherPlayer] = 50
            self.game.gameOver = True
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        # Updating Player state in Game
        if self.game.legalMove:
            self.game.otherPlayer = deepcopy(self.game.currentPlayer)
            self.game.currentPlayer = (self.game.currentPlayer + 1) % self.n_players
        
        # Updating Player state in Env
        self.current_player_num = deepcopy(self.game.currentPlayer)

        # Getting moves for updated player
        self.game.getMoves()

        # Updating state
        state = self.observation

        if 1 not in self.legal_actions:
            self.gameOver = True
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 100
                reward[self.game.otherPlayer] =  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = -100
                reward[self.game.otherPlayer] =  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 50
                reward[self.game.otherPlayer] = 50
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)


        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info            

    # simple 3 scaled 
    def step_02(self, action):
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
       
        # Update Game State if Legal - legal check also in function
        self.game.updateGame(self.game.currentPlayer,self.game.currentTeam,card, act, position)
        
        
       
        # Implement Rewards
        # Legal Move -> +5 Else -5
        # reward = [self.game.score[0],self.game.score[1]]
        if not self.game.legalMove:
            # reward[self.game.currentPlayer] = -1
            self.illegalMoves[self.current_player_num] = self.illegalMoves[self.current_player_num] + 1
            self.gameOver = True
            # done = True
            

        # Last player is not always otherplayer
        self.lastPlayer = deepcopy(self.game.currentPlayer)

        
        # Early Quit Conditions to avoid infinte loops
        done = False
        assert self.game.currentPlayer != self.game.otherPlayer
        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver or self.game.score[0] > 2 or self.game.score[1] > 2:
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 100
                reward[self.game.otherPlayer] =  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = -100
                reward[self.game.otherPlayer] =  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 50
                reward[self.game.otherPlayer] = 50
            self.game.gameOver = True
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        # Updating Player state in Game
        if self.game.legalMove:
            self.game.otherPlayer = deepcopy(self.game.currentPlayer)
            self.game.currentPlayer = (self.game.currentPlayer + 1) % self.n_players
        
        # Updating Player state in Env
        self.current_player_num = deepcopy(self.game.currentPlayer)

        # Getting moves for updated player
        self.game.getMoves()

        # Updating state
        state = self.observation

        if 1 not in self.legal_actions:
            self.gameOver = True
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 100
                reward[self.game.otherPlayer] =  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = -100
                reward[self.game.otherPlayer] =  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 50
                reward[self.game.otherPlayer] = 50
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)


        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info            

    # simple 4 scaled 
    def step_main(self, action):
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
       
        # Update Game State if Legal - legal check also in function
        self.game.updateGame(self.game.currentPlayer,self.game.currentTeam,card, act, position)
        
        
       
        # Implement Rewards
        # Legal Move -> +5 Else -5
        # reward = [self.game.score[0],self.game.score[1]]
        if not self.game.legalMove:
            # reward[self.game.currentPlayer] = -1
            self.illegalMoves[self.current_player_num] = self.illegalMoves[self.current_player_num] + 1
            self.gameOver = True
            # done = True
            

        # Last player is not always otherplayer
        self.lastPlayer = deepcopy(self.game.currentPlayer)

        
        # Early Quit Conditions to avoid infinte loops
        done = False
        assert self.game.currentPlayer != self.game.otherPlayer
        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver or self.game.score[0] > 3 or self.game.score[1] > 3:
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 100
                reward[self.game.otherPlayer] =  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = -100
                reward[self.game.otherPlayer] =  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 50
                reward[self.game.otherPlayer] = 50
            self.game.gameOver = True
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        # Updating Player state in Game
        if self.game.legalMove:
            self.game.otherPlayer = deepcopy(self.game.currentPlayer)
            self.game.currentPlayer = (self.game.currentPlayer + 1) % self.n_players
        
        # Updating Player state in Env
        self.current_player_num = deepcopy(self.game.currentPlayer)

        # Getting moves for updated player
        self.game.getMoves()

        # Updating state
        state = self.observation

        if 1 not in self.legal_actions:
            self.gameOver = True
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 100
                reward[self.game.otherPlayer] =  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = -100
                reward[self.game.otherPlayer] =  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 50
                reward[self.game.otherPlayer] = 50
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)


        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info            

    # simple max scaled 
    def step_04(self, action):
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
       
        # Update Game State if Legal - legal check also in function
        self.game.updateGame(self.game.currentPlayer,self.game.currentTeam,card, act, position)
        
        
       
        # Implement Rewards
        # Legal Move -> +5 Else -5
        # reward = [self.game.score[0],self.game.score[1]]
        if not self.game.legalMove:
            # reward[self.game.currentPlayer] = -1
            self.illegalMoves[self.current_player_num] = self.illegalMoves[self.current_player_num] + 1
            self.gameOver = True
            # done = True
            

        # Last player is not always otherplayer
        self.lastPlayer = deepcopy(self.game.currentPlayer)

        
        # Early Quit Conditions to avoid infinte loops
        done = False
        assert self.game.currentPlayer != self.game.otherPlayer
        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver :
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 100
                reward[self.game.otherPlayer] =  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = -100
                reward[self.game.otherPlayer] =  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 50
                reward[self.game.otherPlayer] = 50
            self.game.gameOver = True
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        # Updating Player state in Game
        if self.game.legalMove:
            self.game.otherPlayer = deepcopy(self.game.currentPlayer)
            self.game.currentPlayer = (self.game.currentPlayer + 1) % self.n_players
        
        # Updating Player state in Env
        self.current_player_num = deepcopy(self.game.currentPlayer)

        # Getting moves for updated player
        self.game.getMoves()

        # Updating state
        state = self.observation

        if 1 not in self.legal_actions:
            self.gameOver = True
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 100
                reward[self.game.otherPlayer] =  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = -100
                reward[self.game.otherPlayer] =  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] = 50
                reward[self.game.otherPlayer] = 50
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)


        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info            


    
    # complex 2
    def step_05(self, action):

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
        # reward = [self.game.score[0],self.game.score[1]]
        if not self.game.legalMove:
            # reward[self.game.currentPlayer] = -1
            self.illegalMoves[self.current_player_num] = self.illegalMoves[self.current_player_num] + 1
            

        # Empty cells can be used as a measure of match complete %age
        flatState = self.game.playState.flatten().tolist()
        emptyCells = flatState.count(0)

        # reward for early sequence 

        # Each new Sequence -> +10 + early bonus
        newSequences = self.game.score[self.game.currentPlayer] - self.game.oldScore[self.game.currentPlayer]
        reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + (newSequences * 10) + (emptyCells * 0.1)


        
        

        # Reward for Jack to make sequence
        if 'J' in card and newSequences > 0:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + (newSequences*5)

        if card == 'J♠'  or card =='J♥' :
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 5
            reward[self.game.otherPlayer] = reward[self.game.otherPlayer] - 5


        # Score Greater -> +20 / Score Lesser -> +0 / Score Equal -> +5 Each 
        if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]) and self.game.legalMove:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 20
        elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]) and self.game.legalMove:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 10
            reward[self.game.otherPlayer] = reward[self.game.otherPlayer] + 10
        # print(reward)
        
       
        

        # Last player is not always otherplayer
        self.lastPlayer = deepcopy(self.game.currentPlayer)

        
        # Early Quit Conditions to avoid infinte loops
        done = False
        assert self.game.currentPlayer != self.game.otherPlayer
        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver or self.game.score[0] > 1 or self.game.score[1] > 1:
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 100
                reward[self.game.otherPlayer] +=  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += -100
                reward[self.game.otherPlayer] +=  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 50
                reward[self.game.otherPlayer] += 50
            self.game.gameOver = True
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        # Updating Player state in Game
        if self.game.legalMove:
            self.game.otherPlayer = deepcopy(self.game.currentPlayer)
            self.game.currentPlayer = (self.game.currentPlayer + 1) % self.n_players
        
        # Updating Player state in Env
        self.current_player_num = deepcopy(self.game.currentPlayer)

        # Getting moves for updated player
        self.game.getMoves()

        # Updating state
        state = self.observation

        if 1 not in self.legal_actions:
            self.gameOver = True
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 100
                reward[self.game.otherPlayer] +=  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += -100
                reward[self.game.otherPlayer] +=  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 50
                reward[self.game.otherPlayer] += 50
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info            

     # complex 2
   

     
    # complex 3
    def step_06(self, action):

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
        # reward = [self.game.score[0],self.game.score[1]]
        if not self.game.legalMove:
            # reward[self.game.currentPlayer] = -1
            self.illegalMoves[self.current_player_num] = self.illegalMoves[self.current_player_num] + 1
            

        # Empty cells can be used as a measure of match complete %age
        flatState = self.game.playState.flatten().tolist()
        emptyCells = flatState.count(0)

        # reward for early sequence 

        # Each new Sequence -> +10 + early bonus
        newSequences = self.game.score[self.game.currentPlayer] - self.game.oldScore[self.game.currentPlayer]
        reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + (newSequences * 10) + (emptyCells * 0.1)


        
        

        # Reward for Jack to make sequence
        if 'J' in card and newSequences > 0:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + (newSequences*5)

        if card == 'J♠'  or card =='J♥' :
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 5
            reward[self.game.otherPlayer] = reward[self.game.otherPlayer] - 5

        

        # Last player is not always otherplayer
        self.lastPlayer = deepcopy(self.game.currentPlayer)

        
        # Early Quit Conditions to avoid infinte loops
        done = False
        assert self.game.currentPlayer != self.game.otherPlayer
        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver or self.game.score[0] > 2 or self.game.score[1] > 2:
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 100
                reward[self.game.otherPlayer] +=  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += -100
                reward[self.game.otherPlayer] +=  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 50
                reward[self.game.otherPlayer] += 50
            self.game.gameOver = True
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        # Updating Player state in Game
        if self.game.legalMove:
            self.game.otherPlayer = deepcopy(self.game.currentPlayer)
            self.game.currentPlayer = (self.game.currentPlayer + 1) % self.n_players
        
        # Updating Player state in Env
        self.current_player_num = deepcopy(self.game.currentPlayer)

        # Getting moves for updated player
        self.game.getMoves()

        # Updating state
        state = self.observation

        if 1 not in self.legal_actions:
            self.gameOver = True
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 100
                reward[self.game.otherPlayer] +=  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += -100
                reward[self.game.otherPlayer] +=  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 50
                reward[self.game.otherPlayer] += 50
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info            

     # complex 2
   

    # complex 4
    def step_07(self, action):

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
        # reward = [self.game.score[0],self.game.score[1]]
        if not self.game.legalMove:
            # reward[self.game.currentPlayer] = -1
            self.illegalMoves[self.current_player_num] = self.illegalMoves[self.current_player_num] + 1
            

        # Empty cells can be used as a measure of match complete %age
        flatState = self.game.playState.flatten().tolist()
        emptyCells = flatState.count(0)

        # reward for early sequence 

        # Each new Sequence -> +10 + early bonus
        newSequences = self.game.score[self.game.currentPlayer] - self.game.oldScore[self.game.currentPlayer]
        reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + (newSequences * 10) + (emptyCells * 0.1)


        
        

        # Reward for Jack to make sequence
        if 'J' in card and newSequences > 0:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + (newSequences*5)

        if card == 'J♠'  or card =='J♥' :
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 5
            reward[self.game.otherPlayer] = reward[self.game.otherPlayer] - 5


        # Score Greater -> +20 / Score Lesser -> +0 / Score Equal -> +5 Each 
        if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]) and self.game.legalMove:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 20
        elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]) and self.game.legalMove:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 10
            reward[self.game.otherPlayer] = reward[self.game.otherPlayer] + 10
        # print(reward)
        
       
        

        # Last player is not always otherplayer
        self.lastPlayer = deepcopy(self.game.currentPlayer)

        
        # Early Quit Conditions to avoid infinte loops
        done = False
        assert self.game.currentPlayer != self.game.otherPlayer
        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver or self.game.score[0] > 3 or self.game.score[1] > 3:
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 100
                reward[self.game.otherPlayer] +=  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += -100
                reward[self.game.otherPlayer] +=  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 50
                reward[self.game.otherPlayer] += 50
            self.game.gameOver = True
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        # Updating Player state in Game
        if self.game.legalMove:
            self.game.otherPlayer = deepcopy(self.game.currentPlayer)
            self.game.currentPlayer = (self.game.currentPlayer + 1) % self.n_players
        
        # Updating Player state in Env
        self.current_player_num = deepcopy(self.game.currentPlayer)

        # Getting moves for updated player
        self.game.getMoves()

        # Updating state
        state = self.observation

        if 1 not in self.legal_actions:
            self.gameOver = True
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 100
                reward[self.game.otherPlayer] +=  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += -100
                reward[self.game.otherPlayer] +=  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 50
                reward[self.game.otherPlayer] += 50
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info            

     # complex 2
   
    # complex max
    def step_08(self, action):

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
        # reward = [self.game.score[0],self.game.score[1]]
        if not self.game.legalMove:
            # reward[self.game.currentPlayer] = -1
            self.illegalMoves[self.current_player_num] = self.illegalMoves[self.current_player_num] + 1
            

        # Empty cells can be used as a measure of match complete %age
        flatState = self.game.playState.flatten().tolist()
        emptyCells = flatState.count(0)

        # reward for early sequence 

        # Each new Sequence -> +10 + early bonus
        newSequences = self.game.score[self.game.currentPlayer] - self.game.oldScore[self.game.currentPlayer]
        reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + (newSequences * 10) + (emptyCells * 0.1)


        
        

        # Reward for Jack to make sequence
        if 'J' in card and newSequences > 0:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + (newSequences*5)

        if card == 'J♠'  or card =='J♥' :
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 5
            reward[self.game.otherPlayer] = reward[self.game.otherPlayer] - 5


        # Score Greater -> +20 / Score Lesser -> +0 / Score Equal -> +5 Each 
        if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]) and self.game.legalMove:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 20
        elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]) and self.game.legalMove:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 10
            reward[self.game.otherPlayer] = reward[self.game.otherPlayer] + 10
        # print(reward)
        
       
        

        # Last player is not always otherplayer
        self.lastPlayer = deepcopy(self.game.currentPlayer)

        
        # Early Quit Conditions to avoid infinte loops
        done = False
        assert self.game.currentPlayer != self.game.otherPlayer
        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver:
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 100
                reward[self.game.otherPlayer] +=  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += -100
                reward[self.game.otherPlayer] +=  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 50
                reward[self.game.otherPlayer] += 50
            self.game.gameOver = True
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        # Updating Player state in Game
        if self.game.legalMove:
            self.game.otherPlayer = deepcopy(self.game.currentPlayer)
            self.game.currentPlayer = (self.game.currentPlayer + 1) % self.n_players
        
        # Updating Player state in Env
        self.current_player_num = deepcopy(self.game.currentPlayer)

        # Getting moves for updated player
        self.game.getMoves()

        # Updating state
        state = self.observation

        if 1 not in self.legal_actions:
            self.gameOver = True
            done = True
            if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 100
                reward[self.game.otherPlayer] +=  -100
            elif (self.game.score[self.game.currentPlayer] < self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += -100
                reward[self.game.otherPlayer] +=  100
            elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
                reward[self.game.currentPlayer] += 50
                reward[self.game.otherPlayer] += 50
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info            

     # complex 2
   



    # simple 2 2
    def step_main_3(self, action):

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
        # reward = [self.game.score[0],self.game.score[1]]
        if not self.game.legalMove:
            # reward[self.game.currentPlayer] = -1
            self.illegalMoves[self.current_player_num] = self.illegalMoves[self.current_player_num] + 1
            

       
        # Last player is not always otherplayer
        self.lastPlayer = deepcopy(self.game.currentPlayer)

        reward = [self.game.score[0],self.game.score[1]]

        # Early Quit Conditions to avoid infinte loops
        done = False
        assert self.game.currentPlayer != self.game.otherPlayer
        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver or self.game.score[0] > 1 or self.game.score[1] > 1:
            done = True
            self.game.gameOver = True
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        # Updating Player state in Game
        if self.game.legalMove:
            self.game.otherPlayer = deepcopy(self.game.currentPlayer)
            self.game.currentPlayer = (self.game.currentPlayer + 1) % self.n_players
        
        # Updating Player state in Env
        self.current_player_num = deepcopy(self.game.currentPlayer)

        # Getting moves for updated player
        self.game.getMoves()

        # Updating state
        state = self.observation

        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info            

    # simple 2 unlimited
    def step_main_4(self, action):

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
        # reward = [self.game.score[0],self.game.score[1]]
        if not self.game.legalMove:
            # reward[self.game.currentPlayer] = -1
            self.illegalMoves[self.current_player_num] = self.illegalMoves[self.current_player_num] + 1
            

       
        # Last player is not always otherplayer
        self.lastPlayer = deepcopy(self.game.currentPlayer)

        reward = [self.game.score[0],self.game.score[1]]

        # Early Quit Conditions to avoid infinte loops
        done = False
        assert self.game.currentPlayer != self.game.otherPlayer
        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver:
            done = True
            self.game.gameOver = True
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        # Updating Player state in Game
        if self.game.legalMove:
            self.game.otherPlayer = deepcopy(self.game.currentPlayer)
            self.game.currentPlayer = (self.game.currentPlayer + 1) % self.n_players
        
        # Updating Player state in Env
        self.current_player_num = deepcopy(self.game.currentPlayer)

        # Getting moves for updated player
        self.game.getMoves()

        # Updating state
        state = self.observation

        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info            

    

    def step11(self, action):
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
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 5
        else:
            self.illegalMoves[self.current_player_num] = self.illegalMoves[self.current_player_num] + 1
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] - 5


        # Empty cells can be used as a measure of match complete %age
        flatState = self.game.playState.flatten().tolist()
        emptyCells = flatState.count(0)

        # reward for early sequence 

        # Each new Sequence -> +10 + early bonus
        newSequences = 0
        newSequences = self.game.score[self.game.currentPlayer] - self.game.oldScore[self.game.currentPlayer]
        if self.game.legalMove and newSequences > 0:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + (newSequences * 100) + (emptyCells * 0.5)


        
        

        # Reward for Jack to make sequence
        # if 'J' in card and newSequences > 0 and self.game.legalMove:
        #     reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + (newSequences * 2)

        # if (card == 'J♠' and self.game.legalMove) or (card =='J♥' and self.game.legalMove):
        #     reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 2
        #     reward[self.game.otherPlayer] = reward[self.game.otherPlayer] - 2


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


        

        # removed condition emptyCells < 3 or (emptyCells < 7 and self.illegalCount > 200)  or

        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver or self.game.score[0] > 1 or self.game.score[1] > 1:
            # give extra reward on game conclusion
            # if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]):
            #     reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 100
            #     reward[self.game.otherPlayer] = reward[self.game.otherPlayer] - 100
            # elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]):
            #     reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 50
            #     reward[self.game.otherPlayer] = reward[self.game.otherPlayer] + 50
            done = True
            self.game.gameOver = True
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        # if not self.game.legalMove:
        #     reward[self.game.currentPlayer] = -10
        #     done = True
        #     print(self.game.playState)
        #     print(self.game.score)

        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info


    
    def step3(self, action):
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
        # if self.lastPlayer == self.game.currentPlayer and not self.game.legalMove:
        #     self.illegalCount = self.illegalCount + 1
        # elif self.lastPlayer != self.game.currentPlayer:
        #     self.illegalCount = 1

        # Implement Rewards
        # Legal Move -> +5 Else -5
        if self.game.legalMove:
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 1
            
        else:
            self.illegalMoves[self.current_player_num] = self.illegalMoves[self.current_player_num] + 1
            reward[self.game.currentPlayer] = reward[self.game.currentPlayer] - 5


       
       
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


        

        # removed condition emptyCells < 3 or (emptyCells < 7 and self.illegalCount > 200)  or

        if not self.game.currentMoves or (len(self.game.players[1]) == 0) or (len(self.game.players[0]) == 0) or  self.game.gameOver or self.game.score[0] > 1 or self.game.score[1] > 1:
            # give extra reward on game conclusion
            # if (self.game.score[self.game.currentPlayer] > self.game.score[self.game.otherPlayer]) and self.game.legalMove:
            #     reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 30
            #     reward[self.game.otherPlayer] = reward[self.game.otherPlayer] - 30
            # elif (self.game.score[self.game.currentPlayer] == self.game.score[self.game.otherPlayer]) and self.game.legalMove:
            #     reward[self.game.currentPlayer] = reward[self.game.currentPlayer] + 10
            #     reward[self.game.otherPlayer] = reward[self.game.otherPlayer] + 10
            done = True
            self.game.gameOver = True
            print("Illegal Moves: ")
            print(self.illegalMoves)
            print(self.game.playState)
            print(self.game.score)

        # if not self.game.legalMove:
        #     reward[self.game.currentPlayer] = -10
        #     done = True
        #     print(self.game.playState)
        #     print(self.game.score)

        info = {}
        # print(reward)
        return deepcopy(state), deepcopy(reward), done, info




    def reset_main(self):
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
        self.illegalMoves = [0,0]
        self.game.currentPlayer = 0
        self.game.otherPlayer = 1
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
    def reset(self):
        obs = self.reset_main()
        self.setup_opponents()
        if self.current_player_num != self.agent_player_num:   
            self.continue_game()

        return obs
    
    @property
    def current_agent(self):
        return self.agents[self.current_player_num]

    def continue_game(self):
        observation = None
        reward = None
        done = None

        while self.current_player_num != self.agent_player_num:
            self.render()
            action = self.current_agent.choose_action(self, choose_best_action = False, mask_invalid_actions = True)
            observation, reward, done, _ = self.step_main(action)
            # # logger.debug(f'Rewards: {reward}')
            # # logger.debug(f'Done: {done}')
            if done:
                break

        return observation, reward, done, None


    def step(self, action):
        self.render()
        observation, reward, done, _ = self.step_main(action)
        # logger.HumanOutputFormat(f'Action played by agent: {action}')
        # logger.HumanOutputFormat(f'Rewards: {reward}')
        # logger.HumanOutputFormat(f'Done: {done}')

        if not done:
            package = self.continue_game()
            if package[0] is not None:
                observation, reward, done, _ = package


        agent_reward = reward[self.agent_player_num]
        # # logger.debug(f'\nReward To Agent: {agent_reward}')

        if done:
            print("Agent is Player Number: " + str(self.agent_player_num) + " Reward:" +str(agent_reward))
            self.render()

        return observation, agent_reward, done, {} 
