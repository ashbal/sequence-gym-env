import numpy as np
suits = ['♠','♥','♦','♣']
series = ['K','Q','J','A','2','3','4','5','6','7','8','9','10']

deck = [n + s for s in suits for n in series ]

# board = [
#             [52,'6♦','7♦','8♦','9♦','10♦','Q♦','K♦','A♦',52],
#             ['5♦','3♥','2♥','2♠','3♠','4♠','5♠','6♠','7♠','A♣'],
#             ['4♦','4♥','K♦','A♦','A♣','K♣','Q♣','10♣','8♠','K♣'],
#             ['3♦','5♥','Q♦','Q♥','10♥','9♥','8♥','9♣','9♠','Q♣'],
#             ['2♦','6♥','10♦','K♥','3♥','2♥','7♥','8♣','10♠','10♣'],
#             ['A♠','7♥','9♦','A♥','4♥','5♥','6♥','7♣','Q♠','9♣'],
#             ['K♠','8♥','8♦','2♣','3♣','4♣','5♣','6♣','K♠','8♣'],
#             ['Q♠','9♥','7♦','6♦','5♦','4♦','3♦','2♦','A♠','7♣'],
#             ['10♠','10♥','Q♥','K♥','A♥','2♣','3♣','4♣','5♣','6♣'],
#             [52,'9♠','8♠','7♠','6♠','5♠','4♠','3♠','2♠',52],
#         ]

numboard = np.array([
            [52, 34, 35, 36, 37, 38, 27, 26, 29, 52], 
            [33, 18, 17, 4,  5,  6,  7,  8,  9,  42], 
            [32, 19, 26, 29, 42, 39, 40, 51, 10, 39], 
            [31, 20, 27, 14, 25, 24, 23, 50, 11, 40], 
            [30, 21, 38, 13, 18, 17, 22, 49, 12, 51], 
            [3,  22, 37, 16, 19, 20, 21, 48, 1,  50], 
            [0,  23, 36, 43, 44, 45, 46, 47, 0,  49], 
            [1,  24, 35, 34, 33, 32, 31, 30, 3,  48], 
            [12, 25, 14, 13, 16, 43, 44, 45, 46, 47], 
            [52, 11, 10, 9,  8,  7,  6,  5,  4,  52]
        ])

numboardList =[
            [52, 34, 35, 36, 37, 38, 27, 26, 29, 52], 
            [33, 18, 17, 4,  5,  6,  7,  8,  9,  42], 
            [32, 19, 26, 29, 42, 39, 40, 51, 10, 39], 
            [31, 20, 27, 14, 25, 24, 23, 50, 11, 40], 
            [30, 21, 38, 13, 18, 17, 22, 49, 12, 51], 
            [3,  22, 37, 16, 19, 20, 21, 48, 1,  50], 
            [0,  23, 36, 43, 44, 45, 46, 47, 0,  49], 
            [1,  24, 35, 34, 33, 32, 31, 30, 3,  48], 
            [12, 25, 14, 13, 16, 43, 44, 45, 46, 47], 
            [52, 11, 10, 9,  8,  7,  6,  5,  4,  52]
        ]


numboard1 = np.array([
            [-1, 34, 35, 36, 37, 38, 27, 26, 29, -1], 
            [33, 18, 17, 4,  5,  6,  7,  8,  9,  42], 
            [32, 19, 26, 29, 42, 39, 40, 51, 10, 39], 
            [31, 20, 27, 14, 25, 24, 23, 50, 11, 40], 
            [30, 21, 38, 13, 18, 17, 22, 49, 12, 51], 
            [3,  22, 37, 16, 19, 20, 21, 48, 1,  50], 
            [0,  23, 36, 43, 44, 45, 46, 47, 0,  49], 
            [1,  24, 35, 34, 33, 32, 31, 30, 3,  48], 
            [12, 25, 14, 13, 16, 43, 44, 45, 46, 47], 
            [-1, 11, 10, 9,  8,  7,  6,  5,  4,  -1]
        ])

numboard2 = np.array([
            [52, 34, 35, 36, 37, 38, 27, 26, 29, 52], 
            [33, 18, 17, 4,  5,  6,  7,  8,  9,  42], 
            [32, 19, 26, 29, 42, 39, 40, 51, 10, 39], 
            [31, 20, 27, 14, 25, 24, 23, 50, 11, 40], 
            [30, 21, 38, 13, 18, 17, 22, 49, 12, 51], 
            [3,  22, 37, 16, 19, 20, 21, 48, 1,  50], 
            [0,  23, 36, 43, 44, 45, 46, 47, 0,  49], 
            [1,  24, 35, 34, 33, 32, 31, 30, 3,  48], 
            [12, 25, 14, 13, 16, 43, 44, 45, 46, 47], 
            [52, 11, 10, 9,  8,  7,  6,  5,  4,  52]
        ])


def cardToNum(card):
    return deck.index(card)

def numToCard(num):
    return deck[num]

def handToNum(hand):
    newHand = []
    for card in hand:
        newHand.append(cardToNum(card))
    return newHand

def handToText(hand):
    newHand = []
    for card in hand:
        newHand.append(cardToNum(card))
    return newHand

# newboard = []

# for row in board:
#     newrow=[]
#     for cell in row:
#         if cell != 52:
#             newrow.append(cardToNum(cell))
#         else:
#             newrow.append(cell)
#     newboard.append(newrow)

# print(newboard)

flatboard = numboard.flatten().tolist()

def getPositions(id):
    positions = []
    # x.index()
    # positions = flatboard.index(id)
    for i in range(100):
        if flatboard[i] == id:
            positions.append(i)
    return positions
    


        

def decodePosition(pos):
    if pos>9:
        return [int(pos/10),pos%10]
    else:
        return[0,pos]


def decodeAction(actionId):
    act = ''
    position = []
    card = []
    actRegion = int(actionId/100)
    position = decodePosition(actionId%100)
    if actRegion == 0:
        act = 'Add'
        card = numToCard( flatboard[actionId%100] % 52 )
    if actRegion == 1:
        act = 'Add'
        card = 'J♦'
    if actRegion == 2:
        act = 'Add'
        card = 'J♣'
    if actRegion == 3:
        act = 'Remove'
        card = 'J♠'
    if actRegion == 4:
        act = 'Remove'
        card = 'J♥'
    
    return act, card, position


def encodeAction(card, position):
    actionID = -1
    positionID = (position[0] * 10) + position[1]
    if 'J' not in card:
        regionID = 0
    elif card == 'J♦':
        regionID = 1
    elif card == 'J♣':
        regionID = 2
    elif card == 'J♠':
        regionID = 3
    elif card == 'J♥':
        regionID = 4

    actionID = (regionID * 100) + positionID
    return actionID

def normalize(values, bounds):  
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]
    

# print(getPositions(5))
# print(decodePosition(14))
# print(decodePosition(97))
# print(decodeAction(301))
# test = encodeAction('Add','9♥',[7,1])
# print(decodeAction(test))

# testhand = ['5♦','8♣', 'A♠','2♥']
# numhand = [cardToNum(x) for x in testhand]
# print(numhand)

# norm = normalize(numhand,{'actual': {'lower': 0, 'upper': 51}, 'desired': {'lower': -1, 'upper': 1}})
# print(norm)

def countSpecialCards(testHand):
    return testHand.count('J♦') + testHand.count('J♣') + testHand.count('J♠') + testHand.count('J♥')  