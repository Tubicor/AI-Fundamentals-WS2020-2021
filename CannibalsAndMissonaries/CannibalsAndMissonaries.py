class State():
    def __init__(self,mL,cL,mR,cR,boat,predecessor):
        self.mL =mL
        self.cL = cL
        self.mR = mR
        self.cR = cR
        self.boat = boat
        self.predecessor = predecessor
    def __repr__(self):
        return "mL:{} | mR:{}\ncL:{} | cR:{}\n boat: {}\n".format(self.mL,self.mR,self.cL,self.cR,self.boat)
    def __eq__(self,other):
        return self.mL == other.mL and self.mR == other.mR and self.cL == other.cL and self.cR == other.cR and self.boat == other.boat

def expand(state):
    newStates = []
    if state.boat =="left":
        #move one cannibal
        if(state.cL >= 1):
            newStates += [State(state.mL,state.cL-1,state.mR,state.cR+1,"right",state)]
        #move one missonary
        if(state.mL >= 1):
            newStates += [State(state.mL-1,state.cL,state.mR+1,state.cR,"right",state)]
        #move two cannibals
        if(state.cL >= 2):
            newStates += [State(state.mL,state.cL-2,state.mR,state.cR+2,"right",state)]
        #move two missonary
        if(state.mL >= 2):
            newStates += [State(state.mL-2,state.cL,state.mR+2,state.cR,"right",state)]
        #move one cannibal and one missonary
        if(state.mL >= 1 and state.cL >=1):
            newStates += [State(state.mL-1,state.cL-1,state.mR+1,state.cR+1,"right",state)]
    elif state.boat =="right":
        #move one cannibal
        if(state.cR >= 1):
            newStates += [State(state.mL,state.cL+1,state.mR,state.cR-1,"left",state)]
        #move one missonary
        if(state.mR >= 1):
            newStates += [State(state.mL+1,state.cL,state.mR-1,state.cR,"left",state)]
        #move two cannibals
        if(state.cR >= 2):
            newStates += [State(state.mL,state.cL+2,state.mR,state.cR-2,"left",state)]
        #move two missonary
        if(state.mR >= 2):
            newStates += [State(state.mL+2,state.cL,state.mR-2,state.cR,"left",state)]
        #move one cannibal and one missonary
        if(state.mR >= 1 and state.cR >=1):
            newStates += [State(state.mL+1,state.cL+1,state.mR-1,state.cR-1,"left",state)]
    return newStates;

def checkStates(newStates,visitedStates):
    #check after Rules: No more Cannibals on one side then Missonaries
    ruleConfirmStates = []
    for state in newStates:
        if(state.boat == "right" and (state.cL <= state.mL or state.mL == 0)):
            ruleConfirmStates.append(state)
        elif(state.boat == "left" and (state.cR <= state.mR or state.mR == 0)):
            ruleConfirmStates.append(state)
    #check if the states already occoured
    notVisitedStates = []
    for state in ruleConfirmStates:
        if not visitedStates.__contains__(state):
            notVisitedStates.append(state)
    return notVisitedStates

def main():
    start = State(3,3,0,0,"left",None)
    visitedStates = []
    visitedStates.append(start)
    iterator = 0
    while True:
        expandedStates = expand(visitedStates[iterator])
        visitedStates.extend(checkStates(expandedStates,visitedStates))
        iterator += 1
        if(visitedStates.__contains__(State(0,0,3,3,"right",None))):
            break
    finishState = visitedStates[visitedStates.index(State(0,0,3,3,"right",None))]
    iteratorState = finishState
    while True:
        print(iteratorState)
        iteratorState = iteratorState.predecessor
        if iteratorState is None:
            break

if __name__ == '__main__':
    main()
