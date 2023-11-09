
import random
import argparse
import codecs
import os
import numpy as np

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans) and emission (basename.emit) files."""

        # Load transitions
        trans_filename = basename + '.trans'
        with open(trans_filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if parts:
                    state = parts[0]
                    if state not in self.transitions:    
                        self.transitions[state] = {}
                    for i in range(1, len(parts), 2):
                        next_state = parts[i]
                        prob = float(parts[i+1])
                        self.transitions[state][next_state] = prob 

        # Load emissions
        emit_filename = basename + '.emit'
        with open(emit_filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if parts:
                    state = parts[0]
                    if state not in self.emissions:
                        self.emissions[state] = {}
                    for i in range(1, len(parts), 2):
                        output = parts[i]
                        prob = float(parts[i+1])
                        self.emissions[state][output] = prob  # Add each emission probability to the state key

   ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        current_state = '#'
        states_seq = []
        output_seq = []

        for _ in range(n):
            next_states, trans_probs = zip(*self.transitions[current_state].items())
            current_state = random.choices(next_states, weights=trans_probs, k=1)[0]
            states_seq.append(current_state)

            emissions, emit_probs = zip(*self.emissions[current_state].items())
            emission = random.choices(emissions, weights=emit_probs, k=1)[0]
            output_seq.append(emission)

        return Observation(stateseq=states_seq, outputseq=output_seq)

    def forward(self, observation):
        """Calculate the forward probability for each state given an observation."""
        fwd = {state: [0] * len(observation.outputseq) for state in self.transitions if state != '#'}

        for state in self.transitions['#']:
            if state in self.emissions:
                fwd[state][0] = self.transitions['#'][state] * self.emissions[state].get(observation.outputseq[0], 0)

        for t in range(1, len(observation.outputseq)):
            for next_state in fwd:
                fwd[next_state][t] = sum(
                    fwd[current_state][t-1] * self.transitions[current_state].get(next_state, 0) * self.emissions[next_state].get(observation.outputseq[t], 0)
                    for current_state in fwd
                )

        final_state, fs_prob = max(fwd.items(), key=lambda item: item[1][-1])
        # print(fwd)
        return final_state, fs_prob[-1]

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.
    def viterbi(self, observation):
        """Given an observation, find and return the most likely state sequence."""
        num_states = len(self.transitions) - 1
        num_time_steps = len(observation.outputseq)
        viterbi_mat = np.full((num_states, num_time_steps), -np.inf)
        backpointer_mat = np.full((num_states, num_time_steps), -1)
        
        states = [state for state in self.transitions if state != '#']
        state_to_index = {state: idx for idx, state in enumerate(states)}
        
        for state in states:
            if state in self.emissions and observation.outputseq[0] in self.emissions[state]:
                viterbi_mat[state_to_index[state], 0] = np.log(self.transitions['#'][state]) + np.log(self.emissions[state][observation.outputseq[0]])
        
        for t in range(1, num_time_steps):
            for next_state in states:
                for current_state in states:
                    idx_current = state_to_index[current_state]
                    idx_next = state_to_index[next_state]
                    trans_prob = self.transitions[current_state].get(next_state, 0)
                    emit_prob = self.emissions[next_state].get(observation.outputseq[t], 0)
                    if trans_prob > 0 and emit_prob > 0:
                        prob = viterbi_mat[idx_current, t-1] + np.log(trans_prob) + np.log(emit_prob)
                        if prob > viterbi_mat[idx_next, t]:
                            viterbi_mat[idx_next, t] = prob
                            backpointer_mat[idx_next, t] = idx_current
                
        final_state_index = np.argmax(viterbi_mat[:, -1])
        final_state = states[final_state_index]
        
        best_path = [final_state]
        for t in range(num_time_steps - 1, 0, -1):
            final_state_index = backpointer_mat[final_state_index, t]
            best_path.insert(0, states[final_state_index])
        
        return best_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMM generator')
    parser.add_argument('model_file', type=str, help='HMM model file without extension')
    parser.add_argument('--generate', type=int, help='Generate a sequence of specified length')
    parser.add_argument('--forward', type=str, help='Run the forward algorithm on the given observation file')
    parser.add_argument('--viterbi', type=str, help='Run the Viterbi algorithm on the given observation file')


    args = parser.parse_args()

    model = HMM()
    model.load(args.model_file)

    # if generate
    if args.generate:
        observation = model.generate(args.generate)
        print(observation)

    # if forward    
    if args.forward:
        with open(args.forward, 'r') as file:
            observation_seq = file.read().strip().split()
        observation = Observation(stateseq=[], outputseq=observation_seq)
        
        fwd_prob = model.forward(observation)
        print(f"The most likely final state's probability is: {fwd_prob}")

    # if viterbi
    if args.viterbi:
        with open(args.viterbi, 'r') as file:
            observation_seq = file.read().strip().split()
        observation = Observation(stateseq=[], outputseq=observation_seq)
        
        best_state_sequence = model.viterbi(observation)
        print('The best state sequence is:')
        print(' '.join(best_state_sequence))
