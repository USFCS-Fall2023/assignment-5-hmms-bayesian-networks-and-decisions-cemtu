import subprocess

def run_command(command):
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.stderr)

if __name__ == '__main__':
    print("Output for: python hmm.py partofspeech.browntags.trained --generate 20\n")
    run_command(['python', 'hmm.py', 'partofspeech.browntags.trained', '--generate', '20'])

    print("Output for: python hmm.py partofspeech.browntags.trained --forward ambiguous_sents.obs\n")
    run_command(['python', 'hmm.py', 'partofspeech.browntags.trained', '--forward', 'ambiguous_sents.obs'])

    print("Output for: python hmm.py partofspeech.browntags.trained --viterbi ambiguous_sents.obs\n")
    run_command(['python', 'hmm.py', 'partofspeech.browntags.trained', '--viterbi', 'ambiguous_sents.obs'])

    print("Output for python alarm.py:")
    run_command(['python', 'alarm.py'])

    print("Output for python carnet.py:")
    run_command(['python', 'carnet.py'])

