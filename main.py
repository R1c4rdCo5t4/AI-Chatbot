import os
from train import model, data
from chat import *

def main():
    with open("ascii.txt", "r", encoding='utf-8') as f:
        for line in f.read().splitlines():
            print(line)
    
    chat(model, data.words, data.labels)


if __name__ == "__main__":
    os.system('cls')
    main()