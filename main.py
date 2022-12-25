import os
from train import model, data
from chat import *

def main():
    
    chat(model, data.words, data.labels)


if __name__ == "__main__":
    os.system('cls')
    main()