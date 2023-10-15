# dask-holdem-simulator
An extensible Texas Hold'Em Simulator that can run at scale and use GPUs for calculations

# Introduction and Motivation

About 20 years ago I learned TH at a startup I was working for. 

I got it in my brain that I could make a bot, that would automatically play. 
In that process, long before Sikuli and OpenCV. I wrote a brute force image recognition
system, and handled simulation of movement with bezier curve calculations and modeled
the game state, playing, cards, and rules with Java. I spent all my free time that I could thinking about it for about 5 years.  I never could figure out how to build an AI for it, and how to appropriately handle the decision-making. It was however my first really non-trivial foray into complex systems running on multiple CPUs and with multi-threading. 

Flash forward to 2023 and I’m taking GPU Specialization learning how to program CUDA.
In 2022 I got to experiment with big data and redundant systems and got a taste for redundant
messaging and computation systems (Apache Kafka, Apache Spark respectively). Here I am with
the idea of “what should I do for my capstone”, and I started to think “OK, build a modular complex system simulation of TH on a system like Kafka, but not so heavyweight, that also has
support for the GPU — and try to figure out where the GPU would actually be useful.”

The pie-in-the-sky-dream is to have thousands of instances of bots running on the GPUs, 
logging the data, and learning from this behavior, collecting the data back and coming up
with some interesting observations. 

I spent a day researching some of the things I could do and settled on Dask. 
https://github.com/dask/dask


Given my track record of taking on rather thorny multi-year projects and not completing them I figured I’d give myself hackathon conditions and see what I could come up with as a best-effort. 

This work is the result of these efforts. 

# Specific research areas that I considered for this project but abandoned for one reason or another:
 
- Loading in historical data of lots of games and trying to process them (found out about the IRC database, but didn’t see a standard format for game moves ; didn’t know if I wanted to spend the time to make a parser for it)
- https://poker.cs.ualberta.ca/irc_poker_database.html

- An unsupervised learning system that plays-to-learn. 
- Learning about OpenCL and Metal and how to convert to-from them so that I could test the CUDA code performance on the Apple Silicon Macs. 
- I started down the road of attempting to parallelize the Treys Evaluator on the GPU but decided against it because it seemed too complex. 

I started implementing these based on the algorithms presented here, but was not clear
that I would see any meaningful speedups
https://en.wikipedia.org/wiki/Effective_hand_strength_algorithm

- Calculate Hand Rank
- Calculate Hand RankValue
- Hand Strength
- Hand Potential

# Technology that I decided on using
https://github.com/dask/dask

A Python framework for poker related operations.
https://github.com/pokerregion/poker
https://poker.readthedocs.io/en/latest/index.html

Treys - A pure Python poker hand evaluation library
https://github.com/ihendley/treys


PyTorch -  a Python package that provides two high-level features
* Tensor computation (like NumPy) with strong GPU acceleration
* Deep neural networks built on a tape-based autograd system
https://github.com/pytorch/pytorch

https://github.com/cupy/cupy

CuPy -  a NumPy/SciPy-compatible array library for GPU-accelerated computing with Python. CuPy acts as a drop-in replacement to run existing NumPy/SciPy code on NVIDIA CUDA or AMD ROCm platforms.

https://github.com/cupy/cupy

numba - A Just-In-Time Compiler for Numerical Functions in Python
https://github.com/numba/numba


# Architecture
- Dask scheduler
- Gameboard - An object that represents the state of the game, and the board, and deals cards out to the players. 
- Player objects that are launched from the command line that have a Pub/Sub connection with the GameBoard. 
- Evaluator
- Utils and conversion methods that convert Card data back and forth between Treys and Pokerlib type Cards
- Hand evaluator that uses and transforms a modified version of the results of Treys to make basic Player move decisions. 


# What is actually needed to use the code: 
```pip3 install "dask[complete] poker treys torch```

# How To Run the Simulator (in separate terminals)

`dask scheduler`
`python GameBoard.py` # Launch the game board and wait for players 30 second retry
`python3 Player.py 1`      # Begin with the first player - Min 2
`python3 Player.py 2`
`python3 Player.py ..`
`python3 Player.py 22` # Players > 22 are ignored, and the game will begin immediately.

What is currently running on the GPU?
- Well the non-functional hand strength and hand potential code seem to peg the GPU and hang.
